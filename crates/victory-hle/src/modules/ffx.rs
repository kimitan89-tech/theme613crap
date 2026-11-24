//! `amd_fidelityfx_loader_xbox.dll`: FidelityFX API upscaler HLE shim.
//!
//! Console upscaler DLLs embed console-compiled shader blobs without DXIL
//! parts, so running the title DLL drops every upscaling compute pass and
//! leaves the upscaled scene color target black. Serving the FidelityFX loader
//! here replaces those passes with a host GPU bilinear upscale from the
//! rendered color buffer to the output target.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::ffi::{c_char, c_void};
use std::sync::atomic::{AtomicU64, Ordering};

use victory_gpu::object::Object;

use crate::{HleExport, HleModule};

type This = *const Object;

type FfxReturnCode = u32;
const FFX_API_RETURN_OK: FfxReturnCode = 0;
const FFX_API_RETURN_ERROR_PARAMETER: FfxReturnCode = 6;

const DESC_QUERY_GET_VERSIONS: u64 = 0x0000_0004;
const DESC_QUERY_GET_PROVIDER_VERSION: u64 = 0x0000_0006;
const DESC_CREATE_UPSCALE: u64 = 0x0001_0000;
const DESC_DISPATCH_UPSCALE: u64 = 0x0001_0001;
const DESC_QUERY_UPSCALE_RATIO: u64 = 0x0001_0002;
const DESC_QUERY_UPSCALE_RENDER_RESOLUTION: u64 = 0x0001_0003;
const DESC_QUERY_UPSCALE_JITTER_PHASE_COUNT: u64 = 0x0001_0004;
const DESC_QUERY_UPSCALE_JITTER_OFFSET: u64 = 0x0001_0005;
const DESC_DISPATCH_REACTIVE_MASK: u64 = 0x0001_0006;
const DESC_QUERY_UPSCALE_GPU_MEMORY_USAGE: u64 = 0x0001_0008;
const DESC_QUERY_UPSCALE_GPU_MEMORY_USAGE_V2: u64 = 0x0001_0009;

const VERSION_ID: u64 = 0x0004_0000_0000_0000;
const VERSION_NAME: &[u8] = b"4.0.0\0";

#[repr(C)]
struct FfxApiHeader {
    kind: u64,
    next: *const FfxApiHeader,
}

struct UpscalerContext {
    max_render: (u32, u32),
    max_upscale: (u32, u32),
    dispatches: AtomicU64,
}

fn upscale_ratio(quality_mode: u32) -> f32 {
    match quality_mode {
        0 => 1.0, // NativeAA
        1 => 1.5, // Quality
        2 => 1.7, // Balanced
        3 => 2.0, // Performance
        4 => 3.0, // UltraPerformance
        _ => 1.5,
    }
}

/// Reads an unaligned `T` at `byte_offset` inside `base`.
///
/// # Safety
/// `base + byte_offset` must be readable for `size_of::<T>()` bytes.
unsafe fn read_at<T: Copy>(base: *const u8, byte_offset: usize) -> T {
    // SAFETY: caller guarantees readability.
    unsafe { base.add(byte_offset).cast::<T>().read_unaligned() }
}

/// Finds the first header in the `pNext` chain whose `kind` matches `wanted`.
fn find_header(mut head: *const FfxApiHeader, wanted: u64) -> Option<*const u8> {
    let mut depth = 0;
    while !head.is_null() && depth < 16 {
        // SAFETY: title passes a valid `ffxApiHeader` chain.
        let hdr = unsafe { head.read_unaligned() };
        if hdr.kind == wanted {
            return Some(head.cast::<u8>());
        }
        head = hdr.next;
        depth += 1;
    }
    None
}

extern "system" fn ffx_create_context(
    context: *mut *mut c_void,
    desc: *const FfxApiHeader,
    _mem_cb: *const c_void,
) -> FfxReturnCode {
    if context.is_null() || desc.is_null() {
        return FFX_API_RETURN_ERROR_PARAMETER;
    }
    let (max_render, max_upscale) = match find_header(desc, DESC_CREATE_UPSCALE) {
        Some(base) => {
            // ffxCreateContextDescUpscale: +0x10 flags (u32), +0x14 maxRenderSize
            // (u32, u32), +0x1c maxUpscaleSize (u32, u32).
            // SAFETY: header kind verified as DESC_CREATE_UPSCALE.
            unsafe {
                (
                    (read_at::<u32>(base, 0x14), read_at::<u32>(base, 0x18)),
                    (read_at::<u32>(base, 0x1c), read_at::<u32>(base, 0x20)),
                )
            }
        }
        None => ((0, 0), (0, 0)),
    };
    tracing::info!(
        ?max_render,
        ?max_upscale,
        "ffxCreateContext: using host bilinear upscaler"
    );
    let state = Box::new(UpscalerContext {
        max_render,
        max_upscale,
        dispatches: AtomicU64::new(0),
    });
    // SAFETY: `context` was checked non-null.
    unsafe { context.write(Box::into_raw(state).cast()) };
    FFX_API_RETURN_OK
}

extern "system" fn ffx_destroy_context(
    context: *mut *mut c_void,
    _mem_cb: *const c_void,
) -> FfxReturnCode {
    if context.is_null() {
        return FFX_API_RETURN_ERROR_PARAMETER;
    }
    // SAFETY: `context` holds a handle returned by `ffx_create_context` or null.
    let ptr = unsafe { *context };
    if !ptr.is_null() {
        // SAFETY: created with `Box::into_raw` in `ffx_create_context`.
        let _ = unsafe { Box::from_raw(ptr.cast::<UpscalerContext>()) };
        // SAFETY: `context` was checked non-null.
        unsafe { context.write(std::ptr::null_mut()) };
    }
    FFX_API_RETURN_OK
}

extern "system" fn ffx_configure(
    _context: *mut *mut c_void,
    _desc: *const FfxApiHeader,
) -> FfxReturnCode {
    FFX_API_RETURN_OK
}

extern "system" fn ffx_query(
    _context: *mut *mut c_void,
    desc: *const FfxApiHeader,
) -> FfxReturnCode {
    if desc.is_null() {
        return FFX_API_RETURN_ERROR_PARAMETER;
    }
    // SAFETY: `desc` checked non-null.
    let kind = unsafe { desc.read_unaligned() }.kind;
    let base = desc.cast::<u8>();
    match kind {
        DESC_QUERY_GET_VERSIONS => {
            // +0x20 outputCount (*mut u64), +0x28 versionIds (*mut u64),
            // +0x30 versionNames (*mut *const c_char).
            // SAFETY: header kind verified as DESC_QUERY_GET_VERSIONS.
            unsafe {
                let count_ptr = read_at::<*mut u64>(base, 0x20);
                let ids_ptr = read_at::<*mut u64>(base, 0x28);
                let names_ptr = read_at::<*mut *const c_char>(base, 0x30);
                if !count_ptr.is_null() {
                    let cap = count_ptr.read_unaligned();
                    count_ptr.write_unaligned(1);
                    if cap >= 1 {
                        if !ids_ptr.is_null() {
                            ids_ptr.write_unaligned(VERSION_ID);
                        }
                        if !names_ptr.is_null() {
                            names_ptr.write_unaligned(VERSION_NAME.as_ptr().cast());
                        }
                    }
                }
            }
        }
        DESC_QUERY_GET_PROVIDER_VERSION => {
            // +0x10 versionId (u64), +0x18 versionName (*const c_char).
            // SAFETY: header kind verified as DESC_QUERY_GET_PROVIDER_VERSION.
            unsafe {
                let out = base.cast_mut();
                out.add(0x10).cast::<u64>().write_unaligned(VERSION_ID);
                out.add(0x18)
                    .cast::<*const c_char>()
                    .write_unaligned(VERSION_NAME.as_ptr().cast());
            }
        }
        DESC_QUERY_UPSCALE_RATIO => {
            // +0x10 qualityMode (u32), +0x18 pOutUpscaleRatio (*mut f32).
            // SAFETY: header kind verified as DESC_QUERY_UPSCALE_RATIO.
            unsafe {
                let mode = read_at::<u32>(base, 0x10);
                let out = read_at::<*mut f32>(base, 0x18);
                if !out.is_null() {
                    out.write_unaligned(upscale_ratio(mode));
                }
            }
        }
        DESC_QUERY_UPSCALE_RENDER_RESOLUTION => {
            // +0x10 displayWidth (u32), +0x14 displayHeight (u32), +0x18 qualityMode (u32),
            // +0x20 pOutRenderWidth (*mut u32), +0x28 pOutRenderHeight (*mut u32).
            // SAFETY: header kind verified as DESC_QUERY_UPSCALE_RENDER_RESOLUTION.
            unsafe {
                let dw = read_at::<u32>(base, 0x10);
                let dh = read_at::<u32>(base, 0x14);
                let mode = read_at::<u32>(base, 0x18);
                let ratio = upscale_ratio(mode);
                let rw = ((dw as f32 / ratio).round() as u32).max(1);
                let rh = ((dh as f32 / ratio).round() as u32).max(1);
                let out_w = read_at::<*mut u32>(base, 0x20);
                let out_h = read_at::<*mut u32>(base, 0x28);
                if !out_w.is_null() {
                    out_w.write_unaligned(rw);
                }
                if !out_h.is_null() {
                    out_h.write_unaligned(rh);
                }
            }
        }
        DESC_QUERY_UPSCALE_JITTER_PHASE_COUNT => {
            // +0x18 pOutPhaseCount (*mut i32): spatial bilinear needs no jitter.
            // SAFETY: header kind verified as DESC_QUERY_UPSCALE_JITTER_PHASE_COUNT.
            unsafe {
                let out = read_at::<*mut i32>(base, 0x18);
                if !out.is_null() {
                    out.write_unaligned(1);
                }
            }
        }
        DESC_QUERY_UPSCALE_JITTER_OFFSET => {
            // +0x18 pOutX (*mut f32), +0x20 pOutY (*mut f32).
            // SAFETY: header kind verified as DESC_QUERY_UPSCALE_JITTER_OFFSET.
            unsafe {
                let out_x = read_at::<*mut f32>(base, 0x18);
                let out_y = read_at::<*mut f32>(base, 0x20);
                if !out_x.is_null() {
                    out_x.write_unaligned(0.0);
                }
                if !out_y.is_null() {
                    out_y.write_unaligned(0.0);
                }
            }
        }
        DESC_QUERY_UPSCALE_GPU_MEMORY_USAGE => {
            // +0x10 pOutUsage (*mut [u64; 2]).
            // SAFETY: header kind verified as DESC_QUERY_UPSCALE_GPU_MEMORY_USAGE.
            unsafe {
                let out = read_at::<*mut [u64; 2]>(base, 0x10);
                if !out.is_null() {
                    out.write_unaligned([0, 0]);
                }
            }
        }
        DESC_QUERY_UPSCALE_GPU_MEMORY_USAGE_V2 => {
            // +0x30 pOutUsage (*mut [u64; 2]).
            // SAFETY: header kind verified as DESC_QUERY_UPSCALE_GPU_MEMORY_USAGE_V2.
            unsafe {
                let out = read_at::<*mut [u64; 2]>(base, 0x30);
                if !out.is_null() {
                    out.write_unaligned([0, 0]);
                }
            }
        }
        other => {
            tracing::debug!(kind = format_args!("{other:#x}"), "ffxQuery: unhandled query");
        }
    }
    FFX_API_RETURN_OK
}

extern "system" fn ffx_dispatch(
    context: *mut *mut c_void,
    desc: *const FfxApiHeader,
) -> FfxReturnCode {
    if context.is_null() || desc.is_null() {
        return FFX_API_RETURN_ERROR_PARAMETER;
    }
    // SAFETY: `desc` checked non-null.
    let kind = unsafe { desc.read_unaligned() }.kind;
    match kind {
        DESC_DISPATCH_UPSCALE => {
            let base = desc.cast::<u8>();
            // Verified layout of `ffxDispatchDescUpscale`:
            // +0x10 commandList, +0x18 color.resource, +0x138 output.resource,
            // +0x178 renderSize (u32, u32), +0x180 upscaleSize (u32, u32).
            // SAFETY: header kind verified as DESC_DISPATCH_UPSCALE.
            let (list, color, output, mut render_size, mut upscale_size) = unsafe {
                (
                    read_at::<This>(base, 0x10),
                    read_at::<This>(base, 0x18),
                    read_at::<This>(base, 0x138),
                    (read_at::<u32>(base, 0x178), read_at::<u32>(base, 0x17c)),
                    (read_at::<u32>(base, 0x180), read_at::<u32>(base, 0x184)),
                )
            };
            // SAFETY: `context` checked non-null; points to our context handle.
            let ctx_ptr = unsafe { *context }.cast::<UpscalerContext>();
            let count = if !ctx_ptr.is_null() {
                // SAFETY: valid `UpscalerContext` created by `ffx_create_context`.
                let ctx = unsafe { &*ctx_ptr };
                if render_size == (0, 0) {
                    render_size = ctx.max_render;
                }
                if upscale_size == (0, 0) {
                    upscale_size = ctx.max_upscale;
                }
                ctx.dispatches.fetch_add(1, Ordering::Relaxed)
            } else {
                0
            };
            let ok = victory_gpu::upscale::dispatch(list, color, output, render_size, upscale_size);
            if count == 0 {
                tracing::info!(
                    ?render_size,
                    ?upscale_size,
                    ?color,
                    ?output,
                    ok,
                    "ffxDispatch: first host bilinear upscale"
                );
            }
        }
        DESC_DISPATCH_REACTIVE_MASK => {}
        other => {
            tracing::debug!(
                kind = format_args!("{other:#x}"),
                "ffxDispatch: ignored dispatch kind"
            );
        }
    }
    FFX_API_RETURN_OK
}

pub(crate) fn module() -> HleModule {
    HleModule {
        name: "ffx",
        dlls: &["amd_fidelityfx_loader_xbox.dll"],
        exports: vec![
            HleExport {
                name: "ffxCreateContext",
                address: ffx_create_context as *const () as usize,
            },
            HleExport {
                name: "ffxDestroyContext",
                address: ffx_destroy_context as *const () as usize,
            },
            HleExport {
                name: "ffxConfigure",
                address: ffx_configure as *const () as usize,
            },
            HleExport {
                name: "ffxQuery",
                address: ffx_query as *const () as usize,
            },
            HleExport {
                name: "ffxDispatch",
                address: ffx_dispatch as *const () as usize,
            },
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_lifecycle_and_queries() {
        #[repr(C)]
        struct CreateDesc {
            header: FfxApiHeader,
            flags: u32,
            max_render: [u32; 2],
            max_upscale: [u32; 2],
        }
        let create = CreateDesc {
            header: FfxApiHeader {
                kind: DESC_CREATE_UPSCALE,
                next: std::ptr::null(),
            },
            flags: 0,
            max_render: [1280, 720],
            max_upscale: [1920, 1080],
        };
        let mut ctx = std::ptr::null_mut();
        assert_eq!(
            ffx_create_context(&mut ctx, &create.header, std::ptr::null()),
            FFX_API_RETURN_OK
        );
        assert!(!ctx.is_null());

        #[repr(C)]
        struct QueryRenderRes {
            header: FfxApiHeader,
            dw: u32,
            dh: u32,
            mode: u32,
            _pad: u32,
            out_w: *mut u32,
            out_h: *mut u32,
        }
        let (mut rw, mut rh) = (0u32, 0u32);
        let qres = QueryRenderRes {
            header: FfxApiHeader {
                kind: DESC_QUERY_UPSCALE_RENDER_RESOLUTION,
                next: std::ptr::null(),
            },
            dw: 1920,
            dh: 1080,
            mode: 1,
            _pad: 0,
            out_w: &mut rw,
            out_h: &mut rh,
        };
        assert_eq!(ffx_query(&mut ctx, &qres.header), FFX_API_RETURN_OK);
        assert_eq!((rw, rh), (1280, 720));

        #[repr(C)]
        struct QueryJitterCount {
            header: FfxApiHeader,
            rw: u32,
            dw: u32,
            out_count: *mut i32,
        }
        let mut phases = 0i32;
        let qj = QueryJitterCount {
            header: FfxApiHeader {
                kind: DESC_QUERY_UPSCALE_JITTER_PHASE_COUNT,
                next: std::ptr::null(),
            },
            rw: 1280,
            dw: 1920,
            out_count: &mut phases,
        };
        assert_eq!(ffx_query(&mut ctx, &qj.header), FFX_API_RETURN_OK);
        assert_eq!(phases, 1);

        assert_eq!(ffx_destroy_context(&mut ctx, std::ptr::null()), FFX_API_RETURN_OK);
        assert!(ctx.is_null());
    }
}
