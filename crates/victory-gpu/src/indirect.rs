//! Indirect execution's arguments, bounded before the GPU reads them. The
//! title's own compute passes write most of them; a pass we drop or
//! translate wrongly leaves whatever the memory held, and a draw of four
//! billion vertices or a dispatch past the host's limits runs until the
//! host resets the display driver. So every record is clamped on the GPU,
//! just before its execution, by a compute shader of ours (HLSL compiled by
//! the system's compiler; a 5.1 shader with its root signature inside).
//!
//! Each list also has a budget (2^27 vertex instances or threads, groups
//! counted as 64 threads, mesh groups as 128): records past it run empty,
//! so a list of thousands of garbage records cannot run for seconds
//! either.
//!
//! shortcut: the bounds (2^24 vertices, 2^26 vertex instances, 65,535
//! groups per dimension and 2^20 per dispatch, one-dimensional mesh
//! dispatches, the budget) cut legitimate work beyond them; raise them if
//! the title needs more.

use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::s;

use crate::host::{HostList, host, transition};

const SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=4, b0), UAV(u0), UAV(u1)"
cbuffer Record : register(b0) { uint kind; uint stride; uint records; uint unused; };
RWByteAddressBuffer args : register(u0);
// The list's work so far (vertex instances, threads): past the budget,
// records run empty.
RWByteAddressBuffer budget : register(u1);
static const uint BUDGET = 1u << 27;
// Groups of one dispatch; more run empty.
static const uint GROUPS = 1u << 20;
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= records) return;
    uint at = id.x * stride;
    uint cost;
    if (kind <= 1) {
        // Draws: vertex or index count, then instances.
        uint count = min(args.Load(at), 1u << 24);
        uint instances = min(args.Load(at + 4), (1u << 26) / max(count, 1u));
        cost = count * instances;
        args.Store2(at, uint2(count, instances));
    } else if (kind == 2) {
        uint3 g = min(args.Load3(at), uint3(65535, 65535, 65535));
        // x * y fits 32 bits; the whole count may not.
        if (g.z != 0 && g.x * g.y > GROUPS / g.z) {
            g = uint3(0, 0, 0);
        }
        cost = g.x * g.y * g.z * 64;
        args.Store3(at, g);
    } else {
        // Mesh dispatches: the console's are one-dimensional.
        uint3 g = args.Load3(at);
        g = uint3(min(g.x, 65535u), min(g.y, 1u), min(g.z, 1u));
        cost = g.x * g.y * g.z * 128;
        args.Store3(at, g);
    }
    uint before;
    budget.InterlockedAdd(0, cost, before);
    if (before >= BUDGET || before + cost > BUDGET) {
        // Draws keep their layout with no vertices; groups become zero.
        args.Store(at, 0);
    }
}
"#;

const RECT_SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=4, b0), UAV(u0), UAV(u1), UAV(u2)"
cbuffer Record : register(b0) { uint stride; uint records; uint max_verts; uint unused; };
RWByteAddressBuffer src : register(u0);
RWByteAddressBuffer dst : register(u1);
RWByteAddressBuffer budget : register(u2);
static const uint BUDGET = 1u << 27;
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= records) return;
    uint at = id.x * stride;
    uint orig_count = src.Load(at);
    uint orig_inst = src.Load(at + 4);
    uint orig_start = src.Load(at + 8);
    uint orig_start_inst = src.Load(at + 12);
    // Clamp in rectangles before scaling 3 -> 6 so garbage cannot wrap 32 bits.
    uint max_rects = max_verts / 6u;
    uint start_rect = min(orig_start / 3u, max_rects);
    uint rects = min(orig_count / 3u, max_rects - start_rect);
    uint count = rects * 6u;
    uint start = start_rect * 6u;
    uint instances = min(orig_inst, (1u << 26) / max(count, 1u));
    uint cost = count * instances;
    uint before;
    budget.InterlockedAdd(0, cost, before);
    if (before >= BUDGET || before + cost > BUDGET) {
        count = 0;
    }
    dst.Store4(at, uint4(count, instances, start, orig_start_inst));
    for (uint off = 16; off < stride; off += 4) {
        dst.Store(at + off, src.Load(at + off));
    }
}
"#;

struct Clamp {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
}

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Clamp {}
// SAFETY: as above.
unsafe impl Sync for Clamp {}

fn clamp() -> Option<&'static Clamp> {
    static CLAMP: std::sync::OnceLock<Option<Clamp>> = std::sync::OnceLock::new();
    CLAMP
        .get_or_init(|| match make(SOURCE, s!("clamp")) {
            Ok(c) => Some(c),
            Err(e) => {
                tracing::warn!(error = %e, "no argument clamp: indirect execution dropped");
                None
            }
        })
        .as_ref()
}

fn rect_clamp() -> Option<&'static Clamp> {
    static CLAMP: std::sync::OnceLock<Option<Clamp>> = std::sync::OnceLock::new();
    CLAMP
        .get_or_init(|| match make(RECT_SOURCE, s!("rect_clamp")) {
            Ok(c) => Some(c),
            Err(e) => {
                tracing::warn!(error = %e, "no rectangle argument clamp: indirect rectangles dropped");
                None
            }
        })
        .as_ref()
}

fn make(source: &str, name: windows::core::PCSTR) -> windows::core::Result<Clamp> {
    let device = &host().ok_or_else(windows::core::Error::empty)?.device;
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: the source and names live through the call; out pointers ours.
    let compiled = unsafe {
        D3DCompile(
            source.as_ptr().cast(),
            source.len(),
            name,
            None,
            None,
            s!("main"),
            s!("cs_5_1"),
            0,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    if let Err(e) = compiled {
        if let Some(errors) = errors {
            // SAFETY: the blob's own bytes.
            let text = unsafe {
                std::slice::from_raw_parts(errors.GetBufferPointer().cast::<u8>(), errors.GetBufferSize())
            };
            tracing::warn!("argument clamp: {}", String::from_utf8_lossy(text));
        }
        return Err(e);
    }
    let code = code.ok_or_else(windows::core::Error::empty)?;
    // SAFETY: the blob's own bytes; the root signature is inside the shader.
    unsafe {
        let bytes = std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
        let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes)?;
        let pso = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&root),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: bytes.as_ptr().cast(),
                BytecodeLength: bytes.len(),
            },
            ..Default::default()
        })?;
        Ok(Clamp { root, pso })
    }
}

/// Records, in `list`, the clamp of `records` records of `kind` (PC
/// argument numbering) `stride` bytes apart at host address `gpu` in
/// `buffer`, and leaves the buffer ready to be read as arguments. Replaces
/// the list's compute root signature and pipeline (callers bind theirs
/// after). False if the clamp is not available.
pub(crate) fn bound(
    list: &HostList,
    buffer: &ID3D12Resource,
    gpu: u64,
    (kind, stride, records): (u32, u32, u32),
) -> bool {
    let (Some(c), Some(budget)) = (clamp(), list.budget()) else {
        return false;
    };
    if !gpu.is_multiple_of(4) || !stride.is_multiple_of(4) {
        return false;
    }
    let l = &list.list;
    let constants = [kind, stride, records, 0];
    // SAFETY: an open list; our root signature, pipeline and a live buffer
    // the caller's list holds.
    unsafe {
        // Earlier shader writes (the title's argument passes) first.
        l.ResourceBarrier(&[uav_barrier()]);
        l.SetComputeRootSignature(&c.root);
        l.SetPipelineState(&c.pso);
        l.SetComputeRoot32BitConstants(0, 4, constants.as_ptr().cast(), 0);
        l.SetComputeRootUnorderedAccessView(1, gpu);
        l.SetComputeRootUnorderedAccessView(2, budget);
        l.Dispatch(records.div_ceil(64), 1, 1);
        l.ResourceBarrier(&[
            uav_barrier(),
            transition(
                buffer,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            ),
        ]);
    }
    true
}

/// Transforms and clamps `records` rectangle-list draw argument records
/// (`D3D12_DRAW_ARGUMENTS`: 3 vertices per rect -> 6 triangle vertices) from
/// `src_gpu` into a list-owned scratch buffer without mutating the title's
/// buffer, leaving the scratch buffer in `INDIRECT_ARGUMENT`. Call [`done`] on
/// the returned buffer after `ExecuteIndirect`.
pub(crate) fn bound_rects(
    list: &HostList,
    src_gpu: u64,
    (stride, records, max_verts): (u32, u32, u32),
) -> Option<(ID3D12Resource, u64)> {
    let (Some(c), Some(budget)) = (rect_clamp(), list.budget()) else {
        return None;
    };
    if stride < 16 || !src_gpu.is_multiple_of(4) || !stride.is_multiple_of(4) {
        return None;
    }
    let len = u64::from(stride) * u64::from(records.max(1));
    let (dst, dst_offset, dst_gpu) = list.indirect_scratch(len)?;
    let l = &list.list;
    let constants = [stride, records, max_verts, 0];
    // SAFETY: an open list; our root signature, pipeline and live buffers
    // the caller's list holds.
    unsafe {
        l.ResourceBarrier(&[uav_barrier()]);
        l.SetComputeRootSignature(&c.root);
        l.SetPipelineState(&c.pso);
        l.SetComputeRoot32BitConstants(0, 4, constants.as_ptr().cast(), 0);
        l.SetComputeRootUnorderedAccessView(1, src_gpu);
        l.SetComputeRootUnorderedAccessView(2, dst_gpu);
        l.SetComputeRootUnorderedAccessView(3, budget);
        l.Dispatch(records.div_ceil(64), 1, 1);
        l.ResourceBarrier(&[
            uav_barrier(),
            transition(
                &dst,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            ),
        ]);
    }
    Some((dst, dst_offset))
}

/// After the execution: the buffer back to unordered access, where the
/// title's passes keep using it.
pub(crate) fn done(list: &HostList, buffer: &ID3D12Resource) {
    // SAFETY: an open list and a live buffer.
    unsafe {
        list.list.ResourceBarrier(&[transition(
            buffer,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        )])
    };
}

fn uav_barrier() -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER { pResource: std::mem::ManuallyDrop::new(None) }),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The clamp compiles with the system's compiler and the host takes it.
    #[test]
    fn argument_clamp_builds() {
        if super::host().is_none() {
            return; // no host GPU here
        }
        assert!(super::clamp().is_some());
        assert!(super::rect_clamp().is_some());
    }

    /// Indirect rectangle argument expansion scales 3-vertex rects to 6-vertex
    /// triangles, bounds against `max_verts`, and leaves the source buffer intact.
    #[test]
    fn rect_clamp_expands_and_bounds_on_device() {
        let Some(host) = super::host() else {
            return;
        };
        let list = host.new_list().expect("host list");
        let (src, src_ptr) = host.shared_buffer(256).expect("src buffer");
        let (rb, rb_ptr) = host.shared_buffer(256).expect("readback buffer");
        let input: [[u32; 4]; 3] = [
            [3, 5, 0, 2],
            [9, 1, 3, 0],
            [u32::MAX, 100, 0, 0],
        ];
        // SAFETY: mapped 256-byte shared buffer, writing 48 bytes.
        unsafe {
            std::ptr::copy_nonoverlapping(input.as_ptr().cast::<u8>(), src_ptr, 48);
        }
        // SAFETY: live buffer.
        let src_gpu = unsafe { src.GetGPUVirtualAddress() };
        let (dst, dst_off) = bound_rects(&list, src_gpu, (16, 3, 24)).expect("bound_rects");
        // SAFETY: transition dst to COPY_SOURCE, copy to readback, restore UAV.
        unsafe {
            list.list.ResourceBarrier(&[transition(
                &dst,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            list.list.CopyBufferRegion(&rb, 0, &dst, dst_off, 48);
            list.list.ResourceBarrier(&[transition(
                &dst,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            )]);
            list.list.Close().expect("close");
        }
        host.execute(&[&list]);
        let mut out = [[0u32; 4]; 3];
        let mut src_after = [[0u32; 4]; 3];
        // SAFETY: submission finished, reading 48 bytes from mapped buffers.
        unsafe {
            std::ptr::copy_nonoverlapping(rb_ptr, out.as_mut_ptr().cast::<u8>(), 48);
            std::ptr::copy_nonoverlapping(src_ptr, src_after.as_mut_ptr().cast::<u8>(), 48);
        }
        assert_eq!(out, [[6, 5, 0, 2], [18, 1, 6, 0], [24, 100, 0, 0]]);
        assert_eq!(src_after, input);
    }
}
