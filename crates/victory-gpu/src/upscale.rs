//! Host GPU spatial bilinear upscaler used by console upscaler HLE paths
//! (for example when a title's console upscaler DLL embeds console-only
//! shader containers without DXIL).

use crate::object::Object;

/// Records a bilinear upscale pass on `list` sampling `[0..render_size]` of
/// `color` and writing `[0..upscale_size]` of `output`. Zero dimensions fall
/// back to each texture's full size. Returns `true` if the pass was recorded.
#[cfg(windows)]
pub fn dispatch(
    list: *const Object,
    color: *const Object,
    output: *const Object,
    render_size: (u32, u32),
    upscale_size: (u32, u32),
) -> bool {
    imp::dispatch(list, color, output, render_size, upscale_size)
}

#[cfg(not(windows))]
pub fn dispatch(
    _list: *const Object,
    _color: *const Object,
    _output: *const Object,
    _render_size: (u32, u32),
    _upscale_size: (u32, u32),
) -> bool {
    false
}

#[cfg(windows)]
mod imp {
    use std::sync::OnceLock;
    use std::sync::atomic::{AtomicU32, Ordering};

    use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
    use windows::Win32::Graphics::Direct3D::ID3DBlob;
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;
    use windows::core::s;

    use crate::host::host;
    use crate::object::This;

    const PAIRS: u32 = 128;

    const SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=4, b0), DescriptorTable(SRV(t0), UAV(u0)), StaticSampler(s0, filter=FILTER_MIN_MAG_LINEAR_MIP_POINT, addressU=TEXTURE_ADDRESS_CLAMP, addressV=TEXTURE_ADDRESS_CLAMP, addressW=TEXTURE_ADDRESS_CLAMP)"
cbuffer Params : register(b0) {
    uint outW;
    uint outH;
    float uvScaleX;
    float uvScaleY;
};
Texture2D<float4> src : register(t0);
RWTexture2D<float4> dst : register(u0);
SamplerState samp : register(s0);

[RootSignature(RS)]
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= outW || id.y >= outH) return;
    float2 uv = ((float2(id.xy) + 0.5) / float2(outW, outH)) * float2(uvScaleX, uvScaleY);
    dst[id.xy] = src.SampleLevel(samp, uv, 0);
}
"#;

    struct Upscaler {
        root: ID3D12RootSignature,
        pso: ID3D12PipelineState,
        heap: ID3D12DescriptorHeap,
        step: u32,
        next: AtomicU32,
    }

    // SAFETY: D3D12 objects are free-threaded; descriptor slots rotate through
    // a ring of `PAIRS` entries.
    unsafe impl Send for Upscaler {}
    // SAFETY: as above.
    unsafe impl Sync for Upscaler {}

    fn upscaler() -> Option<&'static Upscaler> {
        static PASS: OnceLock<Option<Upscaler>> = OnceLock::new();
        PASS.get_or_init(|| match make() {
            Ok(u) => Some(u),
            Err(e) => {
                tracing::warn!(error = %e, "bilinear upscaler not created");
                None
            }
        })
        .as_ref()
    }

    fn make() -> windows::core::Result<Upscaler> {
        let device = &host().ok_or_else(windows::core::Error::empty)?.device;
        let mut code: Option<ID3DBlob> = None;
        let mut errors: Option<ID3DBlob> = None;
        // SAFETY: static shader source and owned out pointers.
        let compiled = unsafe {
            D3DCompile(
                SOURCE.as_ptr().cast(),
                SOURCE.len(),
                s!("upscale"),
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
                // SAFETY: valid error blob.
                let text = unsafe {
                    std::slice::from_raw_parts(
                        errors.GetBufferPointer().cast::<u8>(),
                        errors.GetBufferSize(),
                    )
                };
                tracing::warn!("bilinear upscaler: {}", String::from_utf8_lossy(text));
            }
            return Err(e);
        }
        let code = code.ok_or_else(windows::core::Error::empty)?;
        // SAFETY: valid shader bytecode and plain D3D12 object creation.
        unsafe {
            let bytes = std::slice::from_raw_parts(
                code.GetBufferPointer().cast::<u8>(),
                code.GetBufferSize(),
            );
            let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes)?;
            let pso = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: bytes.as_ptr().cast(),
                    BytecodeLength: bytes.len(),
                },
                ..Default::default()
            })?;
            let heap: ID3D12DescriptorHeap =
                device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    NumDescriptors: PAIRS * 2,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    NodeMask: 0,
                })?;
            let step = device
                .GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            Ok(Upscaler {
                root,
                pso,
                heap,
                step,
                next: AtomicU32::new(0),
            })
        }
    }

    fn uav_barrier() -> D3D12_RESOURCE_BARRIER {
        D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                    pResource: std::mem::ManuallyDrop::new(None),
                }),
            },
        }
    }

    fn srv_format(resource: This, desc: &D3D12_RESOURCE_DESC) -> DXGI_FORMAT {
        match crate::resource::view_format(resource) {
            Some(f) => DXGI_FORMAT(f as i32),
            None => desc.Format,
        }
    }

    fn uav_format(resource: This, desc: &D3D12_RESOURCE_DESC) -> DXGI_FORMAT {
        let mut f = match crate::resource::view_format(resource) {
            Some(f) => f,
            None => desc.Format.0 as u32,
        };
        if let Some(&(_, linear)) = crate::resource::SRGB.iter().find(|&&(s, _)| s == f) {
            f = linear;
        }
        DXGI_FORMAT(f as i32)
    }

    pub(super) fn dispatch(
        list: This,
        color: This,
        output: This,
        render_size: (u32, u32),
        upscale_size: (u32, u32),
    ) -> bool {
        if list.is_null() || color.is_null() || output.is_null() {
            return false;
        }
        if color == output {
            return true;
        }
        let (Some(host), Some(pass), Some(host_list)) =
            (host(), upscaler(), crate::list::host_list(list))
        else {
            return false;
        };
        let (Some(src), Some(dst)) = (
            crate::resource::host_resource(color),
            crate::resource::host_resource(output),
        ) else {
            return false;
        };
        // SAFETY: both host textures are live.
        let (src_desc, dst_desc) = unsafe { (src.GetDesc(), dst.GetDesc()) };
        if src_desc.Flags.0 & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE.0 != 0
            || dst_desc.Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 == 0
        {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                tracing::warn!(
                    src_flags = src_desc.Flags.0,
                    dst_flags = dst_desc.Flags.0,
                    "bilinear upscaler skipped: resource flags forbid SRV or UAV"
                );
            });
            return false;
        }
        let src_w = (src_desc.Width as u32).max(1);
        let src_h = src_desc.Height.max(1);
        let dst_w = (dst_desc.Width as u32).max(1);
        let dst_h = dst_desc.Height.max(1);
        let render_w = if render_size.0 > 0 {
            render_size.0.min(src_w)
        } else {
            src_w
        };
        let render_h = if render_size.1 > 0 {
            render_size.1.min(src_h)
        } else {
            src_h
        };
        let out_w = if upscale_size.0 > 0 {
            upscale_size.0.min(dst_w)
        } else {
            dst_w
        };
        let out_h = if upscale_size.1 > 0 {
            upscale_size.1.min(dst_h)
        } else {
            dst_h
        };

        if crate::texture::title_written(src) {
            crate::texture::upload(host_list, color, src);
        }
        const SHADER_RESOURCE: u32 = 0x40 | 0x80;
        const UNORDERED_ACCESS: u32 = 0x8;
        crate::list::ensure_state(host_list, color, SHADER_RESOURCE);
        crate::list::ensure_state(host_list, output, UNORDERED_ACCESS);
        crate::texture::host_written(output);
        #[cfg(feature = "diagnostics")]
        {
            crate::capture::note_upscale();
            crate::capture::note_target(color);
            crate::capture::note_target(output);
        }

        let pair = pass.next.fetch_add(1, Ordering::Relaxed) % PAIRS;
        let base_slot = pair * 2;
        let srv_desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: srv_format(color, &src_desc),
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_SRV {
                    MipLevels: 1,
                    ..Default::default()
                },
            },
        };
        let uav_desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: uav_format(output, &dst_desc),
            ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_UAV {
                    MipSlice: 0,
                    PlaneSlice: 0,
                },
            },
        };
        let constants = [
            out_w,
            out_h,
            (render_w as f32 / src_w as f32).to_bits(),
            (render_h as f32 / src_h as f32).to_bits(),
        ];

        // SAFETY: an open command list, live host textures, and valid slots in
        // our shader-visible descriptor heap.
        unsafe {
            let cpu_start = pass.heap.GetCPUDescriptorHandleForHeapStart().ptr;
            let gpu_start = pass.heap.GetGPUDescriptorHandleForHeapStart().ptr;
            let srv_cpu = D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: cpu_start + (base_slot * pass.step) as usize,
            };
            let uav_cpu = D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: cpu_start + ((base_slot + 1) * pass.step) as usize,
            };
            let table_gpu = D3D12_GPU_DESCRIPTOR_HANDLE {
                ptr: gpu_start + u64::from(base_slot) * u64::from(pass.step),
            };
            host.device
                .CreateShaderResourceView(src, Some(&srv_desc), srv_cpu);
            host.device.CreateUnorderedAccessView(
                dst,
                None::<&ID3D12Resource>,
                Some(&uav_desc),
                uav_cpu,
            );
            let l = &host_list.list;
            l.ResourceBarrier(&[uav_barrier()]);
            l.SetDescriptorHeaps(&[Some(pass.heap.clone())]);
            l.SetComputeRootSignature(&pass.root);
            l.SetPipelineState(&pass.pso);
            l.SetComputeRoot32BitConstants(0, 4, constants.as_ptr().cast(), 0);
            l.SetComputeRootDescriptorTable(1, table_gpu);
            l.Dispatch(out_w.div_ceil(8), out_h.div_ceil(8), 1);
            l.ResourceBarrier(&[uav_barrier()]);
        }
        crate::list::invalidate_host(list);
        true
    }
}

#[cfg(all(test, windows))]
mod tests {
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::*;

    use crate::object::{S_OK, This, slot};

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct HeapProperties {
        kind: u32,
        cpu_pages: u32,
        pool: u32,
        creation_nodes: u32,
        visible_nodes: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct ResourceDesc {
        dimension: u32,
        alignment: u64,
        width: u64,
        height: u32,
        depth_or_array_size: u16,
        mip_levels: u16,
        format: u32,
        sample_count: u32,
        sample_quality: u32,
        layout: u32,
        flags: u32,
    }

    fn f16_bits(v: f32) -> u16 {
        // Exact for small normal values used in the test (e.g. 0.5, 1.0, 2.0).
        let bits = v.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
        let mant = ((bits >> 13) & 0x3ff) as u16;
        if exp <= 0 {
            sign
        } else {
            sign | ((exp as u16) << 10) | mant
        }
    }

    #[test]
    fn promoted_console_targets_create_and_upscale_cleanly() {
        let Some(host) = crate::host::host() else {
            return;
        };
        let device = crate::device::device();
        let null: This = std::ptr::null();
        let null_void = std::ptr::null::<std::ffi::c_void>();
        let heap = HeapProperties {
            kind: 1,
            cpu_pages: 0,
            pool: 0,
            creation_nodes: 1,
            visible_nodes: 1,
        };
        let tex_desc = |w: u64, h: u32, format: u32, flags: u32| ResourceDesc {
            dimension: 3,
            alignment: 0,
            width: w,
            height: h,
            depth_or_array_size: 1,
            mip_levels: 1,
            format,
            sample_count: 1,
            sample_quality: 0,
            layout: 0,
            flags,
        };
        type CreateCommitted = extern "system" fn(
            This,
            *const HeapProperties,
            u32,
            *const ResourceDesc,
            u32,
            *const std::ffi::c_void,
            *const crate::object::Iid,
            *mut This,
        ) -> crate::object::HRESULT;
        let create = slot::<CreateCommitted>(device, 0xd8);
        let iid = crate::object::guid(
            0x696442be,
            0xa72e,
            0x4059,
            [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad],
        );

        // Verify all console target formats that PC forbids directly now get
        // promoted host copies.
        for (w, h, fmt, flags, expected) in [
            (16u64, 16u32, 67u32, 5u32, DXGI_FORMAT_R16G16B16A16_FLOAT),
            (16, 16, 67, 4, DXGI_FORMAT_R16G16B16A16_FLOAT),
            (16, 16, 86, 5, DXGI_FORMAT_R8G8B8A8_UNORM),
            (16, 16, 115, 1, DXGI_FORMAT_R8G8B8A8_UNORM),
        ] {
            let desc = tex_desc(w, h, fmt, flags);
            let mut r = null;
            assert_eq!(create(device, &heap, 0, &desc, 0, null_void, &iid, &mut r), S_OK);
            let host_tex = crate::resource::host_resource(r).expect("host texture promoted");
            // SAFETY: live host resource.
            assert_eq!(unsafe { host_tex.GetDesc() }.Format, expected);
            slot::<extern "system" fn(This) -> u32>(r, 0x10)(r);
        }

        // Create a 4x4 RGBA16F source (RTV) and an 8x8 format-67 UAV output,
        // clear the source to bright orange (2.0, 0.5, 0.25, 1.0), upscale it,
        // and verify the readback pixels match.
        let mut src = null;
        let mut dst = null;
        let src_d = tex_desc(4, 4, 10, 1);
        let dst_d = tex_desc(8, 8, 67, 5);
        assert_eq!(create(device, &heap, 0, &src_d, 0, null_void, &iid, &mut src), S_OK);
        assert_eq!(create(device, &heap, 0, &dst_d, 0, null_void, &iid, &mut dst), S_OK);

        let mut allocator = null;
        let mut list = null;
        let alloc_iid = crate::object::guid(
            0x6102dee4,
            0xaf59,
            0x4b09,
            [0xb9, 0x99, 0xb4, 0x4d, 0x73, 0xf0, 0x9b, 0x24],
        );
        let list_iid = crate::object::guid(
            0x5b160d0f,
            0xac1b,
            0x4185,
            [0x8b, 0xa8, 0xb3, 0xae, 0x42, 0xa5, 0xa4, 0x55],
        );
        assert_eq!(
            crate::list::create_allocator(device, 0, &alloc_iid, &mut allocator),
            S_OK
        );
        assert_eq!(
            crate::list::create_list(device, 0, 0, allocator, null, &list_iid, &mut list),
            S_OK
        );

        let mut rtv_slot = [0u64; 16];
        let rtv_handle = rtv_slot.as_mut_ptr() as u64;
        crate::descriptor::create_rtv(device, src, std::ptr::null(), rtv_handle);
        let clear_color = [2.0f32, 0.5, 0.25, 1.0];
        slot::<extern "system" fn(This, u64, *const [f32; 4], u32, *const std::ffi::c_void)>(
            list, 0x1a0,
        )(list, rtv_handle, &clear_color, 0, null_void);

        assert!(super::dispatch(list, src, dst, (4, 4), (8, 8)));

        let dst_tex = crate::resource::host_resource(dst).unwrap();
        let host_list = crate::list::host_list(list).unwrap();
        crate::list::ensure_state(host_list, dst, 0x800); // COPY_SOURCE
        let mut fp = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
        let mut total = 0u64;
        // SAFETY: live device and destination texture.
        unsafe {
            let desc = dst_tex.GetDesc();
            host.device
                .GetCopyableFootprints(&desc, 0, 1, 0, Some(&mut fp), None, None, Some(&mut total));
        }
        let readback_heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_READBACK,
            ..Default::default()
        };
        let readback_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: total,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut readback: Option<ID3D12Resource> = None;
        // SAFETY: plain readback creation and copy on an open list.
        unsafe {
            host.device
                .CreateCommittedResource(
                    &readback_heap,
                    D3D12_HEAP_FLAG_NONE,
                    &readback_desc,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut readback,
                )
                .unwrap();
            let rb = readback.as_ref().unwrap();
            let to = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(rb),
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: fp },
            };
            let from = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(dst_tex),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    SubresourceIndex: 0,
                },
            };
            host_list
                .list
                .CopyTextureRegion(&to, 0, 0, 0, &from, None);
        }
        assert_eq!(slot::<extern "system" fn(This) -> crate::object::HRESULT>(list, 0x48)(list), S_OK);
        host.execute(&[host_list]);

        let rb = readback.unwrap();
        let mut ptr = std::ptr::null_mut();
        // SAFETY: execution completed; read back center pixel of 8x8 RGBA16F.
        unsafe {
            rb.Map(0, None, Some(&mut ptr)).unwrap();
            let row = (ptr as *const u8).add(4 * fp.Footprint.RowPitch as usize);
            let px = std::slice::from_raw_parts(row.add(4 * 8).cast::<u16>(), 4);
            assert_eq!(px[0], f16_bits(2.0));
            assert_eq!(px[1], f16_bits(0.5));
            assert_eq!(px[2], f16_bits(0.25));
            assert_eq!(px[3], f16_bits(1.0));
            rb.Unmap(0, None);
            assert!(host.device.GetDeviceRemovedReason().is_ok());
        }

        let release = |o: This| slot::<extern "system" fn(This) -> u32>(o, 0x10)(o);
        release(list);
        release(allocator);
        release(src);
        release(dst);
    }
}
