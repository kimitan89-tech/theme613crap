//! Synthetic shaders that exercise lowering without a game or a GPU.

use super::lower::{Lowered, Parts, lower};
use crate::native::{
    ComputeIo, HwStage, MeshIo, NativeProgram, ResourceType, RootParam, RootType, StageIo,
    UsedResource,
};
use crate::{ShaderError, io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;
const LDS_STORE_V1: [u32; 2] = [0xD834_0000, 0x0000_0100]; // ds_write_b32 v0, v1

fn lower_words(
    words: &[u32],
    wave64: bool,
    mesh: bool,
    layout: Vec<RootParam>,
    used: Vec<UsedResource>,
) -> Result<Lowered, ShaderError> {
    lower_checked(words, wave64, mesh, layout, used, |_| Ok(()))
}

fn lower_checked(
    words: &[u32],
    wave64: bool,
    mesh: bool,
    layout: Vec<RootParam>,
    used: Vec<UsedResource>,
    check: impl FnOnce(&ir::Program) -> Result<(), ShaderError>,
) -> Result<Lowered, ShaderError> {
    let native = NativeProgram {
        stage: if mesh { HwStage::Ms } else { HwStage::Cs },
        wave64,
        vgprs: 256,
        io: if mesh {
            StageIo::Ms(MeshIo {
                outputs: vec![],
                clip_cntl: 0,
                threads: 128,
                max_vertices: 128,
                max_primitives: 128,
                topology: 2,
                lds_bytes: 512,
            })
        } else {
            StageIo::Cs(ComputeIo {
                rsrc2: 1 << 15 | layout.iter().map(|p| p.sgprs).sum::<u32>() << 1,
                threads: [128, 1, 1],
            })
        },
        layout,
        used,
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, wave64, native.vgprs)?;
    check(&p)?;
    let st = structure::structure(&p)?;
    let res = resource::track(&p, &native.layout, native.stage, Some(&st))?;
    let io = io::raise(&p, &native)?;
    lower(&Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: vec![],
        outputs: vec![],
        prims: vec![],
        mesh_stride: 0,
    })
}

#[test]
fn a_split_wave64_select_mask_stays_scalar_data() {
    let words = [
        0xBE80_0380, // s_mov_b32 s0, 0
        0xBE81_0381, // s_mov_b32 s1, 1 (only lane 32 is set)
        0xD501_0001,
        128 | (129 << 9), // v_cndmask_b32 v1, 0, 1, s[0:1]
        LDS_STORE_V1[0],
        LDS_STORE_V1[1],
        0x7E02_0201, // v_mov_b32 v1, s1 (also read the high dword as data)
        LDS_STORE_V1[0],
        LDS_STORE_V1[1],
        S_ENDPGM,
    ];
    for mesh in [false, true] {
        let shader = lower_words(&words, true, mesh, vec![], vec![]).unwrap();
        assert_eq!(shader.wave_ops, !mesh);
    }
}

#[test]
fn a_wave64_select_uses_the_rewritten_high_dword() {
    let words = [
        0x7D8E_0080, // v_cmp_t_u32 vcc, 0, v0 (both halves true)
        0xBEEB_0380, // s_mov_b32 vcc_hi, 0 (only the low half remains true)
        0x0202_0080, // v_cndmask_b32 v1, 0, v0, vcc (implicit wave64 VCC)
        LDS_STORE_V1[0],
        LDS_STORE_V1[1],
        S_ENDPGM,
    ];
    // Reading the two halves needs the lane index. Reusing the whole compare
    // predicate would select true for every lane and need no wave ops.
    assert!(
        lower_words(&words, true, false, vec![], vec![])
            .unwrap()
            .wave_ops
    );
}

#[test]
fn a_symbolic_select_mask_can_need_the_console_lane_index() {
    let words = [
        0x8780_817E, // s_and_b64 s[0:1], exec, 1 (only lane 0)
        0xD501_0001,
        128 | (129 << 9), // v_cndmask_b32 v1, 0, 1, s[0:1]
        LDS_STORE_V1[0],
        LDS_STORE_V1[1],
        S_ENDPGM,
    ];
    for mesh in [false, true] {
        let shader = lower_words(&words, true, mesh, vec![], vec![]).unwrap();
        assert_eq!(shader.wave_ops, !mesh);
    }
}

#[test]
fn a_wave64_carry_in_can_read_an_explicit_zero_mask() {
    let words = [
        0xD528_0201,
        129 | (129 << 9) | (125 << 18), // v_add_co_ci_u32 v1, s[2:3], 1, 1, null
        LDS_STORE_V1[0],
        LDS_STORE_V1[1],
        S_ENDPGM,
    ];
    assert!(
        !lower_words(&words, true, false, vec![], vec![])
            .unwrap()
            .wave_ops
    );
}

#[test]
fn indexed_buffer_operations_default_to_a_four_dword_stride() {
    let run = |words: &[u32], stride: Option<u32>| {
        let (layout, used) = buffer_table(stride);
        lower_words(words, false, false, layout, used)
    };
    for (op, use_result) in [
        (0xE030_2000, true),  // buffer_load_dword
        (0xE070_2000, false), // buffer_store_dword
        (0xE0C8_2000, false), // buffer_atomic_add
        (0xE070_0000, false), // raw store, without idxen
    ] {
        let mut words = vec![
            0xBE84_0300, // s_mov_b32 s4, s0
            0xBE85_0380, // s_mov_b32 s5, 0
            0xF408_0202,
            0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], null
            0x7E02_0281, // v_mov_b32 v1, 1
            op,
            0x8002_0100, // v1, v0, s[8:11], 0
        ];
        if use_result {
            words.extend(LDS_STORE_V1);
        }
        // Also read the descriptor's element count. getDimensions must use
        // the same structured view as the indexed access.
        words.extend([0x7E02_020A, LDS_STORE_V1[0], LDS_STORE_V1[1]]);
        words.push(S_ENDPGM);
        let default = run(&words, Some(0)).unwrap();
        let sixteen = run(&words, Some(16)).unwrap();
        let thirty_two = run(&words, Some(32)).unwrap();
        assert_eq!(default.dxil, sixteen.dxil, "opcode {op:#x}");
        assert_eq!(
            run(&words, None).unwrap().dxil,
            default.dxil,
            "opcode {op:#x}"
        );
        assert_eq!(default.dxil == thirty_two.dxil, op & 0x2000 == 0);
        if op & 0x2000 != 0 {
            assert!(run(&words, Some(6)).is_err(), "opcode {op:#x}");
        }
    }
}

fn buffer_table(stride: Option<u32>) -> (Vec<RootParam>, Vec<UsedResource>) {
    (
        vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 0,
            slot: 0,
        }],
        stride
            .map(|stride| UsedResource {
                kind: ResourceType::Srv,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride,
                dwords: 4,
                user_data: 0,
                slot: 0,
            })
            .into_iter()
            .collect(),
    )
}

/// A straight-line GFX10 excerpt ending at the instruction under test.
/// Keep its output live and check that the store reads that instruction's SSA definition.
fn lower_snippet(
    body: &[u32],
    result: u8,
    wave64: bool,
    mesh: bool,
    (layout, used): (Vec<RootParam>, Vec<UsedResource>),
) -> Result<Lowered, ShaderError> {
    let mut words = body.to_vec();
    words.extend([LDS_STORE_V1[0], u32::from(result) << 8, S_ENDPGM]);
    lower_checked(&words, wave64, mesh, layout, used, |p| {
        let [block] = p.blocks.as_slice() else {
            return Err(ShaderError::Malformed(
                "expected a straight-line diagnostic excerpt",
            ));
        };
        let [.., producer, sink, end] = block.insts.as_slice() else {
            return Err(ShaderError::Malformed(
                "missing diagnostic result instruction",
            ));
        };
        let reg = ir::Reg::V(result);
        let def = producer.write(reg).ok_or(ShaderError::Malformed(
            "last instruction must write the requested VGPR",
        ))?;
        assert_eq!(sink.name(), "ds_write_b32");
        assert_eq!(end.name(), "s_endpgm");
        assert_eq!(sink.read(reg), Some(def));
        Ok(())
    })
}

#[test]
fn log_mesh_split_mask_as_data() -> Result<(), ShaderError> {
    // TODO(from-log): Replace with both scalar writes, the select and the next numeric use.
    let body = [
        0xBE80_0380,
        0xBE81_0381, // s0 = 0, s1 = 1
        0xD501_0001,
        128 | (129 << 9), // v1 = select(0, 1, s[0:1])
        0x7E04_0201,      // v2 = s1
        0xD501_0003,
        257 | (258 << 9), // select v1 or v2 into v3 using s[0:1]
    ];
    let shader = lower_snippet(&body, 3, true, true, (vec![], vec![]))?;
    assert!(!shader.wave_ops);
    Ok(())
}

#[test]
fn log_compute_carry_chain() -> Result<(), ShaderError> {
    // TODO(from-log): Replace with the failing producer, carry register pair and consumer.
    let body = [
        0xD528_0201,
        129 | (129 << 9) | (125 << 18), // v1 and carry s[2:3]
        0xD501_0002,
        257 | (129 << 9) | (2 << 18), // select using that carry
    ];
    let shader = lower_snippet(&body, 2, true, false, (vec![], vec![]))?;
    assert!(!shader.wave_ops);
    Ok(())
}

#[test]
fn log_indexed_load_with_unknown_stride() -> Result<(), ShaderError> {
    // TODO(from-log): Replace with the descriptor setup, indexed load and reported resource fields.
    let body = [
        0xBE84_0300,
        0xBE85_0380, // descriptor table address
        0xF408_0202,
        0xFA00_0000, // load descriptor into s[8:11]
        0xE030_2004,
        0x8002_0100, // load v1 by index with a four-byte offset
    ];
    let run = |stride| lower_snippet(&body, 1, false, false, buffer_table(stride));
    let expected = run(Some(16))?;
    assert_eq!(run(Some(0))?.dxil, expected.dxil);
    assert_eq!(run(None)?.dxil, expected.dxil);
    assert_ne!(run(Some(32))?.dxil, expected.dxil);
    Ok(())
}

#[test]
fn non_active_mask_read_as_data_s_not_s_and_s_or() {
    const fn s_not_b32(sdst: u8, src0: u8) -> u32 {
        0xBE80_0000 | (sdst as u32) << 16 | 0x07 << 8 | src0 as u32
    }
    const fn s_not_b64(sdst: u8, src0: u8) -> u32 {
        0xBE80_0000 | (sdst as u32) << 16 | 0x08 << 8 | src0 as u32
    }
    const fn s_bcnt1_i32_b64(sdst: u8, src0: u8) -> u32 {
        0xBE80_0000 | (sdst as u32) << 16 | 0x10 << 8 | src0 as u32
    }
    const fn sop2(op: u8, sdst: u8, src0: u8, src1: u8) -> u32 {
        0x8000_0000
            | (op as u32 & 0x7F) << 23
            | (sdst as u32 & 0x7F) << 16
            | (src1 as u32) << 8
            | src0 as u32
    }
    const fn v_mov_b32(vdst: u8, src0: u16) -> u32 {
        0x7E00_0000 | (vdst as u32) << 17 | (0x001 << 9) | (src0 as u32 & 0x1FF)
    }
    const fn vop3(op: u16, vdst: u8, src0: u16, src1: u16, src2: u16) -> [u32; 2] {
        [
            0x35 << 26 | (op as u32 & 0x3FF) << 16 | vdst as u32,
            (src0 as u32 & 0x1FF) | (src1 as u32 & 0x1FF) << 9 | (src2 as u32 & 0x1FF) << 18,
        ]
    }
    let layout = vec![RootParam {
        root: RootType::Constants,
        table_offset_dwords: 0,
        sgprs: 4,
        base_register: 0,
        user_data: 0,
        slot: 0,
    }];

    // Wave64: active compare -> s_not_b64 -> read both low/high dwords and popcount as data.
    let w64_not = [
        v_mov_b32(0, 0),
        0x7D84_0080, // v_cmp_eq_u32 vcc, 0, v0
        s_not_b64(6, 106),
        v_mov_b32(1, 6),
        LDS_STORE_V1[0],
        LDS_STORE_V1[1],
        v_mov_b32(1, 7),
        LDS_STORE_V1[0],
        LDS_STORE_V1[1],
        s_bcnt1_i32_b64(8, 6),
        v_mov_b32(1, 8),
        LDS_STORE_V1[0],
        LDS_STORE_V1[1],
        S_ENDPGM,
    ];
    lower_words(&w64_not, true, false, layout.clone(), vec![])
        .expect("wave64 s_not_b64 over active compare read as data should lower");

    // Wave64: s_not_b64 + s_and_b64 + s_or_b64 over two active compares read as data.
    let mut w64_logic = vec![v_mov_b32(0, 0), v_mov_b32(1, 1)];
    w64_logic.extend_from_slice(&vop3(0x0c2, 4, 0x80, 256, 0));
    w64_logic.extend_from_slice(&vop3(0x0c2, 6, 0x81, 257, 0));
    w64_logic.extend_from_slice(&[
        s_not_b64(8, 4),
        sop2(0x0f, 10, 8, 6), // s_and_b64
        sop2(0x11, 12, 8, 6), // s_or_b64
        v_mov_b32(1, 10),
        LDS_STORE_V1[0],
        LDS_STORE_V1[1],
        v_mov_b32(1, 11),
        LDS_STORE_V1[0],
        LDS_STORE_V1[1],
        v_mov_b32(1, 12),
        LDS_STORE_V1[0],
        LDS_STORE_V1[1],
        v_mov_b32(1, 13),
        LDS_STORE_V1[0],
        LDS_STORE_V1[1],
        S_ENDPGM,
    ]);
    lower_words(&w64_logic, true, false, layout.clone(), vec![])
        .expect("wave64 s_not/s_and/s_or masks read as data should lower");

    // Wave32: s_not_b32 + s_and_b32 + s_or_b32 over active compares read as data.
    let mut w32 = vec![v_mov_b32(0, 0), v_mov_b32(1, 1)];
    w32.extend_from_slice(&vop3(0x0c2, 4, 0x80, 256, 0));
    w32.extend_from_slice(&vop3(0x0c2, 5, 0x81, 257, 0));
    w32.extend_from_slice(&[
        s_not_b32(6, 4),
        sop2(0x0e, 7, 6, 5), // s_and_b32
        sop2(0x10, 8, 6, 5), // s_or_b32
        v_mov_b32(1, 7),
        LDS_STORE_V1[0],
        LDS_STORE_V1[1],
        v_mov_b32(1, 8),
        LDS_STORE_V1[0],
        LDS_STORE_V1[1],
        S_ENDPGM,
    ]);
    lower_words(&w32, false, false, layout, vec![])
        .expect("wave32 s_not/s_and/s_or masks read as data should lower");
}

#[cfg(windows)]
#[test]
fn non_active_mask_read_as_data_on_device() {
    use crate::emit::host::{Dispatch, Host, Root, View};
    use crate::emit::{HostShader, validator};

    // Wave32 1-thread compute shader:
    //   v1 = s0 (root constant 0)
    //   v_cmp_eq_u32 vcc_lo, 0, v1
    //   s_not_b32 s6, vcc_lo
    //   v0 = s6
    //   buffer_store_dword v0, 0, s[8:11], 0
    let words: [u32; 10] = [
        0xBE85_0380,                             // s_mov_b32 s5, 0
        0x3D << 26 | 0x02 << 18 | 8 << 6 | 2,    // s_load_dwordx4 s[8:11], s[4:5], null
        125 << 25,                               // soffset = null
        0x7E02_0200,                             // v_mov_b32 v1, s0
        0x7D84_0280,                             // v_cmp_eq_u32 vcc_lo, 0, v1
        0xBE86_076A,                             // s_not_b32 s6, vcc_lo
        0x7E00_0206,                             // v_mov_b32 v0, s6
        0x38 << 26 | 0x1C << 18,                 // buffer_store_dword v0, off=0, s[8:11], 0
        0x80 << 24 | ((8u32 >> 2) & 0x1F) << 16, // srsrc = s[8:11], vdata = v0
        S_ENDPGM,
    ];
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 16,
        layout: vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 4,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 4,
                slot: 1,
            },
        ],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 4,
            slot: 1,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 5 << 1,
            threads: [1, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, false, 16).unwrap();
    let st = structure::structure(&p).unwrap();
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
    let io = io::raise(&p, &native).unwrap();
    let lowered = lower(&Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: vec![],
        outputs: vec![],
        prims: vec![],
        mesh_stride: 0,
    })
    .expect("non-active s_not_b32 mask read as data should lower");
    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
    };

    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }

    let _guard = crate::emit::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(false) else {
        return;
    };
    let (root, pso) = host
        .compute_pipeline(&shader)
        .expect("compute_pipeline failed");
    for (input, expected) in [(0u32, !1u32), (1u32, !0u32)] {
        let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        for u in &shader.user_data {
            ud[u.index as usize] = match (u.root, u.slot) {
                (RootType::Constants, 0) if u.dword == 0 => input,
                (RootType::Constants, 0) | (RootType::Table, 1) => 0,
                other => panic!("unexpected user data {other:?}"),
            };
        }
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &[View::RawUav(1)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .expect("dispatch failed");
        let got = out[0][0];
        assert_eq!(got, expected, "input={input}: expected {expected:#x}, got {got:#x}");
    }
}

#[test]
fn root_descriptor_and_buffer_only_descriptor_data_loads() {
    // 1. Direct `s_load_dwordx4` from a Root CBV (`RootType::Descriptor`).
    let root_cbv_layout = vec![RootParam {
        root: RootType::Descriptor,
        table_offset_dwords: 0,
        sgprs: 2,
        base_register: 0,
        user_data: 0,
        slot: 0,
    }];
    let root_cbv_used = vec![UsedResource {
        kind: ResourceType::Cbv,
        root: RootType::Descriptor,
        table_offset_dwords: 0,
        lower_bound: 0,
        stride: 0,
        dwords: 4,
        user_data: 0,
        slot: 0,
    }];
    let root_load = [
        0xF408_0200,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[0:1], null
        0x7E02_0208, // v_mov_b32 v1, s8
        LDS_STORE_V1[0],
        LDS_STORE_V1[1],
        S_ENDPGM,
    ];
    lower_words(&root_load, false, false, root_cbv_layout.clone(), root_cbv_used)
        .expect("s_load_dwordx4 from Root CBV should lower");

    // 2. Direct `s_load_dwordx4` from a Root SRV (`RootType::Descriptor`).
    let root_srv_used = vec![UsedResource {
        kind: ResourceType::Srv,
        root: RootType::Descriptor,
        table_offset_dwords: 0,
        lower_bound: 0,
        stride: 0,
        dwords: 4,
        user_data: 0,
        slot: 0,
    }];
    lower_words(&root_load, false, false, root_cbv_layout, root_srv_used)
        .expect("s_load_dwordx4 from Root SRV should lower");

    // 3. 4-dword SRV buffer descriptor in a table read ONLY for dword 2 (GetDimensions).
    let (layout, used) = buffer_table(Some(16));
    let buf_dims_only = [
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0202,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], null
        0x7E02_020A, // v_mov_b32 v1, s10 (dword 2 = record count)
        LDS_STORE_V1[0],
        LDS_STORE_V1[1],
        S_ENDPGM,
    ];
    lower_words(&buf_dims_only, false, false, layout.clone(), used.clone())
        .expect("4-dword SRV buffer descriptor read only for dimensions should lower");

    // 4. Sibling `s_load_dword` at offset +8 reading dword 2 of a buffer descriptor
    //    loaded separately by `s_load_dwordx4` at offset 0.
    let sibling_load = [
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0202,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], null
        0xE030_2000,
        0x8002_0100, // buffer_load_dword v1, v0, s[8:11], 0 idxen
        LDS_STORE_V1[0],
        LDS_STORE_V1[1],
        0xF400_0302,
        0xFA00_0008, // s_load_dword s12, s[4:5], null offset:8 (dword 2 of same descriptor)
        0x7E02_020C, // v_mov_b32 v1, s12
        LDS_STORE_V1[0],
        LDS_STORE_V1[1],
        S_ENDPGM,
    ];
    lower_words(&sibling_load, false, false, layout.clone(), used)
        .expect("sibling s_load_dword reading dword 2 of buffer descriptor should lower");

    // 5. 8-dword image descriptor read ONLY as data (dimension unknown) still refuses.
    let image_only_used = vec![UsedResource {
        kind: ResourceType::Srv,
        root: RootType::Table,
        table_offset_dwords: 0,
        lower_bound: 0,
        stride: 0,
        dwords: 8,
        user_data: 0,
        slot: 0,
    }];
    let Err(err) = lower_words(&buf_dims_only, false, false, layout, image_only_used) else {
        panic!("8-dword image descriptor with no image use should still be refused");
    };
    assert!(
        matches!(
            err,
            ShaderError::Unsupported {
                what: "a descriptor read only as data",
                ..
            }
        ),
        "unexpected error: {err}"
    );
}

#[test]
fn s_buffer_load_descriptor_dimensions_and_cache_inv() {
    // 1. Buffer descriptor used by `s_buffer_load_dword` (Fields::Smem on SRV)
    //    whose dword 2 is also read as data (previously "a scalar memory read as data").
    let (layout, used) = buffer_table(Some(0));
    let smem_desc_dims = [
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0202,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], null
        0xF420_0304,
        0xFA00_0000, // s_buffer_load_dword s12, s[8:11], null
        0x7E02_020C, // v_mov_b32 v1, s12
        LDS_STORE_V1[0],
        LDS_STORE_V1[1],
        0x7E02_020A, // v_mov_b32 v1, s10 (dword 2 of descriptor used by s_buffer_load)
        LDS_STORE_V1[0],
        LDS_STORE_V1[1],
        S_ENDPGM,
    ];
    lower_words(&smem_desc_dims, false, false, layout, used)
        .expect("s_buffer_load_dword descriptor dimensions read as data should lower");

    // 2. Scalar cache invalidation hints (`s_gl1_inv`, `s_dcache_inv`) are no-ops.
    let cache_inv = [
        0xF47C_0000,
        0x0000_0000, // s_gl1_inv
        0xF480_0000,
        0x0000_0000, // s_dcache_inv
        S_ENDPGM,
    ];
    lower_words(&cache_inv, false, false, vec![], vec![])
        .expect("s_gl1_inv and s_dcache_inv should lower as no-ops");
}

#[test]
fn lane_mask_phi_join_and_loop_mask_read_as_data() {
    let layout = vec![RootParam {
        root: RootType::Constants,
        table_offset_dwords: 0,
        sgprs: 4,
        base_register: 0,
        user_data: 0,
        slot: 0,
    }];

    // 1. Wave64 if-else joining a compare mask with `s_mov_b64 0` on the other arm,
    //    consumed by `v_cndmask_b32` (previously "a lane-mask join of an untracked mask").
    let phi_join_mov64 = [
        0xBF06_8000, // s_cmp_eq_u32 s0, 0
        0xBF85_0003, // s_cbranch_scc1 -> else (skip 3 dwords)
        0xD4C2_0004,
        0x0002_0080, // v_cmp_eq_u32_e64 s[4:5], 0, v0
        0xBF82_0001, // s_branch -> join (skip 1 dword)
        0xBE84_0480, // s_mov_b64 s[4:5], 0
        0xD501_0001,
        128 | (129 << 9) | (4 << 18), // v_cndmask_b32 v1, 0, 1, s[4:5]
        LDS_STORE_V1[0],
        LDS_STORE_V1[1],
        S_ENDPGM,
    ];
    lower_words(&phi_join_mov64, true, false, layout.clone(), vec![])
        .expect("wave64 phi join of compare and s_mov_b64 0 should lower");

    // 2. Wave64 if-else joining a compare mask with split `s_mov_b32 s4, 0; s_mov_b32 s5, 0`.
    let phi_join_split32 = [
        0xBF06_8000, // s_cmp_eq_u32 s0, 0
        0xBF85_0003, // s_cbranch_scc1 -> else (skip 3 dwords)
        0xD4C2_0004,
        0x0002_0080, // v_cmp_eq_u32_e64 s[4:5], 0, v0
        0xBF82_0002, // s_branch -> join (skip 2 dwords)
        0xBE84_0380, // s_mov_b32 s4, 0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xD501_0001,
        128 | (129 << 9) | (4 << 18), // v_cndmask_b32 v1, 0, 1, s[4:5]
        LDS_STORE_V1[0],
        LDS_STORE_V1[1],
        S_ENDPGM,
    ];
    lower_words(&phi_join_split32, true, false, layout.clone(), vec![])
        .expect("wave64 phi join of compare and split s_mov_b32 0 should lower");

    // 3. Wave64 loop accumulating a lane mask via phi + `s_or_b64`, then reading the
    //    phi-joined mask as data (previously "a lane mask holding lanes elsewhere read as data").
    let loop_mask_as_data = [
        0xBE82_0480, // s_mov_b64 s[2:3], 0
        // loop header (offset 4):
        0x7D84_0080, // v_cmp_eq_u32 vcc, 0, v0
        0x8882_6A02, // s_or_b64 s[2:3], s[2:3], vcc
        0x8000_8100, // s_add_u32 s0, s0, 1
        0xBF06_8400, // s_cmp_eq_u32 s0, 4
        0xBF84_FFFB, // s_cbranch_scc0 -> offset 4 (-5 dwords from PC=24)
        // after loop: read phi-joined mask s[2:3] as data
        0xBE86_1002, // s_bcnt1_i32_b64 s6, s[2:3]
        0x7E02_0206, // v_mov_b32 v1, s6
        LDS_STORE_V1[0],
        LDS_STORE_V1[1],
        0x7E02_0203, // v_mov_b32 v1, s3 (high dword)
        LDS_STORE_V1[0],
        LDS_STORE_V1[1],
        S_ENDPGM,
    ];
    lower_words(&loop_mask_as_data, true, false, layout, vec![])
        .expect("wave64 loop phi-joined lane mask read as data should lower");
}

#[test]
fn buffer_descriptor_stride_dword_read_as_data() {
    let (layout, used) = buffer_table(Some(16));
    let read = |dword_sgpr: u32| {
        [
            0xBE84_0300, // s_mov_b32 s4, s0
            0xBE85_0380, // s_mov_b32 s5, 0
            0xF408_0202,
            0xFA00_0000,                       // s_load_dwordx4 s[8:11], s[4:5], null
            0x7E02_0200 | dword_sgpr,          // v_mov_b32 v1, s<dword_sgpr>
            LDS_STORE_V1[0],
            LDS_STORE_V1[1],
            S_ENDPGM,
        ]
    };
    // Dword 1 (stride) and dword 3 (swizzle) are known.
    for sgpr in [9, 11] {
        lower_words(&read(sgpr), false, false, layout.clone(), used.clone())
            .unwrap_or_else(|e| panic!("s{sgpr} of a buffer descriptor should lower: {e}"));
    }
    // Dword 0 is the address: it stays refused.
    let Err(err) = lower_words(&read(8), false, false, layout, used) else {
        panic!("the address dword must stay refused");
    };
    assert!(
        matches!(
            err,
            ShaderError::Unsupported {
                what: "a descriptor's address or format read as data",
                ..
            }
        ),
        "unexpected error: {err}"
    );
}

#[test]
fn wqm_of_an_active_compare_read_as_data() {
    const fn s_wqm_b32(sdst: u8, src0: u8) -> u32 {
        0xBE80_0000 | (sdst as u32) << 16 | 0x09 << 8 | src0 as u32
    }
    const fn s_wqm_b64(sdst: u8, src0: u8) -> u32 {
        0xBE80_0000 | (sdst as u32) << 16 | 0x0A << 8 | src0 as u32
    }
    const fn v_mov_b32(vdst: u8, src0: u16) -> u32 {
        0x7E00_0000 | (vdst as u32) << 17 | (0x001 << 9) | (src0 as u32 & 0x1FF)
    }
    let layout = vec![RootParam {
        root: RootType::Constants,
        table_offset_dwords: 0,
        sgprs: 4,
        base_register: 0,
        user_data: 0,
        slot: 0,
    }];
    let store = |sgpr: u16| [v_mov_b32(1, sgpr), LDS_STORE_V1[0], LDS_STORE_V1[1]];
    // Wave32.
    let mut w32 = vec![v_mov_b32(0, 0), 0x7D84_0080, s_wqm_b32(6, 106)];
    w32.extend(store(6));
    w32.push(S_ENDPGM);
    lower_words(&w32, false, false, layout.clone(), vec![])
        .expect("wave32 s_wqm_b32 of an active compare read as data should lower");
    // Wave64, both dwords.
    let mut w64 = vec![v_mov_b32(0, 0), 0x7D84_0080, s_wqm_b64(6, 106)];
    w64.extend(store(6));
    w64.extend(store(7));
    w64.push(S_ENDPGM);
    lower_words(&w64, true, false, layout, vec![])
        .expect("wave64 s_wqm_b64 of an active compare read as data should lower");
}

#[test]
fn inverted_mask_in_a_mesh_shader_refuses_with_its_reason() {
    const fn s_not_b64(sdst: u8, src0: u8) -> u32 {
        0xBE80_0000 | (sdst as u32) << 16 | 0x08 << 8 | src0 as u32
    }
    let words = [
        0x7E00_0100, // v_mov_b32 v0, v0 (anything reading nothing)
        0x7D84_0080, // v_cmp_eq_u32 vcc, 0, v0
        s_not_b64(6, 106),
        0x7E02_0206, // v_mov_b32 v1, s6
        LDS_STORE_V1[0],
        LDS_STORE_V1[1],
        S_ENDPGM,
    ];
    match lower_words(&words, true, true, vec![], vec![]) {
        Ok(_) => {} // a mesh program may not read it as data at all; no regression
        Err(ShaderError::Unsupported { what, .. }) => assert!(
            what.contains("inverted") || what.contains("lane mask"),
            "unexpected reason: {what}"
        ),
        Err(e) => panic!("{e}"),
    }
}

// --- Wave32 / uniform-LDS / gather+grad (Codex) ---
use super::container::{self, SigElement};
use super::{HostShader, emit, lower};
use crate::dxbc::{Container, FourCC};

const END: u32 = 0xbf81_0000;
const WRITE: [u32; 2] = [0xd834_0000, 0x0000_0002]; // ds_write_b32 v2, v0
const READ: [u32; 2] = [0xd8d8_0000, 0x0300_0002]; // ds_read_b32 v3, v2
const SYNC: u32 = 0xbf8a_0000;
const UAV: [u32; 4] = [0xbe84_0300, 0xbe85_0380, 0xf408_0a02, 0xfa00_0040];
const STORE: [u32; 2] = [0xe070_1000, 0x800a_0302]; // raw[v2] = v3
fn compute(threads: [u32; 3]) -> StageIo {
    StageIo::Cs(ComputeIo {
        threads,
        rsrc2: 1 << 15 | 1 << 11 | 2 << 1, // LDS, thread x/y, two user SGPRs
    })
}

fn shader(words: &[u32], stage_io: StageIo) -> HostShader {
    let (layout, used) = lower::tests::texture_and_uav_tables();
    let pixel = matches!(stage_io, StageIo::Ps { .. });
    let native = NativeProgram {
        stage: if pixel { HwStage::Ps } else { HwStage::Cs },
        wave64: true,
        vgprs: 32,
        layout,
        used,
        io: stage_io,
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, true, 32).unwrap();
    let st = structure::structure(&p).unwrap();
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
    let io = io::raise(&p, &native).unwrap();
    let output = container::signature(
        &[SigElement {
            name: "SV_Target".into(),
            index: 0,
            kind: 16,
            comp: 9,
            interp: 0,
            row: Some(0),
            col: 0,
            cols: 4,
            usage: 15,
            half: false,
        }],
        false,
    );
    let bytes = if pixel {
        crate::dxbc::build(&[(FourCC::OSG1, &output)])
    } else {
        crate::dxbc::build(&[])
    };
    emit(
        &Container::parse(&bytes).unwrap(),
        &native,
        &p,
        &st,
        &io,
        &res,
    )
    .unwrap()
}

fn wave(shader: &HostShader) -> (u32, u32) {
    let psv = Container::parse(&shader.dxil)
        .unwrap()
        .part(FourCC::PSV0)
        .unwrap()
        .data;
    (
        crate::dxbc::u32_at(psv, 20).unwrap(),
        crate::dxbc::u32_at(psv, 24).unwrap(),
    )
}

#[test]
fn divergent_lds_keeps_wave64() {
    let masked = [0x7da4_0080, WRITE[0], WRITE[1], END]; // v_cmpx, predicated write
    let branch = [0x7da4_0080, 0xbf88_0002, WRITE[0], WRITE[1], END];
    // Remove lanes from EXEC on each iteration; LDS inside the loop.
    let varying_loop = [
        0xbe82_0480,
        WRITE[0],
        WRITE[1],
        0x4a00_0081,
        0x7d84_0080,
        0x8882_026a,
        0x8afe_027e,
        0xbf89_fff9,
        0x88fe_027e,
        END,
    ];
    for words in [&masked[..], &branch, &varying_loop] {
        assert_eq!(wave(&shader(words, compute([64, 1, 1]))), (64, 64));
    }
}

#[test]
fn lds_fences_order_atomics_and_stop_at_existing_sync() {
    let atomic = [0xd800_0000, 0x0000_0002]; // ds_add_u32 v2, v0
    let words = [
        &WRITE[..],
        &READ,
        &READ,
        &atomic,
        &READ,
        &[SYNC],
        &WRITE,
        &[END],
    ]
    .concat();
    let code: Vec<_> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let p = ir::build(&code, true, 32).unwrap();
    let st = structure::structure(&p).unwrap();
    let sites = super::lds::split_barriers(&p, &st).unwrap();
    assert_eq!(sites, [(0, 1), (0, 3), (0, 4)].into());
}

#[test]
fn scalar_controlled_lds_loop_can_split() {
    // s2 counts down to zero; the loop back edge carries a scalar phi.
    let words = [
        0xbe82_0384,
        WRITE[0],
        WRITE[1],
        0x8082_8102,
        0xbf06_8002,
        0xbf84_fffb,
        END,
    ];
    assert_eq!(wave(&shader(&words, compute([33, 1, 1]))), (0, u32::MAX));
}

#[cfg(windows)]
mod lds_wave_device {
    use super::*;
    use crate::emit::{
        host::{Dispatch, Host, Root, View, stub_vs},
        validator,
        valu_tests::ONE_AT_A_TIME,
    };

    /// Prefer the discrete adapter on hybrid systems.
    fn host() -> Host {
        use windows::Win32::Graphics::{
            Direct3D::D3D_FEATURE_LEVEL_12_0,
            Direct3D12::D3D12CreateDevice,
            Dxgi::{
                CreateDXGIFactory2, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IDXGIAdapter1,
                IDXGIFactory6,
            },
        };
        // SAFETY: DXGI owns the returned adapter; D3D12 initializes the device output.
        unsafe {
            let factory: IDXGIFactory6 = CreateDXGIFactory2(Default::default()).unwrap();
            if let Ok(adapter) = factory.EnumAdapterByGpuPreference::<IDXGIAdapter1>(
                0,
                DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
            ) {
                let mut device = None;
                if D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device).is_ok() {
                    let name = adapter.GetDesc1().unwrap().Description;
                    let len = name.iter().position(|&c| c == 0).unwrap_or(name.len());
                    eprintln!("D3D12 adapter: {}", String::from_utf16_lossy(&name[..len]));
                    return Host {
                        device: device.unwrap(),
                    };
                }
            }
        }
        Host::new(true).expect("SM 6.6 WARP")
    }

    fn validate(shader: &HostShader) {
        let path = validator::Validator::default_path();
        if let Ok(mut v) = validator::Validator::load(path.as_deref()) {
            v.validate(&mut shader.dxil.clone())
                .expect("DXIL validator");
        } else {
            eprintln!("dxil.dll unavailable; skipping validator");
        }
    }

    fn image_call(shader: &HostShader, intrinsic: &str) -> Option<(String, String)> {
        let dxc = validator::Validator::default_path()?.with_file_name("dxc.exe");
        if !dxc.is_file() {
            return None;
        }
        let path =
            std::env::temp_dir().join(format!("victory-image-test-{}.dxil", std::process::id()));
        std::fs::write(&path, &shader.dxil).unwrap();
        let output = std::process::Command::new(dxc)
            .arg("-dumpbin")
            .arg(&path)
            .output();
        std::fs::remove_file(path).unwrap();
        let output = output.unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let ir = String::from_utf8(output.stdout).unwrap();
        let call = ir
            .lines()
            .find(|l| l.contains("call ") && l.contains(intrinsic))
            .unwrap()
            .to_owned();
        Some((ir, call))
    }

    fn dispatch(shader: &HostShader, count: u32) -> Vec<u32> {
        validate(shader);
        let host = host();
        let (root, pso) = host.compute_pipeline(shader).expect("create compute PSO");
        // The table starts at heap index zero; its UAV is entry two.
        let constants = vec![0; shader.user_data_dwords().max(1) as usize];
        let views = [View::RawUav(1), View::RawUav(1), View::RawUav(count)];
        host.dispatch(
            &root,
            &pso,
            &Dispatch {
                roots: vec![(0, Root::Constants(constants))],
                views: &views,
                samplers: &[],
                groups: [1, 1, 1],
            },
        )
        .expect("dispatch")
        .pop()
        .unwrap()
    }

    #[test]
    fn small_wave64_masks_and_thread_ids_on_device() {
        let _lock = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
        for [x, y] in [[32u32, 1], [8, 4], [4, 4]] {
            let mut words = UAV.to_vec();
            // v2 = 4 * (x * tid.y + tid.x).
            words.extend([
                0x3404_0200 | (128 + x.trailing_zeros()),
                0x4a04_0500,
                0x3404_0482,
            ]);
            // mbcnt_lo(-1, 0), mbcnt_hi(-1, lo), add high EXEC ballot.
            words.extend([
                0xd765_0003,
                193 | 128 << 9,
                0xd766_0003,
                193 | 259 << 9,
                0x4a06_067f,
            ]);
            words.extend(STORE);
            words.push(END);
            let cs = shader(&words, compute([x, y, 1]));
            assert_eq!(wave(&cs), (32, 32));
            assert_eq!(dispatch(&cs, x * y), (0..x * y).collect::<Vec<_>>());
        }
    }

    #[test]
    fn split_lds_exchanges_data_between_host_waves() {
        let _lock = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
        for explicit in [false, true] {
            let mut words = UAV.to_vec();
            words.push(0x3404_0082); // v2 = 4 * tid.x
            words.extend(WRITE);
            if explicit {
                words.push(SYNC);
            }
            words.extend([0x3a04_04ff, 128]); // exchange halves of the group
            words.extend(READ);
            words.push(0x3404_0082);
            words.extend(STORE);
            words.push(END);
            let cs = shader(&words, compute([64, 1, 1]));
            assert_eq!(wave(&cs), (0, u32::MAX));
            assert_eq!(
                dispatch(&cs, 64),
                (0..64).map(|i| i ^ 32).collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn explicit_gather_and_clamped_grad_create_pipelines() {
        let _lock = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
        let host = host();
        for (op, a16) in [0x44u32, 0x54, 0x23, 0x33]
            .into_iter()
            .flat_map(|op| [(op, false), (op, true)])
        {
            let gather = op & 0x40 != 0;
            let offset = op & 0x10 != 0;
            let mut words = vec![
                0xbe84_0300,
                0xbe85_0380,
                0xbe86_0301,
                0xbe87_0380,
                0xf40c_0602,
                0xfa00_0000,
                0xf408_0203,
                0xfa00_0000,
            ]; // SRV s24, sampler s8
            let mut operands = vec![];
            if offset {
                operands.push(0x3f01);
            } // (+1,-1)
            if a16 {
                if !gather {
                    operands.extend([0x0000_3800, 0x3400_0000]);
                }
                operands.extend([0x3a00_3400, 0x3c00]); // packed uv, then LOD/clamp
            } else {
                if !gather {
                    operands.extend([0.5f32, 0., 0., 0.25].map(f32::to_bits));
                }
                operands.extend([0.25f32, 0.75, 1.0].map(f32::to_bits));
            }
            for (i, v) in operands.into_iter().enumerate() {
                words.extend([0x7e00_02ff | (i as u32 + 4) << 17, v]);
            }
            words.extend([0xf000_0108 | op << 18, 0x0046_0c04 | u32::from(a16) << 30]);
            let cs_or_ps = if gather {
                words.extend([0xd834_0000, 0x0000_0c00, END]);
                compute([8, 4, 1])
            } else {
                words.extend([0xf800_080f, 0x0c0c_0c0c, END]);
                StageIo::Ps {
                    inputs: vec![],
                    input_addr: 0,
                    db_shader_control: 0,
                    patches: vec![],
                }
            };
            let s = shader(&words, cs_or_ps);
            validate(&s);
            if !a16
                && let Some((ir, call)) = image_call(
                    &s,
                    if gather {
                        "@dx.op.textureGather."
                    } else {
                        "@dx.op.sampleGrad."
                    },
                )
            {
                // Validate the values in the intrinsic slots. Swapping uv,
                // derivatives or clamp still produces perfectly legal DXIL.
                let args: Vec<_> = call
                    .split_once('(')
                    .unwrap()
                    .1
                    .split(')')
                    .next()
                    .unwrap()
                    .split(", ")
                    .collect();
                let check_float = |index: usize, expected: f32| {
                    let reg = args[index].strip_prefix("float ").unwrap();
                    let def = format!("{reg} = bitcast i32 {} to float", expected.to_bits());
                    assert!(ir.lines().any(|l| l.trim() == def), "{def} in {call}");
                };
                check_float(3, 0.25);
                check_float(4, 0.75);
                if !gather {
                    if offset && args.get(7) == Some(&"i32 1") {
                        assert_eq!(&args[7..9], ["i32 1", "i32 -1"]);
                    } else if !offset {
                        assert_eq!(&args[7..9], ["i32 0", "i32 0"]);
                    }
                    for (i, f) in [(10, 0.5), (11, 0.), (13, 0.), (14, 0.25), (16, 1.)] {
                        check_float(i, f);
                    }
                }
            }
            if gather {
                host.compute_pipeline(&s).unwrap();
            } else {
                host.pipeline(&stub_vs(&s).unwrap(), &s).unwrap();
            }
        }
    }
}

