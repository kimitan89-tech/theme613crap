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
        root_uavs: lowered.root_uavs,
        const_views: lowered.const_views,
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
fn spill_table_constants_and_root_uav_lower() {
    // Wave32 CS: spill pointer in s[0:1], Constants at ud 16, root UAV at ud 20.
    // s_and_b32 s3, s1, 0xff; s_mov_b32 s2, s0;
    // s_load_dwordx4 s[4:7], s[2:3], null;     // Constants
    // s_load_dwordx2 s[8:9], s[2:3], 0x10;     // Descriptor UAV
    // build buffer desc; v0 = s4; buffer_store_dword v0, s[8:11]
    let words = [
        0x8703_FF01,
        0x0000_00FF, // s_and_b32 s3, s1, lit(0xff)
        0xBE82_0300, // s_mov_b32 s2, s0
        0xF408_0101,
        0xFA00_0000, // s_load_dwordx4 s[4:7], s[2:3], null
        0xF404_0201,
        0xFA00_0010, // s_load_dwordx2 s[8:9], s[2:3], 0x10
        0x8809_FF09,
        0x0010_0000, // s_or_b32 s9, s9, lit(0x100000)
        0xBE8A_03C1, // s_mov_b32 s10, -1
        0xBE8B_03FF,
        0x1101_4FAC, // s_mov_b32 s11, 0x11014fac
        0x7E00_0204, // v_mov_b32 v0, s4
        0x38 << 26 | 0x1C << 18, // buffer_store_dword v0, off=0, s[8:11], 0
        0x80 << 24 | ((8u32 >> 2) & 0x1F) << 16,
        S_ENDPGM,
    ];
    let layout = vec![
        RootParam {
            root: RootType::Constants,
            table_offset_dwords: 0,
            sgprs: 4,
            base_register: 0,
            user_data: 16,
            slot: 0,
        },
        RootParam {
            root: RootType::Descriptor,
            table_offset_dwords: 0,
            sgprs: 2,
            base_register: 0,
            user_data: 20,
            slot: 1,
        },
    ];
    let used = vec![UsedResource {
        kind: ResourceType::Uav,
        root: RootType::Descriptor,
        table_offset_dwords: 0,
        lower_bound: 0,
        stride: 0,
        dwords: 4,
        user_data: 20,
        slot: 1,
    }];
    let lowered = lower_words(&words, false, false, layout, used)
        .expect("spill-table Constants + root UAV should lower");
    assert!(
        !lowered.root_uavs.is_empty(),
        "expected a root UAV binding, got cbvs={:?} srvs={:?} uavs={:?}",
        lowered.root_cbvs,
        lowered.root_srvs,
        lowered.root_uavs
    );
    assert!(
        lowered.user_data.iter().any(|u| u.slot == 0 && u.dword == 0),
        "expected compact user-data for spilled Constants: {:?}",
        lowered.user_data
    );
    assert!(
        lowered.user_data.iter().all(|u| u.index < 16),
        "compact user-data indices must stay below the spill base: {:?}",
        lowered.user_data
    );
}

#[cfg(windows)]
#[test]
fn spill_table_root_uav_store_on_device() {
    use crate::emit::host::{Dispatch, Host, Root};
    use crate::emit::{HostShader, validator};

    let words = [
        0x8703_FF01,
        0x0000_00FF, // s_and_b32 s3, s1, lit(0xff)
        0xBE82_0300, // s_mov_b32 s2, s0
        0xF408_0101,
        0xFA00_0000, // s_load_dwordx4 s[4:7], s[2:3], null
        0xF404_0201,
        0xFA00_0010, // s_load_dwordx2 s[8:9], s[2:3], 0x10
        0x8809_FF09,
        0x0010_0000, // s_or_b32 s9, s9, lit(0x100000)
        0xBE8A_03C1, // s_mov_b32 s10, -1
        0xBE8B_03FF,
        0x1101_4FAC, // s_mov_b32 s11, 0x11014fac
        0x7E00_0204, // v_mov_b32 v0, s4
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | ((8u32 >> 2) & 0x1F) << 16,
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
                user_data: 16,
                slot: 0,
            },
            RootParam {
                root: RootType::Descriptor,
                table_offset_dwords: 0,
                sgprs: 2,
                base_register: 0,
                user_data: 20,
                slot: 1,
            },
        ],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Descriptor,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 20,
            slot: 1,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 4 << 1,
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
    .expect("spill root UAV should lower");
    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        root_uavs: lowered.root_uavs.clone(),
        const_views: lowered.const_views,
    };
    assert_eq!(shader.root_uavs, vec![1]);

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
    let expected = 0xA5A5_5A5Au32;
    let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
    for u in &shader.user_data {
        ud[u.index as usize] = match (u.root, u.slot, u.dword) {
            (RootType::Constants, 0, 0) => expected,
            (RootType::Constants, 0, _) => 0,
            other => panic!("unexpected user data {other:?}"),
        };
    }
    // Root parameter order: user data, then root UAV (slot 1).
    let out = host
        .dispatch(
            &root,
            &pso,
            &Dispatch {
                roots: vec![
                    (0, Root::Constants(ud)),
                    (1, Root::UavView { data: vec![0; 64] }),
                ],
                views: &[],
                samplers: &[],
                groups: [1, 1, 1],
            },
        )
        .expect("dispatch failed");
    assert_eq!(out[0][0], expected);
}

#[test]
fn subdword_lds_read_write_lower() {
    // ds_write_b16 v1, v2 offset:10; ds_read_u16 v0, v1 offset:10; ds_write_b8; ds_read_i8
    let words = [
        0xD87C_000A,
        0x0000_0201, // ds_write_b16 v1, v2 offset:10
        0xD8F0_000A,
        0x0000_0001, // ds_read_u16 v0, v1 offset:10
        0xD878_0004,
        0x0000_0201, // ds_write_b8 v1, v2 offset:4
        0xD8E4_0004,
        0x0000_0001, // ds_read_i8 v0, v1 offset:4
        0xD8E8_0004,
        0x0000_0001, // ds_read_u8 v0, v1 offset:4
        0xD8EC_000A,
        0x0000_0001, // ds_read_i16 v0, v1 offset:10
        S_ENDPGM,
    ];
    lower_words(&words, false, false, vec![], vec![])
        .expect("sub-dword LDS read/write should lower");
}

#[test]
fn subdword_buffer_load_lower() {
    let (layout, used) = buffer_table(Some(0));
    // Same skeleton as dword loads; MUBUF op 0x0a = ushort, 0x08 = ubyte.
    let load = |op: u32| {
        [
            0xBE84_0300u32, // s_mov_b32 s4, s0
            0xBE85_0380,    // s_mov_b32 s5, 0
            0xF408_0202,
            0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], null
            0xE030_2000 ^ ((0x0Cu32 ^ op) << 18),
            0x8002_0100, // buffer_load_* v1, off=0, s[8:11], 0
            LDS_STORE_V1[0],
            LDS_STORE_V1[1],
            S_ENDPGM,
        ]
    };
    for (name, op) in [("ushort", 0x0Au32), ("ubyte", 0x08), ("sshort", 0x0B), ("sbyte", 0x09)] {
        lower_words(&load(op), false, false, layout.clone(), used.clone())
            .unwrap_or_else(|e| panic!("buffer_load_{name} should lower: {e}"));
    }
}

#[test]
fn constants_gpu_address_64bit_and_shifted_track() {
    use crate::resource::{self, Address};
    // 64-bit pointer in Constants dword 0:1, then s_load_dword from it.
    let r = {
        let code: Vec<u8> = [
            0xBE82_0300u32, // s_mov_b32 s2, s0
            0xBE83_0301,    // s_mov_b32 s3, s1
            0xF400_0001,
            0xFA00_0000, // s_load_dword s0, s[2:3], null
            S_ENDPGM,
        ]
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect();
        let p = ir::build(&code, false, 16).unwrap();
        let layout = [RootParam {
            root: RootType::Constants,
            table_offset_dwords: 0,
            sgprs: 4,
            base_register: 0,
            user_data: 0,
            slot: 0,
        }];
        resource::track(&p, &layout, HwStage::Cs, None).unwrap()
    };
    assert!(
        r.scalar.iter().any(|(_, a)| {
            *a == Address {
                param: 0,
                offset: 0,
                dynamic: None,
                dword: 0,
                shift: 0,
            }
        }),
        "expected 64-bit Constants address: {:?}",
        r.scalar
    );

    // s_mov_b32 lo, c; s_mov_b32 hi, 0; s_lshl_b64 [lo:hi], [lo:hi], 8; s_or_b32 lo, lo, 12
    // then s_load_dword from [lo:hi]
    let r = {
        let code: Vec<u8> = [
            0xBE82_0300u32, // s_mov_b32 s2, s0
            0xBE83_0380,    // s_mov_b32 s3, 0
            0x8F82_FF02,
            0x0000_0008, // s_lshl_b64 s[2:3], s[2:3], lit(8)
            0x8802_8C02, // s_or_b32 s2, s2, 12 (inline 140=12)
            0xF400_0001,
            0xFA00_0000, // s_load_dword s0, s[2:3], null
            S_ENDPGM,
        ]
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect();
        let p = ir::build(&code, false, 16).unwrap();
        let layout = [RootParam {
            root: RootType::Constants,
            table_offset_dwords: 0,
            sgprs: 2,
            base_register: 0,
            user_data: 0,
            slot: 0,
        }];
        resource::track(&p, &layout, HwStage::Cs, None).unwrap()
    };
    assert!(
        r.scalar.iter().any(|(_, a)| a.param == 0 && a.shift == 8 && a.dword == 0 && a.offset == 12),
        "expected shifted Constants address: {:?}",
        r.scalar
    );
}

#[test]
fn constants_gpu_address_lowers_const_view() {
    let words = [
        0xBE82_0300u32, // s_mov_b32 s2, s0
        0xBE83_0301,    // s_mov_b32 s3, s1
        0xF400_0201,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[2:3], null
        0x7E02_0208, // v_mov_b32 v1, s8
        LDS_STORE_V1[0],
        LDS_STORE_V1[1],
        S_ENDPGM,
    ];
    let layout = vec![RootParam {
        root: RootType::Constants,
        table_offset_dwords: 0,
        sgprs: 4,
        base_register: 0,
        user_data: 0,
        slot: 0,
    }];
    let lowered = lower_words(&words, false, false, layout, vec![])
        .expect("Constants GPU address s_load should lower");
    assert!(
        !lowered.const_views.is_empty(),
        "expected ConstView binding: {:?}",
        lowered.const_views
    );
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

    // 5. 8-dword image descriptor read ONLY as data (no image_* use): dimensions
    //    from GetDimensions with a default 2D view.
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
    lower_words(&buf_dims_only, false, false, layout, image_only_used)
        .expect("8-dword image descriptor dimensions-only should lower as 2D");
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
    // Dword 0 (non-null token), 1 (stride), 2 (count), and 3 (swizzle) are known.
    for sgpr in [8, 9, 10, 11] {
        lower_words(&read(sgpr), false, false, layout.clone(), used.clone())
            .unwrap_or_else(|e| panic!("s{sgpr} of a buffer descriptor should lower: {e}"));
    }
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
fn predicated_lds_in_uniform_block_can_split() {
    // v_cmpx predicates the write, but every thread still reaches the block:
    // the inserted group fence runs outside begin_guarded, so WaveSize(64)
    // is unnecessary.
    let masked = [0x7da4_0080, WRITE[0], WRITE[1], END];
    assert_eq!(wave(&shader(&masked, compute([64, 1, 1]))), (0, u32::MAX));
}

#[test]
fn divergent_lds_avoids_wave64_refusal() {
    // Divergent LDS cannot get safe split fences, but forcing WaveSize(64)
    // only refuses the shader on Wave32 hosts. Leave WaveSize unset instead.
    let branch = [0x7da4_0080, 0xbf88_0002, WRITE[0], WRITE[1], END];
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
    for words in [&branch[..], &varying_loop] {
        assert_eq!(wave(&shader(words, compute([64, 1, 1]))), (0, u32::MAX));
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

    /// Wave64-on-Wave32: bit-exact cross-half reads, mbcnt, readfirstlane, and
    /// predicated LDS without WaveSize(64).
    #[test]
    fn wave64_on_wave32_bit_exact_on_device() {
        let _lock = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
        // VOP3 readlane: src0=v0 is operand 256, src1=inline lane (128+n).
        let readlane = |sdst: u32, lane: u32| -> [u32; 2] {
            [0xd760_0000 | sdst, 0x100 | ((128 + lane) << 9)]
        };

        // 1) v_readlane of lane 0 and lane 63 across Wave32 halves.
        for (lane, expect) in [(0u32, 0u32), (63, 63)] {
            let mut words = UAV.to_vec();
            words.extend([
                0xd765_0000,
                193 | 128 << 9,
                0xd766_0000,
                193 | 256 << 9, // v0 = console lane
            ]);
            words.extend(readlane(2, lane));
            words.extend([
                0x7e06_0202, // v3 = s2
                0x3404_0082, // v2 = 4 * tid.x
            ]);
            words.extend(STORE);
            words.push(END);
            let cs = shader(&words, compute([64, 1, 1]));
            assert_eq!(wave(&cs), (32, 32), "readlane({lane}) should emulate");
            let out = dispatch(&cs, 64);
            assert!(
                out.iter().all(|&v| v == expect),
                "readlane({lane}): expected {expect}, got {out:?}"
            );
        }

        // 2) mbcnt_lo + mbcnt_hi over all-ones → console lane index 0..63.
        {
            let mut words = UAV.to_vec();
            words.extend([
                0xd765_0003,
                193 | 128 << 9,
                0xd766_0003,
                193 | 259 << 9,
                0x3404_0082,
            ]);
            words.extend(STORE);
            words.push(END);
            let cs = shader(&words, compute([64, 1, 1]));
            assert_eq!(wave(&cs), (32, 32));
            assert_eq!(dispatch(&cs, 64), (0..64).collect::<Vec<_>>());
        }

        // 3) v_readfirstlane where only lanes >= 32 are active → value 32.
        {
            let mut words = UAV.to_vec();
            words.extend([
                0xd765_0000,
                193 | 128 << 9,
                0xd766_0000,
                193 | 256 << 9, // v0 = lane
                0x7d82_009f,    // v_cmp_lt_u32 vcc, 31, v0  (v0 > 31)
                0xbe82_246a,    // s_and_saveexec_b64 s[2:3], vcc
                0x7e00_0500,    // v_readfirstlane_b32 s0, v0
                0x7e06_0200,    // v3 = s0
                0x3404_0082,
            ]);
            words.extend(STORE);
            words.push(END);
            let cs = shader(&words, compute([64, 1, 1]));
            assert_eq!(wave(&cs), (32, 32));
            let out = dispatch(&cs, 64);
            for (i, &v) in out.iter().enumerate() {
                if i >= 32 {
                    assert_eq!(v, 32, "lane {i} expected readfirstlane=32, got {v}");
                }
            }
        }

        // 4) Cross-half v_readlane(32) (DPP+readlane style reduction).
        {
            let mut words = UAV.to_vec();
            words.extend([
                0xd765_0000,
                193 | 128 << 9,
                0xd766_0000,
                193 | 256 << 9,
            ]);
            words.extend(readlane(2, 32));
            words.extend([0x7e06_0202, 0x3404_0082]);
            words.extend(STORE);
            words.push(END);
            let cs = shader(&words, compute([64, 1, 1]));
            assert_eq!(wave(&cs), (32, 32));
            assert!(
                dispatch(&cs, 64).iter().all(|&v| v == 32),
                "expected readlane(32)=32"
            );
        }

        // 5) Predicated ds_write + ds_read in a uniform block: no WaveSize(64).
        {
            let mut words = UAV.to_vec();
            words.push(0x3404_0082);
            words.push(0x7e00_0202);
            words.push(0x7da4_0080);
            words.extend(WRITE);
            words.push(SYNC);
            words.extend(READ);
            words.push(0x3404_0082);
            words.extend(STORE);
            words.push(END);
            let cs = shader(&words, compute([64, 1, 1]));
            assert_ne!(wave(&cs), (64, 64));
            host().compute_pipeline(&cs).expect("predicated LDS PSO");
        }

        // 6) Exclusive prefix-of-ones over 64 lanes ≡ console lane index.
        // `v_mbcnt` lowers through the Wave64-on-Wave32 ballot path; a true
        // DPP `wavePrefixOp` section is covered offline by `wave::tests`.
        // Dispatch checks the 64-lane exclusive-prefix identity on device.
        {
            let mut words = UAV.to_vec();
            words.extend([
                0xd765_0003,
                193 | 128 << 9,
                0xd766_0003,
                193 | 259 << 9, // v3 = console lane
                0x3404_0082,
            ]);
            words.extend(STORE);
            words.push(END);
            let cs = shader(&words, compute([64, 1, 1]));
            assert_eq!(wave(&cs), (32, 32));
            assert_eq!(
                dispatch(&cs, 64),
                (0..64).collect::<Vec<_>>(),
                "exclusive prefix of ones over 64 lanes"
            );
        }
    }

}

#[test]
fn divergent_s_barrier_keeps_wave64() {
    // A divergent s_barrier is a no-op inside one console wave, but hangs when
    // the group is two host Wave32s: keep WaveSize(64).
    let words = [
        0x7da4_0080, // v_cmpx_gt_u32 exec, v0, 0 (half the lanes leave)
        0xbf88_0001, // s_cbranch_execz +1
        SYNC,
        END,
    ];
    assert_eq!(wave(&shader(&words, compute([64, 1, 1]))), (64, 64));
}

#[test]
fn divergent_cross32_readlane_keeps_wave64() {
    // v_readlane in a divergent block needs an LDS barrier there: refuse
    // Wave32 emulation rather than deadlock.
    let readlane = [0xd760_0002u32, 0x100 | ((128 + 32) << 9)]; // s2 = readlane(v0, 32)
    let mut words = vec![
        0xd765_0000,
        193 | 128 << 9,
        0xd766_0000,
        193 | 256 << 9, // v0 = console lane
        0x7da4_0080,    // v_cmpx
        0xbf88_0002,    // s_cbranch_execz skip readlane
    ];
    words.extend(readlane);
    words.push(END);
    assert_eq!(wave(&shader(&words, compute([64, 1, 1]))), (64, 64));
}

#[test]
fn divergent_ballot_keeps_wave64() {
    // A compare defined inside a divergent block, then read as data: the
    // Wave64-on-Wave32 ballot barrier would land in that block and hang.
    let words = [
        0x7da4_0080u32, // v_cmpx_gt_u32 exec, v0, 0
        0xbf88_0002,    // s_cbranch_execz skip compare+bcnt
        0x7d84_0080,    // v_cmp_eq_u32 vcc, 0, v0 (mask def, divergent)
        0xbe86_106a,    // s_bcnt1_i32_b64 s6, vcc
        END,
    ];
    assert_eq!(wave(&shader(&words, compute([64, 1, 1]))), (64, 64));
}

#[test]
fn mesh_lane_mask_as_data_lowers() {
    // Mesh uses console lane masks; reading a compare mask as numbers must
    // lower (same pattern as the remaining failed mesh shaders).
    let words = [
        0x7D84_0080u32, // v_cmp_eq_u32 vcc, 0, v0
        0xBE86_106A,    // s_bcnt1_i32_b64 s6, vcc
        0x7E02_0206,    // v_mov_b32 v1, s6
        LDS_STORE_V1[0],
        LDS_STORE_V1[1],
        S_ENDPGM,
    ];
    lower_words(&words, true, true, vec![], vec![])
        .expect("mesh lane mask as data should lower");
}

#[test]
fn sampler_heap_descriptor_as_data_lowers() {
    // CS: heap base in s0. Mesh: user data starts at s8 (merged-stage layout).
    let words_at = |base: u8| {
        let s4 = 0xBE84_0300u32 | u32::from(base); // s_mov_b32 s4, s{base}
        [
            s4,
            0xBE85_0380, // s_mov_b32 s5, 0
            0xF408_0202,
            0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], null
            0x7E02_0208, // v_mov_b32 v1, s8 (null-check dword)
            LDS_STORE_V1[0],
            LDS_STORE_V1[1],
            S_ENDPGM,
        ]
    };
    let layout = |user_data: u8| {
        vec![RootParam {
            root: RootType::SamplerHeap,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data,
            slot: 0,
        }]
    };
    lower_words(&words_at(0), false, false, layout(0), vec![])
        .expect("CS sampler-heap descriptor as data should lower");
    lower_words(&words_at(8), false, true, layout(0), vec![])
        .expect("mesh sampler-heap descriptor as data should lower");
}


#[test]
fn buffer_store_short_and_byte_lower() {
    let (layout, used) = buffer_table(Some(0));
    // Same skeleton as buffer_store_dword (0xE070_0000 = op 0x1c raw); short=0x1a, byte=0x18.
    let store = |op: u32| {
        let word0 = 0xE070_0000u32 ^ ((0x1Cu32 ^ op) << 18);
        [
            0xBE84_0300u32, // s_mov_b32 s4, s0
            0xBE85_0380,    // s_mov_b32 s5, 0
            0xF408_0202,
            0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], null
            0x7E02_0281, // v_mov_b32 v1, 1
            word0,
            0x8002_0100, // buffer_store_* v1, off=0, s[8:11], 0 (raw)
            S_ENDPGM,
        ]
    };
    for (name, op) in [("short", 0x1Au32), ("byte", 0x18)] {
        lower_words(&store(op), false, false, layout.clone(), used.clone())
            .unwrap_or_else(|e| panic!("buffer_store_{name} should lower: {e}"));
    }
}

#[test]
fn image_descriptor_dword0_null_token_and_bfe_lower() {
    // s_load_dwordx8 of a table image, then s_bfe base-level + read dword 0.
    let layout = vec![RootParam {
        root: RootType::Table,
        table_offset_dwords: 0,
        sgprs: 1,
        base_register: 0,
        user_data: 0,
        slot: 0,
    }];
    let used = vec![UsedResource {
        kind: ResourceType::Srv,
        root: RootType::Table,
        table_offset_dwords: 0,
        lower_bound: 0,
        stride: 4,
        dwords: 8,
        user_data: 0,
        slot: 0,
    }];
    let words = [
        0xBE84_0300u32, // s_mov_b32 s4, s0
        0xBE85_0380,    // s_mov_b32 s5, 0
        0xF410_0202,
        0xFA00_0000, // s_load_dwordx8 s[8:15], s[4:5], null
        0x7E02_0208, // v_mov_b32 v1, s8 (dword 0 null token)
        LDS_STORE_V1[0],
        LDS_STORE_V1[1],
        0x7E02_020B, // v_mov_b32 v1, s11 (dword 3: base/last level + type)
        LDS_STORE_V1[0],
        LDS_STORE_V1[1],
        0x7E02_020A, // v_mov_b32 v1, s10 (dword 2: width/height)
        LDS_STORE_V1[0],
        LDS_STORE_V1[1],
        S_ENDPGM,
    ];
    lower_words(&words, false, false, layout, used)
        .expect("image descriptor dword0/2/3 as data should lower");
}

#[test]
fn address_phi_null_vs_root_constants() {
    use crate::resource::{self, Address};
    // Wave32: if (s0 == 0) s[2:3]=0 else s[2:3]=s[0:1]; then global_load via saddr.
    // After structurisation the join is a phi of null vs the Constants address.
    let layout = vec![RootParam {
        root: RootType::Constants,
        table_offset_dwords: 0,
        sgprs: 4,
        base_register: 0,
        user_data: 0,
        slot: 0,
    }];
    // Use SGPR address form: global_load_dword v0, v0, s[2:3]
    // Encoding borrowed from resource tests (FLAT with saddr).
    let words = [
        0xBF06_8000u32, // s_cmp_eq_u32 s0, 0
        0xBF85_0003,    // s_cbranch_scc1 -> null arm (PC+4 + 3 dwords)
        0xBE82_0300,    // s_mov_b32 s2, s0
        0xBE83_0301,    // s_mov_b32 s3, s1
        0xBF82_0002,    // s_branch join
        0xBE82_0380,    // null: s_mov_b32 s2, 0
        0xBE83_0380,    // s_mov_b32 s3, 0
        0x7E00_0280, // join: v_mov_b32 v0, 0 (offset)
        0xDC30_8000,
        0x0002_0000, // global_load_dword v0, v0, s[2:3]
        S_ENDPGM,
    ];
    let p = ir::build(
        &words.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<_>>(),
        false,
        256,
    )
    .unwrap();
    let st = structure::structure(&p).unwrap();
    let res = resource::track(&p, &layout, HwStage::Cs, Some(&st)).unwrap();
    let a = res
        .scalar
        .iter()
        .find_map(|(_, a)| (a.param == 0).then_some(*a))
        .expect("phi of null vs Constants should track as Constants address");
    assert_eq!(
        a,
        Address {
            param: 0,
            offset: 0,
            dynamic: None,
            dword: 0,
            shift: 0,
        }
    );
    lower_words(&words, false, false, layout, vec![])
        .expect("null-vs-root Constants address should lower");
}

