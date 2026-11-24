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
