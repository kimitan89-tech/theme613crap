use std::collections::HashSet;

use crate::gfx10::Fields;
use crate::ir::{Program, Reg, Value, reverse_postorder};
use crate::resource::Site;
use crate::structure::{Atom, Bool, LaneExit, Structured};

/// Reconvergence analysis for splitting a console wave across host waves.
pub(super) struct Analysis {
    /// Blocks that only some threads of the group reach after a divergent
    /// branch. Barriers in these blocks deadlock.
    pub divergent: Vec<bool>,
    varying: HashSet<Value>,
}

impl Analysis {
    /// Whether a lane boolean is group-uniform under this analysis.
    pub fn uniform(&self, st: &Structured, g: Bool) -> bool {
        st.bools
            .support(g)
            .iter()
            .all(|a| matches!(a, Atom::Uniform(v) if !self.varying.contains(v)))
    }
}

/// Fences needed to split a single console wave across host waves. None means
/// at least one access or existing barrier cannot be proved group-uniform.
pub(super) fn split_barriers(p: &Program, st: &Structured) -> Option<HashSet<Site>> {
    let a = analyze(p, st);
    barriers_for(p, st, &a)
}

pub(super) fn analyze(p: &Program, st: &Structured) -> Analysis {
    let blocks = &p.blocks;
    let order = reverse_postorder(blocks);
    let mut rank = vec![usize::MAX; blocks.len()];
    for (n, &b) in order.iter().enumerate() {
        rank[b] = n;
    }
    // If all branch conditions are group-uniform, all lanes choose the same
    // phi inputs on every iteration (including scalar induction variables).
    // Otherwise a phi can select different inputs after reconvergence.
    let uniform = |g, varying: &HashSet<Value>| {
        st.bools
            .support(g)
            .iter()
            .all(|a| matches!(a, Atom::Uniform(v) if !varying.contains(v)))
    };
    let mut varying = varying_values(p, &order, true);
    if order
        .iter()
        .any(|&b| matches!(st.exits[b], LaneExit::Branch { when, .. } if !uniform(when, &varying)))
    {
        varying = varying_values(p, &order, false);
    }
    // Compute the first guaranteed reconvergence in reverse execution order.
    // A back edge can also exit at the emitted loop limit, so it guarantees
    // no later block. 'end' represents a return or such a loop exit.
    let end = order.len();
    let mut join = vec![end; end + 1];
    let mut divergent = vec![false; blocks.len()];
    for (n, &b) in order.iter().enumerate().rev() {
        let forward = |t: usize| if rank[t] > n { rank[t] } else { end };
        join[n] = match st.exits[b] {
            LaneExit::Jump(t) => forward(t),
            LaneExit::Branch {
                when,
                taken,
                not_taken,
            } => {
                let (mut a, mut f) = (forward(taken), forward(not_taken));
                while a != f {
                    if a < f {
                        a = join[a];
                    } else {
                        f = join[f];
                    }
                }
                if !uniform(when, &varying) {
                    let mut visited = HashSet::new();
                    let mut todo = vec![taken, not_taken];
                    while let Some(q) = todo.pop() {
                        if rank[q] != a && visited.insert(q) {
                            divergent[q] = true;
                            todo.extend(blocks[q].exit.successors());
                        }
                    }
                }
                a
            }
            LaneExit::End | LaneExit::Indirect => end,
        };
    }
    Analysis { divergent, varying }
}

fn barriers_for(p: &Program, st: &Structured, a: &Analysis) -> Option<HashSet<Site>> {
    let blocks = &p.blocks;
    let order = reverse_postorder(blocks);

    for &b in &order {
        for (i, inst) in blocks[b].insts.iter().enumerate() {
            let name = inst.name();
            if a.divergent[b] && (matches!(inst.fields, Fields::Ds { .. }) || name == "s_barrier") {
                return None;
            }
            // An `s_barrier` with a non-uniform guard is divergent even in a
            // reconverged block: only some threads execute it. A `ds_*` with
            // a non-uniform inline guard is fine when `!divergent[b]`: the
            // inserted group fence runs before `begin_guarded`, so every
            // thread that reached the block hits it in lockstep.
            if name == "s_barrier" && st.guards[b][i].is_some_and(|g| !a.uniform(st, g)) {
                return None;
            }
        }
    }
    Some(fence_sites(p, a, false))
}

/// Fences for LDS accesses that sit in group-uniform blocks only. Divergent
/// `ds_*` are skipped (never insert a barrier there) so Wave64-on-Wave32
/// emulation can still sync the uniform part of the group's own LDS.
pub(super) fn uniform_split_barriers(
    p: &Program,
    _st: &Structured,
    a: &Analysis,
) -> HashSet<Site> {
    fence_sites(p, a, true)
}

fn fence_sites(p: &Program, a: &Analysis, skip_divergent: bool) -> HashSet<Site> {
    let blocks = &p.blocks;
    let order = reverse_postorder(blocks);
    // Order RAW, WAR and WAW, including dependencies across loop edges.
    // Otherwise a wave can reuse a tile while the other is still reading it.
    let mut out = vec![0u8; blocks.len()];
    let mut barriers = HashSet::new();
    loop {
        let mut changed = false;
        for &b in &order {
            let mut pending = blocks[b].preds.iter().fold(0, |p, &q| p | out[q]);
            for (i, inst) in blocks[b].insts.iter().enumerate() {
                if inst.name() == "s_barrier" {
                    pending = 0;
                } else if matches!(inst.fields, Fields::Ds { .. }) {
                    if skip_divergent && a.divergent[b] {
                        continue;
                    }
                    let read_only = inst.name().starts_with("ds_read");
                    if pending & 2 != 0 || (!read_only && pending != 0) {
                        barriers.insert((b, i));
                    }
                    pending = if read_only { 1 } else { 2 };
                }
            }
            changed |= out[b] != pending;
            out[b] = pending;
        }
        if !changed {
            return barriers;
        }
    }
}

/// Values whose contents may differ between host waves. A scalar register
/// alone proves nothing: read-first-lane, for example, writes an SGPR.
fn varying_values(p: &Program, order: &[usize], uniform_control: bool) -> HashSet<Value> {
    let mut varying: HashSet<_> = p
        .inputs
        .iter()
        .filter_map(|&(reg, v)| (!matches!(reg, Reg::S(0..=105))).then_some(v))
        .collect();
    loop {
        let count = varying.len();
        for &b in order {
            for phi in &p.blocks[b].phis {
                if !uniform_control || phi.args.iter().any(|v| varying.contains(v)) {
                    varying.insert(phi.value);
                }
            }
            for i in &p.blocks[b].insts {
                if !matches!(i.fields, Fields::Sop { .. } | Fields::Smem { .. })
                    || i.reads.iter().any(|r| varying.contains(&r.1))
                {
                    varying.extend(i.writes.iter().map(|w| w.1));
                }
            }
        }
        if count == varying.len() {
            return varying;
        }
    }
}
