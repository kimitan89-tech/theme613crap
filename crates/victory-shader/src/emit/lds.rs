use std::collections::HashSet;

use crate::gfx10::Fields;
use crate::ir::{Program, Reg, Value, reverse_postorder};
use crate::resource::Site;
use crate::structure::{Atom, LaneExit, Structured};

/// Fences needed to split a single console wave across host waves. None means
/// at least one access or existing barrier cannot be proved group-uniform.
pub(super) fn split_barriers(p: &Program, st: &Structured) -> Option<HashSet<Site>> {
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
    let uniform_condition = |g| uniform(g, &varying);
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
                if !uniform_condition(when) {
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

    for &b in &order {
        for (i, inst) in blocks[b].insts.iter().enumerate() {
            if (matches!(inst.fields, Fields::Ds { .. }) || inst.name() == "s_barrier")
                && (divergent[b] || st.guards[b][i].is_some_and(|g| !uniform_condition(g)))
            {
                return None;
            }
        }
    }
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
            return Some(barriers);
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
