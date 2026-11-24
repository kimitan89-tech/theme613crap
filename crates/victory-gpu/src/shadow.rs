//! Host copies of title buffers (decision 0007, step 3 and "Copies of title
//! memory"). Title buffers are title memory, which the host GPU cannot
//! read, so a buffer a draw uses gets a persistent host buffer of its size
//! on first use (system memory shaders may also write), and the ranges each
//! submission reads (vertex, index and constant data, buffer views) are
//! brought up to date from title memory before it runs: the 64 KiB pages
//! the title wrote since they were last copied ([`crate::watch`]: write
//! watching, or read-only guards on physical-page views), all of it where
//! neither works. A title GPU address inside the buffer is the same offset
//! in its copy. Ranges shaders write go back to title memory after the
//! submission ([`Shadow::gpu_writes`]).
//!
//! shortcut: only the ranges draws are known to read are brought up to
//! date (vertex and index buffer views, constant buffers in tables, root
//! descriptors, buffer views in tables); buffers read through directly
//! indexed heaps keep what their first use copied.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex};

use windows::Win32::Graphics::Direct3D12::ID3D12Resource;

use crate::host::{HostList, host};
use crate::watch::PAGE;

/// A title buffer's host copy.
pub(crate) struct Shadow {
    pub(crate) buffer: ID3D12Resource,
    data: *mut u8,
    gpu: u64,
    /// Where the title buffer starts and how long it is.
    start: u64,
    len: u64,
    /// Per page of title memory from `start`'s on: the write-watch asking
    /// after which it was last copied (0: never).
    copied: Mutex<Vec<u64>>,
    /// Whether the memory is write-watched: unknown, yes, no.
    watched: AtomicU8,
    /// Ranges shaders of a submission not yet copied back may be writing:
    /// title memory there is older than the copy.
    pending: Mutex<Vec<(usize, usize)>>,
    /// The copy's bytes before the submission ran, as sorted non-overlapping
    /// intervals covering only the ranges shaders write: what the GPU wrote
    /// differs. Keeping disjoint intervals avoids snapshotting huge untouched
    /// gaps across the title's 174 MB GPU heap.
    before: Mutex<Vec<(usize, Vec<u8>)>>,
}

/// Snapshot memory kept for reuse (already touched, so no page faults).
static POOL: Mutex<Vec<Vec<u8>>> = Mutex::new(Vec::new());

/// An empty buffer with room for `n` bytes, from the pool if it has one.
fn pooled(n: usize) -> Vec<u8> {
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    let fit = pool
        .iter()
        .enumerate()
        .filter(|(_, v)| v.capacity() >= n)
        .min_by_key(|(_, v)| v.capacity())
        .map(|(i, _)| i);
    let mut v = match fit {
        Some(i) => pool.swap_remove(i),
        None => Vec::with_capacity(n),
    };
    v.clear();
    v
}

/// Returns a snapshot buffer to the pool (at most 8 kept, the largest).
fn give_back(v: Vec<u8>) {
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    pool.push(v);
    if pool.len() > 8 {
        pool.sort_by_key(|v| std::cmp::Reverse(v.capacity()));
        pool.truncate(8);
    }
}

const NOT_WATCHED: u8 = 2;

// SAFETY: D3D12 resources are free-threaded; `data` is a persistent mapping
// written only on the submitting thread.
unsafe impl Send for Shadow {}
// SAFETY: as above.
unsafe impl Sync for Shadow {}

impl Shadow {
    /// The host GPU address of a title address inside the buffer.
    pub(crate) fn gpu(&self, address: u64) -> u64 {
        self.gpu + (address - self.start)
    }

    /// Bytes from a title address to the buffer's end.
    pub(crate) fn left(&self, address: u64) -> u64 {
        (self.start + self.len).saturating_sub(address)
    }

    /// Offset of a title address from the buffer's start.
    pub(crate) fn offset(&self, address: u64) -> u64 {
        address - self.start
    }

    /// Brings `len` bytes at `address` up to date when `list` is submitted,
    /// and keeps the copy alive until the list is reset.
    pub(crate) fn refresh(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        let len = len.min(self.start + self.len - address);
        list.defer_refresh(self, address as usize, len as usize);
    }

    fn new(buffer: ID3D12Resource, data: *mut u8, gpu: u64, start: u64, len: u64) -> Shadow {
        let pages = (start + len).div_ceil(PAGE as u64) - start / PAGE as u64;
        Shadow {
            buffer,
            data,
            gpu,
            start,
            len,
            copied: Mutex::new(vec![0; pages as usize]),
            watched: AtomicU8::new(0),
            pending: Mutex::default(),
            before: Mutex::default(),
        }
    }

    /// `[address, address + len)` cut to the buffer, as offsets in the copy.
    fn inside(&self, address: usize, len: usize) -> (usize, usize) {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let from = address.clamp(start, end);
        (from - start, (address + len).clamp(from, end) - start)
    }

    /// Shaders of `list` may write `[address, address + len)` of the copy
    /// (unordered access): after the list's submission has run, the range
    /// is copied back into title memory, where the title reads it; until
    /// then no refresh copies title memory over it. Call after the
    /// [`Shadow::refresh`] that gives the shaders the title's data.
    ///
    /// shortcut: what shaders wrote is found by keeping the range's bytes
    /// before the submission (a copy per declared range and submission);
    /// a GPU-side dirty mask would save that if big ranges make it slow.
    pub(crate) fn gpu_writes(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        let len = len.min(self.left(address));
        list.defer_gpu_write(self, address as usize, len as usize);
    }

    /// Whether `[address, address + len)` is already covered by a pending
    /// shader-write range in this submission.
    pub(crate) fn covers_pending(&self, address: usize, len: usize) -> bool {
        if len == 0 {
            return true;
        }
        let end = address.saturating_add(len);
        self.pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .any(|&(a, e)| a <= address && end <= e)
    }

    /// At submission, after the range's refresh: a submitted list's shaders
    /// write this range. Its bytes are kept in sorted non-overlapping
    /// intervals to tell later what the GPU wrote, without bridging disjoint
    /// ranges across the title's 174 MB heap, in reused memory (fresh
    /// snapshots cost ~2.3 million page faults per world frame).
    pub(crate) fn writing(&self, address: usize, len: usize) {
        let range = (address, address.saturating_add(len));
        {
            let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
            if !pending.contains(&range) {
                pending.push(range);
            }
        }
        let (from, to) = self.inside(address, len);
        if from >= to {
            return;
        }
        // SAFETY: inside the mapping.
        let host = |a: usize, b: usize| unsafe { std::slice::from_raw_parts(self.data.add(a), b - a) };
        let mut intervals = self.before.lock().unwrap_or_else(|e| e.into_inner());
        merge_snapshot(&mut intervals, from, to, host);
    }

    /// After the submission ran: copies back all coalesced shader-written
    /// ranges for this copy in one pass (reusing one 64 KiB scratch buffer
    /// and issuing one write-watch sync), then clears `pending` and returns
    /// snapshot buffers to the pool.
    pub(crate) fn copy_back_ranges(&self, ranges: &[(usize, usize)]) -> usize {
        let mut intervals = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let mut written = 0;
        let start = self.start as usize;
        const PIECE: usize = 64 << 10;
        let mut title_buf = vec![0u8; PIECE];
        let mut synced = Vec::new();
        let mut sync_first = usize::MAX;
        let mut sync_last = 0usize;

        for &(address, end) in ranges {
            let len = end.saturating_sub(address);
            let (from, to) = self.inside(address, len);
            if from >= to {
                continue;
            }
            // A coalesced range may span multiple contiguous/near intervals;
            // walk every snapshot interval overlapping `[from, to)`.
            let mut pos = intervals.partition_point(|(at, bytes)| at + bytes.len() <= from);
            while pos < intervals.len() {
                let (taken, before_all) = &intervals[pos];
                let iv_end = *taken + before_all.len();
                if *taken >= to {
                    break;
                }
                let sub_from = from.max(*taken);
                let sub_to = to.min(iv_end);
                let before = &before_all[sub_from - *taken..sub_to - *taken];
                let mut at = sub_from;
                while at < sub_to {
                    let n = (PIECE - (start + at) % PIECE).min(sub_to - at);
                    // SAFETY: inside the mapping; only the submitting thread
                    // writes it.
                    let host = unsafe { std::slice::from_raw_parts_mut(self.data.add(at), n) };
                    let old = &before[at - sub_from..at - sub_from + n];
                    if let Some((d_lo, d_hi)) = diff_bounds(host, old) {
                        let title = &mut title_buf[..n];
                        if crate::host::read_into((start + at) as *const u8, title) {
                            // Merge GPU writes into title bytes only inside the
                            // differing span `[d_lo, d_hi)`, while keeping the
                            // whole 64 KiB piece in `host` synced with `title`.
                            let mut w_lo = usize::MAX;
                            let mut w_hi = 0usize;
                            for i in d_lo..d_hi {
                                let h = host[i];
                                if h != old[i] && title[i] != h {
                                    title[i] = h;
                                    w_lo = w_lo.min(i);
                                    w_hi = i + 1;
                                }
                            }
                            host.copy_from_slice(title);
                            if w_lo < w_hi {
                                let write_len = w_hi - w_lo;
                                crate::watch::writing(start + at + w_lo, write_len);
                                crate::host::write_mapped(
                                    (start + at + w_lo) as *mut u8,
                                    title[w_lo..w_hi].as_ptr(),
                                    write_len,
                                );
                                written += write_len;
                            }
                            let page = (start + at) / PAGE;
                            if (start + at).is_multiple_of(PAGE) && n == PAGE {
                                synced.push(page);
                                sync_first = sync_first.min(page);
                                sync_last = sync_last.max(page + 1);
                            }
                        }
                    }
                    at += n;
                }
                pos += 1;
            }
        }
        // Pages made equal to title memory are current in the copy, so our
        // own write is not taken for the title's; others keep what they had.
        if sync_first < sync_last
            && !synced.is_empty()
            && self.watched.load(Ordering::Relaxed) != NOT_WATCHED
            && let Some((now, _)) = crate::watch::writes(sync_first * PAGE, sync_last * PAGE, true)
        {
            let base = start / PAGE;
            let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
            for page in synced {
                if let Some(slot) = copied.get_mut(page - base) {
                    *slot = now;
                }
            }
        }
        // Every submitted step that declared writes on this shadow is done
        // (one submission runs at a time).
        self.pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
        for (_, bytes) in intervals.drain(..) {
            give_back(bytes);
        }
        written
    }

    /// At submission: copies what changed of `[address, address + len)`
    /// since it was last copied (all of it if the memory is not
    /// write-watched). Returns the bytes copied.
    pub(crate) fn update(&self, address: usize, len: usize) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (address.max(start), (address + len).min(end));
        if from >= to {
            return 0;
        }
        // Expand to 64 KiB watched-page boundaries so partial-page static
        // ranges (small VBs/IBs/CBVs) get marked current in `copied` instead
        // of staying stale and re-copying on every submission. Never overlap
        // any range shaders are writing in this submission.
        let page_from = (from & !(PAGE - 1)).max(start);
        let page_to = to.next_multiple_of(PAGE).min(end);
        let mut busy: Vec<(usize, usize)> = self
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .copied()
            .filter(|&(a, e)| a < page_to && e > page_from)
            .collect();
        if busy.is_empty() {
            return self.update_free(page_from, page_to);
        }
        // Around ranges shaders are writing: clamp to `[from, to)` near busy
        // boundaries so we never overwrite a shader-written range on the same
        // 64 KiB page.
        busy.retain(|&(a, e)| a < to && e > from);
        if busy.is_empty() {
            return self.update_free(from, to);
        }
        busy.sort_unstable();
        let (mut at, mut bytes) = (from, 0);
        for (a, e) in busy {
            if a > at {
                bytes += self.update_free(at, a);
            }
            at = at.max(e);
            if at >= to {
                return bytes;
            }
        }
        if at < to {
            bytes += self.update_free(at, to);
        }
        bytes
    }

    /// [`Shadow::update`] of `[from, to)` (inside the buffer) where no
    /// shader writes.
    fn update_free(&self, from: usize, to: usize) -> usize {
        if self.watched.load(Ordering::Relaxed) != NOT_WATCHED {
            if let Some((now, written)) = crate::watch::writes(from, to, true) {
                self.watched.store(1, Ordering::Relaxed);
                let copied = self.copy_stale(from, to, now, &written);
                #[cfg(feature = "diagnostics")]
                {
                    use crate::copystat::{C, add};
                    add(C::Checked, (to - from) as u64);
                    add(C::Unchanged, (to - from).saturating_sub(copied) as u64);
                }
                return copied;
            }
            self.watched.store(NOT_WATCHED, Ordering::Relaxed);
        }
        self.copy(from, to);
        to - from
    }

    /// Copies the pages of `[from, to)` written after their last copy
    /// (`written`: last writes by page, in page order; `now`: this
    /// asking), whole pages within the buffer.
    fn copy_stale(&self, from: usize, to: usize, now: u64, written: &[(usize, u64)]) -> usize {
        let base = self.start as usize / PAGE;
        let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
        let mut writes = written.iter().peekable();
        let (mut bytes, mut run) = (0, None);
        let last = (to - 1) / PAGE;
        for page in from / PAGE..=last {
            let write = match writes.peek() {
                Some(&&(p, w)) if p == page => {
                    writes.next();
                    w
                }
                _ => 0,
            };
            let at = &mut copied[page - base];
            if *at == 0 || write > *at {
                // A page only partly in the range is copied in part: it
                // stays stale for the rest.
                let (start, end) = (self.start as usize, (self.start + self.len) as usize);
                if (page * PAGE).max(start) >= from && ((page + 1) * PAGE).min(end) <= to {
                    *at = now;
                }
                run.get_or_insert(page);
            } else if let Some(first) = run.take() {
                bytes += self.copy((first * PAGE).max(from), (page * PAGE).min(to));
            }
        }
        // Cut to the range: next to it shaders may be writing the copy.
        if let Some(first) = run {
            bytes += self.copy((first * PAGE).max(from), ((last + 1) * PAGE).min(to));
        }
        bytes
    }

    /// Copies title memory `[from, to)`, cut to the buffer, into the copy.
    fn copy(&self, from: usize, to: usize) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (from.max(start), to.min(end));
        // SAFETY: inside the mapping (`from` and `to` are within the buffer).
        let at = unsafe { self.data.add(from - start) };
        crate::host::copy_mapped(at, from as *const u8, to - from, false);
        to - from
    }
}

/// Returns the tightest byte span `(lo, hi)` where `a` and `b` differ,
/// scanning 8 bytes at a time from both ends so unchanged 64 KiB chunks skip
/// quickly without byte-by-byte work.
fn diff_bounds(a: &[u8], b: &[u8]) -> Option<(usize, usize)> {
    debug_assert_eq!(a.len(), b.len());
    let len = a.len();
    let mut lo = 0usize;
    while lo + 8 <= len {
        let wa = u64::from_ne_bytes(a[lo..lo + 8].try_into().unwrap());
        let wb = u64::from_ne_bytes(b[lo..lo + 8].try_into().unwrap());
        if wa != wb {
            break;
        }
        lo += 8;
    }
    while lo < len && a[lo] == b[lo] {
        lo += 1;
    }
    if lo == len {
        return None;
    }
    let mut hi = len;
    while hi >= lo + 8 {
        let wa = u64::from_ne_bytes(a[hi - 8..hi].try_into().unwrap());
        let wb = u64::from_ne_bytes(b[hi - 8..hi].try_into().unwrap());
        if wa != wb {
            break;
        }
        hi -= 8;
    }
    while hi > lo && a[hi - 1] == b[hi - 1] {
        hi -= 1;
    }
    Some((lo, hi))
}

/// Merges `[from, to)` into sorted non-overlapping snapshot `intervals`,
/// reading only newly covered regions via `host` and preserving bytes already
/// captured in existing intervals.
fn merge_snapshot<'a>(
    intervals: &mut Vec<(usize, Vec<u8>)>,
    from: usize,
    to: usize,
    host: impl Fn(usize, usize) -> &'a [u8],
) {
    if from >= to {
        return;
    }
    let start = intervals.partition_point(|(at, bytes)| at + bytes.len() < from);
    if let Some((at, bytes)) = intervals.get(start)
        && *at <= from
        && to <= at + bytes.len()
    {
        return;
    }
    let mut end = start;
    while end < intervals.len() && intervals[end].0 <= to {
        end += 1;
    }
    if start == end {
        let mut snap = pooled(to - from);
        snap.extend_from_slice(host(from, to));
        intervals.insert(start, (from, snap));
        return;
    }
    if end == start + 1 && intervals[start].0 <= from {
        let (at, bytes) = &mut intervals[start];
        let cur_end = *at + bytes.len();
        if to > cur_end {
            bytes.extend_from_slice(host(cur_end, to));
        }
        return;
    }
    let merged_start = from.min(intervals[start].0);
    let merged_end = to.max(intervals[end - 1].0 + intervals[end - 1].1.len());
    let mut grown = pooled(merged_end - merged_start);
    let mut cursor = merged_start;
    for (at, bytes) in intervals.drain(start..end) {
        if at > cursor {
            grown.extend_from_slice(host(cursor, at));
        }
        grown.extend_from_slice(&bytes);
        cursor = at + bytes.len();
        give_back(bytes);
    }
    if merged_end > cursor {
        grown.extend_from_slice(host(cursor, merged_end));
    }
    intervals.insert(start, (merged_start, grown));
}

struct Entry {
    end: u64,
    shadow: Option<Arc<Shadow>>,
}

/// Title buffers by (start address, resource), with their copy once made.
static BUFFERS: Mutex<BTreeMap<(u64, usize), Entry>> = Mutex::new(BTreeMap::new());

fn buffers() -> std::sync::MutexGuard<'static, BTreeMap<(u64, usize), Entry>> {
    BUFFERS.lock().unwrap_or_else(|e| e.into_inner())
}

/// A title buffer exists at `start` (`len` bytes).
pub(crate) fn register(resource: usize, start: u64, len: u64) {
    buffers().insert(
        (start, resource),
        Entry {
            end: start + len,
            shadow: None,
        },
    );
}

/// The title released it; lists that still hold its copy keep it alive.
pub(crate) fn unregister(resource: usize, start: u64) {
    buffers().remove(&(start, resource));
}

/// The copy of the innermost title buffer holding `[address, address +
/// len)`, made on first use.
pub(crate) fn find(address: u64, len: u64) -> Option<Arc<Shadow>> {
    let end = address.checked_add(len.max(1))?;
    let mut map = buffers();
    let key = map
        .range(..=(address, usize::MAX))
        .rev()
        .take(64)
        .find(|(_, e)| e.end >= end)
        .map(|(k, _)| *k);
    let Some(key) = key else {
        drop(map);
        return loose(address, end);
    };
    let entry = map.get_mut(&key)?;
    if entry.shadow.is_none() {
        let len = entry.end - key.0;
        // Rounded up so that views ending in the buffer's last 256 bytes fit.
        // Shaders write some buffers (unordered access), so not upload memory.
        let (buffer, data) = host()?.shared_buffer(len.next_multiple_of(1 << 16))?;
        crate::host::name(&buffer, || format!("title buffer {:#x}+{len:#x}", key.0));
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        if len >= 16 << 20 {
            tracing::info!(
                start = format_args!("{:#x}", key.0),
                len,
                "large buffer copied to the host"
            );
        } else {
            tracing::debug!(
                start = format_args!("{:#x}", key.0),
                len,
                "buffer copied to the host"
            );
        }
        entry.shadow = Some(Arc::new(Shadow::new(buffer, data, gpu, key.0, len)));
    }
    entry.shadow.clone()
}

/// Titles also use memory outside any buffer object as GPU memory (Grand
/// Theft Auto VI's interface vertices, from its own allocator). Such memory
/// gets copies of the aligned windows it touches.
/// shortcut: copies of windows are never freed; key them to the title's
/// allocations if memory use grows.
const WINDOW: u64 = 4 << 20;

/// Copies of title memory outside buffers, by (start, length).
static LOOSE: Mutex<BTreeMap<(u64, u64), Arc<Shadow>>> = Mutex::new(BTreeMap::new());

fn loose(address: u64, end: u64) -> Option<Arc<Shadow>> {
    let start = address & !(WINDOW - 1);
    let len = end.next_multiple_of(WINDOW) - start;
    let mut map = LOOSE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(shadow) = map.get(&(start, len)) {
        return Some(shadow.clone());
    }
    let (buffer, data) = host()?.shared_buffer(len)?;
    crate::host::name(&buffer, || format!("title memory {start:#x}+{len:#x}"));
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    tracing::debug!(
        start = format_args!("{start:#x}"),
        len,
        "loose memory copied to the host"
    );
    let shadow = Arc::new(Shadow::new(buffer, data, gpu, start, len));
    map.insert((start, len), shadow.clone());
    Some(shadow)
}

/// For device-removed reports: the copy of title memory whose host range
/// starts closest below host GPU address `va`, and where `va` is in it.
pub(crate) fn near(va: u64) -> String {
    let mut all: Vec<(u64, u64, u64)> = buffers()
        .values()
        .filter_map(|e| e.shadow.as_deref().map(|s| (s.gpu, s.start, s.len)))
        .collect();
    all.extend(
        LOOSE
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .map(|s| (s.gpu, s.start, s.len)),
    );
    let below = all.iter().filter(|c| c.0 <= va).max();
    let above = all.iter().filter(|c| c.0 > va).min();
    let mut text = match below {
        Some(&(gpu, start, len)) => format!(
            "; the nearest title memory copy below it: {start:#x}+{len:#x} at host {gpu:#x}, offset {:#x}",
            va - gpu
        ),
        None => "; no title memory copy below it".into(),
    };
    if let Some(&(gpu, start, len)) = above {
        text.push_str(&format!(
            "; the nearest above: {start:#x}+{len:#x} at host {gpu:#x} ({} copies in all)",
            all.len()
        ));
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshots_keep_disjoint_ranges_separate_and_preserve_prior_bytes() {
        let mut backing: Vec<u8> = (0..128u8).collect();
        let mut intervals: Vec<(usize, Vec<u8>)> = Vec::new();
        merge_snapshot(&mut intervals, 10, 20, |a, b| &backing[a..b]);
        merge_snapshot(&mut intervals, 100, 110, |a, b| &backing[a..b]);
        assert_eq!(intervals.len(), 2);
        assert_eq!(intervals[0].0, 10);
        assert_eq!(intervals[1].0, 100);

        // Mutate backing memory inside already-snapshotted [10, 20); extending
        // right and then bridging must keep the original snapshot bytes.
        backing[15] = 250;
        merge_snapshot(&mut intervals, 15, 25, |a, b| &backing[a..b]);
        assert_eq!(intervals.len(), 2);
        assert_eq!(intervals[0].0, 10);
        assert_eq!(intervals[0].1.len(), 15);
        assert_eq!(intervals[0].1[5], 15);
        assert_eq!(intervals[0].1[10], 20);

        // Sub-range is a no-op.
        merge_snapshot(&mut intervals, 12, 18, |a, b| &backing[a..b]);
        assert_eq!(intervals.len(), 2);

        // Bridge [10, 25) and [100, 110) with [5, 115).
        backing[105] = 251;
        merge_snapshot(&mut intervals, 5, 115, |a, b| &backing[a..b]);
        assert_eq!(intervals.len(), 1);
        assert_eq!(intervals[0].0, 5);
        assert_eq!(intervals[0].1.len(), 110);
        assert_eq!(intervals[0].1[10], 15); // original offset 15
        assert_eq!(intervals[0].1[100], 105); // original offset 105
        assert_eq!(intervals[0].1[50], 55); // newly covered middle
    }

    #[test]
    fn diff_bounds_finds_exact_changed_span() {
        let a = vec![7u8; 4096];
        let mut b = a.clone();
        assert_eq!(diff_bounds(&a, &b), None);
        b[123] = 9;
        assert_eq!(diff_bounds(&a, &b), Some((123, 124)));
        b[3999] = 11;
        assert_eq!(diff_bounds(&a, &b), Some((123, 4000)));
        b[0] = 1;
        b[4095] = 2;
        assert_eq!(diff_bounds(&a, &b), Some((0, 4096)));
    }
}
