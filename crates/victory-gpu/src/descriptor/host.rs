//! Host descriptor heaps behind the title's shader-visible heaps (decision
//! 0007, step 3). Translated shaders index the heaps directly at the
//! title's descriptor positions, so each host heap keeps the title heap's
//! order: slot `i` is made from what the title wrote at slot `i`, when it
//! writes it (view creation and descriptor copies) and again when a draw
//! reads it through a table and finds the slot changed.

use std::collections::HashMap;
use std::sync::Mutex;

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use super::{VIEW_TAG, View, view_at};
use crate::host::host;

/// Bitset allocator for dynamic descriptor slots: O(1) fast-fail when full
/// and word-level scanning instead of a linear byte scan.
struct DynamicPool {
    total: u32,
    free: u32,
    next: u32,
    /// Any request `>= min_failed` is known not to fit until slots are freed.
    min_failed: u32,
    /// Bit `i` == 1 means slot `i` is taken; padding bits past `total` are 1.
    bits: Vec<u64>,
}

impl DynamicPool {
    fn new(total: u32) -> Self {
        let words = (total as usize).div_ceil(64);
        let mut bits = vec![0u64; words];
        let rem = total as usize % 64;
        if rem != 0 && let Some(last) = bits.last_mut() {
            *last = !0u64 << rem;
        }
        DynamicPool {
            total,
            free: total,
            next: 0,
            min_failed: u32::MAX,
            bits,
        }
    }

    fn alloc(&mut self, n: u32) -> Option<u32> {
        if n == 0 || n > self.free || n >= self.min_failed {
            return None;
        }
        let n_usize = n as usize;
        let total = self.total as usize;
        let next = self.next as usize;
        let found = find_free_run(&self.bits, next, total, n_usize).or_else(|| {
            if next == 0 {
                None
            } else {
                find_free_run(&self.bits, 0, (next + n_usize - 1).min(total), n_usize)
            }
        });
        let Some(first) = found else {
            self.min_failed = self.min_failed.min(n);
            return None;
        };
        set_bits(&mut self.bits, first, first + n_usize, true);
        self.free -= n;
        self.next = ((first + n_usize) % total) as u32;
        Some(first as u32)
    }

    fn release(&mut self, at: u32, n: u32) {
        let start = at.min(self.total) as usize;
        let end = at.saturating_add(n).min(self.total) as usize;
        if start >= end {
            return;
        }
        let cleared = set_bits(&mut self.bits, start, end, false);
        self.free += cleared;
        if cleared > 0 {
            self.min_failed = u32::MAX;
        }
    }
}

/// Finds the first run of `n` zero bits inside `[start, end)` without wrapping.
fn find_free_run(bits: &[u64], start: usize, end: usize, n: usize) -> Option<usize> {
    let mut i = start;
    let mut run = 0usize;
    while i < end {
        let w = i / 64;
        let b = i % 64;
        let word = bits[w];
        if b == 0 && word == u64::MAX {
            run = 0;
            i += 64;
            continue;
        }
        if b == 0 && word == 0 {
            let avail = (end - i).min(64);
            if run + avail >= n {
                return Some(i - run);
            }
            run += avail;
            i += avail;
            continue;
        }
        let shifted = word >> b;
        let rem = (end - i).min(64 - b);
        if shifted & 1 != 0 {
            let ones = (shifted.trailing_ones() as usize).min(rem);
            run = 0;
            i += ones;
        } else {
            let zeros = (shifted.trailing_zeros() as usize).min(rem);
            if run + zeros >= n {
                return Some(i - run);
            }
            run += zeros;
            i += zeros;
        }
    }
    None
}

/// Sets or clears `[start, end)` in `bits`; returns the number of bits flipped.
fn set_bits(bits: &mut [u64], start: usize, end: usize, value: bool) -> u32 {
    let mut i = start;
    let mut changed = 0u32;
    while i < end {
        let w = i / 64;
        let b = i % 64;
        let count = (end - i).min(64 - b);
        let mask = if count == 64 {
            u64::MAX
        } else {
            ((1u64 << count) - 1) << b
        };
        let prev = bits[w];
        if value {
            changed += (mask & !prev).count_ones();
            bits[w] = prev | mask;
        } else {
            changed += (mask & prev).count_ones();
            bits[w] = prev & !mask;
        }
        i += count;
    }
    changed
}

/// Persistent fresh tables shared across open command lists by exact
/// descriptor records, reference-counted until every list holding them resets.
#[derive(Default)]
struct SharedTables {
    by_key: HashMap<Vec<[u32; 3]>, (u32, u32, u32)>,
    by_slot: HashMap<u32, Vec<[u32; 3]>>,
}

pub(crate) struct HostHeap {
    pub(crate) heap: ID3D12DescriptorHeap,
    cpu: usize,
    step: u32,
    /// Slots mirroring the title's; view heaps have `dynamic` more.
    count: u32,
    /// Dynamic slots past `count` (none in sampler heaps): [`DYNAMIC`]
    /// where the host takes them, else [`MIN_DYNAMIC`].
    dynamic: u32,
    sampler: bool,
    /// The record (view index and generation) each slot was last made from.
    made: Mutex<Vec<u64>>,
    /// The host resource each slot views, held while the slot does: shaders
    /// index heaps directly, so a slot the title has not rewritten since it
    /// released the resource may still be read, and a freed host resource
    /// read there is a GPU page fault (the console reads stale memory).
    kept: Mutex<Vec<Option<ID3D12Resource>>>,
    /// Dynamic slot allocator (taken by lists not yet reset).
    pool: Mutex<DynamicPool>,
    /// Ref-counted dynamic tables shared across lists (non-scratch tables).
    shared: Mutex<SharedTables>,
}

/// Host heaps' GPU ranges (start, bytes, bytes per slot, sampler), for
/// device-removed reports: a shader indexing a heap past its end faults.
static GPU_RANGES: Mutex<Vec<(u64, u64, u64, bool)>> = Mutex::new(Vec::new());

/// For device-removed reports: where host GPU address `va` lies against
/// the host descriptor heaps.
pub(crate) fn near_heaps(va: u64) -> String {
    let ranges = GPU_RANGES.lock().unwrap_or_else(|e| e.into_inner());
    let mut text = String::new();
    for &(start, bytes, step, sampler) in ranges.iter() {
        let kind = if sampler { "sampler" } else { "view" };
        let place = if va < start {
            format!("{:#x} below", start - va)
        } else if va < start + bytes {
            format!("inside, slot {}", (va - start) / step)
        } else {
            format!("{:#x} past its end (slot {})", va - start - bytes, (va - start) / step)
        };
        text.push_str(&format!("; {kind} heap {start:#x}+{bytes:#x}: {place}"));
    }
    text
}

/// The host's limit for shader-visible view heaps on resource binding
/// tiers 1 and 2; tier 3 hardware may take more.
const PORTABLE_VIEWS: u32 = 1_000_000;

/// Slots past the title's in a view heap, for tables a draw needs made
/// afresh (constant buffers copied per draw). A list holds the slots it
/// took until it is reset (its submissions have run) or released.
/// 2^16 ran out seconds into the world (every world run on an RX 9070 XT,
/// each a device removal shortly after; #59).
const DYNAMIC: u32 = 1 << 18;

/// The dynamic slots of a heap whose host would not take [`DYNAMIC`] more
/// (a capped heap, NVIDIA): reserve enough room for multi-list world frames
/// while keeping static slots in place.
const MIN_DYNAMIC: u32 = 3 << 16;

impl HostHeap {
    pub(crate) fn new(sampler: bool, count: u32) -> Option<Self> {
        let host = host()?;
        let kind = if sampler {
            D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER
        } else {
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV
        };
        let make = |count: u32| -> windows::core::Result<ID3D12DescriptorHeap> {
            // SAFETY: plain heap creation.
            unsafe {
                host.device
                    .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                        Type: kind,
                        NumDescriptors: count,
                        Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                        NodeMask: 0,
                    })
            }
        };
        // The larger dynamic range only where the host takes it on top of
        // every title slot; else the smaller one, and the cap after that.
        let (heap, count, extra) = if sampler {
            match make(count) {
                Ok(heap) => (heap, count, 0),
                Err(e) => {
                    tracing::warn!(error = %e, count, sampler, "no host descriptor heap");
                    return None;
                }
            }
        } else if let Ok(heap) = make(count + DYNAMIC) {
            (heap, count, DYNAMIC)
        } else {
            match make(count + MIN_DYNAMIC) {
                Ok(heap) => (heap, count, MIN_DYNAMIC),
                Err(e) if count + MIN_DYNAMIC > PORTABLE_VIEWS => {
                    // shortcut: descriptors past the host's limit are left
                    // out; a title that reaches them needs remapping.
                    tracing::warn!(error = %e, count, "host view heap capped at {PORTABLE_VIEWS}");
                    (make(PORTABLE_VIEWS).ok()?, PORTABLE_VIEWS - MIN_DYNAMIC, MIN_DYNAMIC)
                }
                Err(e) => {
                    tracing::warn!(error = %e, count, sampler, "no host descriptor heap");
                    return None;
                }
            }
        };
        // SAFETY: a live heap.
        let (cpu, step) = unsafe {
            (
                heap.GetCPUDescriptorHandleForHeapStart().ptr,
                host.device.GetDescriptorHandleIncrementSize(kind),
            )
        };
        tracing::debug!(count, sampler, "host descriptor heap");
        // SAFETY: a live shader-visible heap.
        let gpu = unsafe { heap.GetGPUDescriptorHandleForHeapStart().ptr };
        GPU_RANGES.lock().unwrap_or_else(|e| e.into_inner()).push((
            gpu,
            u64::from(count + extra) * u64::from(step),
            u64::from(step),
            sampler,
        ));
        let h = HostHeap {
            heap,
            cpu,
            step,
            count,
            dynamic: extra,
            sampler,
            made: Mutex::new(vec![0; count as usize]),
            kept: Mutex::new(vec![None; (count + extra) as usize]),
            pool: Mutex::new(DynamicPool::new(extra)),
            shared: Mutex::default(),
        };
        // A slot never written is undefined: shaders reading one (a slot
        // the title wrote in a way we do not know, or a table entry no
        // used-resource entry names) read a wild address. Every slot starts
        // null instead.
        let started = std::time::Instant::now();
        if let Err(e) = h.fill_null(kind, count + extra) {
            tracing::warn!(error = %e, "host descriptor heap not made null");
        }
        tracing::debug!(ms = started.elapsed().as_millis(), "host descriptor heap made null");
        Some(h)
    }

    /// Makes the first `total` slots null: one null slot in a CPU-side heap,
    /// doubled by copies to a block, the block copied over the heap.
    fn fill_null(&self, kind: D3D12_DESCRIPTOR_HEAP_TYPE, total: u32) -> windows::core::Result<()> {
        const BLOCK: u32 = 1 << 16;
        let Some(host) = host() else { return Ok(()) };
        // SAFETY: plain heap creation and copies inside live heaps.
        unsafe {
            let block: ID3D12DescriptorHeap =
                host.device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: kind,
                    NumDescriptors: BLOCK,
                    ..Default::default()
                })?;
            let start = block.GetCPUDescriptorHandleForHeapStart();
            let at = |i: u32| D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: start.ptr + (i * self.step) as usize,
            };
            if self.sampler {
                null_sampler(start);
            } else {
                null_view(start);
            }
            let mut filled = 1;
            while filled < BLOCK {
                let n = filled.min(BLOCK - filled);
                host.device.CopyDescriptorsSimple(n, at(filled), start, kind);
                filled += n;
            }
            let mut done = 0;
            while done < total {
                let n = BLOCK.min(total - done);
                host.device.CopyDescriptorsSimple(n, self.cpu(done), start, kind);
                done += n;
            }
        }
        Ok(())
    }

    /// A plain point sampler at slot `index`, for sampler slots without a
    /// view.
    fn null_sampler(&self, index: u32) {
        null_sampler(self.cpu(index));
    }
}

/// A plain point sampler at `at`.
fn null_sampler(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_SAMPLER_DESC {
        Filter: D3D12_FILTER_MIN_MAG_MIP_POINT,
        AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        MaxAnisotropy: 1,
        ComparisonFunc: D3D12_COMPARISON_FUNC_NEVER,
        ..Default::default()
    };
    // SAFETY: a slot of a sampler heap of ours.
    unsafe { host.device.CreateSampler(&desc, at) };
    #[cfg(feature = "diagnostics")]
    crate::host::check_device_alive(format_args!("CreateSampler null point"));
}

impl HostHeap {
    /// `n` consecutive free dynamic slots (view heaps only), taken until
    /// [`HostHeap::release`]: the first's index.
    pub(crate) fn dynamic(&self, n: u32) -> Option<u32> {
        if self.sampler || n == 0 || n > self.dynamic {
            return None;
        }
        let found = self.pool.lock().unwrap_or_else(|e| e.into_inner()).alloc(n);
        let Some(first) = found else {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| tracing::warn!(n, "no free dynamic descriptor slots: a table is not made"));
            return None;
        };
        Some(self.count + first)
    }

    /// Reuses an already-populated shared dynamic table matching `key`,
    /// bumping its reference count so it stays alive until this caller also
    /// releases it.
    pub(crate) fn acquire_shared(&self, key: &[[u32; 3]]) -> Option<u32> {
        let mut shared = self.shared.lock().unwrap_or_else(|e| e.into_inner());
        let (first, _, refs) = shared.by_key.get_mut(key)?;
        *refs += 1;
        Some(*first)
    }

    /// Registers a freshly populated dynamic table `[first, first + n)` under
    /// `key` for cross-list sharing. If another thread published the same key
    /// first, returns the existing table (ref-counted) and releases `[first,
    /// first + n)`.
    pub(crate) fn publish_shared(&self, key: Vec<[u32; 3]>, first: u32, n: u32) -> u32 {
        let existing = {
            let mut shared = self.shared.lock().unwrap_or_else(|e| e.into_inner());
            if let Some((slot, _, refs)) = shared.by_key.get_mut(&key) {
                *refs += 1;
                Some(*slot)
            } else {
                shared.by_slot.insert(first, key.clone());
                shared.by_key.insert(key, (first, n, 1));
                None
            }
        };
        if let Some(existing) = existing {
            self.release_raw(first, n);
            existing
        } else {
            first
        }
    }

    /// The title slots this heap has host slots for; fewer than the title's
    /// when the host caps its size.
    pub(crate) fn count(&self) -> u32 {
        self.count
    }

    /// Gives back `n` dynamic slots from `first` (an index [`dynamic`] or
    /// [`acquire_shared`] returned).
    ///
    /// [`dynamic`]: HostHeap::dynamic
    pub(crate) fn release(&self, first: u32, n: u32) {
        let should_free = {
            let mut shared = self.shared.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(key) = shared.by_slot.get(&first).cloned() {
                let mut drop_entry = false;
                if let Some((_, _, refs)) = shared.by_key.get_mut(&key) {
                    *refs = refs.saturating_sub(1);
                    drop_entry = *refs == 0;
                }
                if drop_entry {
                    shared.by_key.remove(&key);
                    shared.by_slot.remove(&first);
                    true
                } else {
                    false
                }
            } else {
                true
            }
        };
        if should_free {
            self.release_raw(first, n);
        }
    }

    fn release_raw(&self, first: u32, n: u32) {
        let Some(at) = first.checked_sub(self.count) else {
            return;
        };
        // Drop kept resource references before handing slots back so another
        // allocator cannot race with clearing them.
        {
            let mut kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
            let end = first.saturating_add(n).min(self.count + self.dynamic) as usize;
            if let Some(slots) = kept.get_mut(first as usize..end) {
                for slot in slots {
                    *slot = None;
                }
            }
        }
        self.pool
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .release(at, n);
    }

    /// Makes slot `index` (a dynamic one) from `view`; a null view if the
    /// host cannot.
    pub(crate) fn make_at(&self, index: u32, view: &View) {
        let kept = make(view, self.cpu(index));
        if kept.is_none() {
            null_for(view, self.cpu(index));
        }
        self.keep(index, kept.flatten());
    }

    /// Makes slot `index` read nothing: a null view, or a plain sampler.
    pub(crate) fn make_null(&self, index: u32) {
        if self.sampler {
            self.null_sampler(index);
        } else {
            null_view(self.cpu(index));
        }
        self.keep(index, None);
    }

    /// Slot `index` now views `resource` (or nothing of the host's).
    fn keep(&self, index: u32, resource: Option<ID3D12Resource>) {
        let mut kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(slot) = kept.get_mut(index as usize) {
            *slot = resource;
        }
    }

    /// Makes slot `index` (a dynamic one) a constant buffer view of host
    /// memory (either the list's own scratch upload memory, or a persistent
    /// buffer copy kept alive in `kept`).
    pub(crate) fn constant_at(
        &self,
        index: u32,
        gpu: u64,
        size: u32,
        kept: Option<ID3D12Resource>,
    ) {
        self.keep(index, kept);
        let Some(host) = host() else { return };
        let at = self.cpu(index);
        // D3D12 rejects a zero GPU address and non-256 alignment with device removal.
        if gpu == 0 || !gpu.is_multiple_of(256) {
            let desc = D3D12_CONSTANT_BUFFER_VIEW_DESC::default();
            // SAFETY: a slot of our heap; a zero location makes a null view.
            unsafe {
                host.device.CreateConstantBufferView(Some(&desc), at);
            }
            #[cfg(feature = "diagnostics")]
            crate::host::check_device_alive(format_args!(
                "CreateConstantBufferView null (bad gpu={gpu:#x} size={size})"
            ));
            return;
        }
        let desc = D3D12_CONSTANT_BUFFER_VIEW_DESC {
            BufferLocation: gpu,
            SizeInBytes: size.max(1).next_multiple_of(256).min(65536),
        };
        // SAFETY: a slot of our heap; the memory is kept alive by the caller or `kept`.
        unsafe {
            host.device.CreateConstantBufferView(Some(&desc), at);
        }
        #[cfg(feature = "diagnostics")]
        crate::host::check_device_alive(format_args!(
            "CreateConstantBufferView gpu={gpu:#x} size={}",
            desc.SizeInBytes
        ));
    }

    fn cpu(&self, index: u32) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: self.cpu + index as usize * self.step as usize,
        }
    }

    /// Makes host slot `index` from the record the title has at `handle`
    /// if it changed since, and returns the view recorded there. Buffer
    /// views are made only for draws (`for_draw`), which also copy their
    /// data: a host copy of a buffer is made on first use.
    pub(crate) fn mirror(&self, index: u32, handle: u64, for_draw: bool) -> Option<View> {
        if index >= self.count {
            // Diagnostic (NVIDIA heap cap): the highest title slot past the
            // host heap seen so far.
            static HIGHEST: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if HIGHEST.fetch_max(index, std::sync::atomic::Ordering::Relaxed) < index {
                tracing::warn!(
                    index,
                    count = self.count,
                    for_draw,
                    "title slot past the host heap"
                );
            }
            return None;
        }
        // SAFETY: the handle is a descriptor slot in title memory.
        let [tag, i, generation] = unsafe { (handle as *const [u32; 3]).read_unaligned() };
        // Generations start at 1, so a made slot's key is never 0.
        let key = u64::from(generation) << 32 | u64::from(i);
        let mut made = self.made.lock().unwrap_or_else(|e| e.into_inner());
        let view = (tag == VIEW_TAG).then(|| view_at(i, generation)).flatten();
        let Some(view) = view else {
            // Not a view of ours (never written, or of a resource the
            // title released): nothing to read.
            if made[index as usize] != 0 {
                made[index as usize] = 0;
                self.make_null(index);
            }
            return None;
        };
        if made[index as usize] != key {
            let later = !for_draw && view.is_buffer();
            made[index as usize] = if later { 0 } else { key };
            let kept = if later {
                None
            } else {
                make(&view, self.cpu(index))
            };
            if kept.is_none() && !self.sampler {
                null_for(&view, self.cpu(index));
            }
            self.keep(index, kept.flatten());
        }
        Some(view)
    }

    /// The view host slot `index` was made from, in a few words, with the
    /// title and host addresses of a buffer view (diagnostics). "null" for
    /// a slot never made or left null.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn describe_slot(&self, index: u32) -> Option<String> {
        let key = *self.made.lock().unwrap_or_else(|e| e.into_inner()).get(index as usize)?;
        if key == 0 {
            return Some("null".into());
        }
        let view = view_at(key as u32, (key >> 32) as u32)?;
        let mut text: String = format!("{view:?}").chars().take(120).collect();
        if let Some((address, len)) = buffer_range(&view) {
            let host = crate::shadow::find(address, len.max(1)).map(|s| s.gpu(address));
            text.push_str(&format!(" title {address:#x}+{len:#x} host {host:#x?}"));
        }
        Some(text)
    }
}

/// Logs `what` the first time it happens.
macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

/// A view description from its words, with the format as the host has it.
fn srv_desc(words: &[u32; 10]) -> D3D12_SHADER_RESOURCE_VIEW_DESC {
    let mut words = *words;
    words[0] = crate::resource::pc_format(words[0]);
    // SAFETY: 40 bytes in the host's layout.
    unsafe {
        std::ptr::from_ref(&words)
            .cast::<D3D12_SHADER_RESOURCE_VIEW_DESC>()
            .read_unaligned()
    }
}

fn uav_desc(words: &[u32; 10]) -> D3D12_UNORDERED_ACCESS_VIEW_DESC {
    let mut words = *words;
    words[0] = crate::resource::pc_target_format(words[0]);
    // SAFETY: 40 bytes in the host's layout.
    unsafe {
        std::ptr::from_ref(&words)
            .cast::<D3D12_UNORDERED_ACCESS_VIEW_DESC>()
            .read_unaligned()
    }
}

/// A buffer view of title memory at `address` on its host copy: the first
/// element moves by the copy's offset.
/// Bytes per element of a buffer view.
fn element_size(desc: &D3D12_SHADER_RESOURCE_VIEW_DESC) -> u64 {
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    if buffer.StructureByteStride > 0 {
        u64::from(buffer.StructureByteStride)
    } else if buffer.Flags.0 & D3D12_BUFFER_SRV_FLAG_RAW.0 != 0 {
        4
    } else {
        crate::resource::element(desc.Format.0 as u32).map_or(0, |(bytes, _)| bytes)
    }
}

/// The title memory a buffer view (constant, shader, or unordered access)
/// reads: its address and length.
pub(crate) fn buffer_range(view: &View) -> Option<(u64, u64)> {
    let (base, words): (u64, &[u32; 10]) = match view {
        &View::Constant(address, size) => return Some((address, u64::from(size))),
        View::Shader(r, Some(desc)) => (
            crate::resource::buffer_bytes(*r as crate::object::This)?.0 as u64,
            desc,
        ),
        View::Placed(false, resource, desc) if resource[0] == 1 => (
            u64::from(desc[10]) | u64::from(desc[11]) << 32,
            desc[..10].try_into().ok()?,
        ),
        View::Unordered(r, _, Some(desc)) => {
            let base = crate::resource::buffer_bytes(*r as crate::object::This)?.0 as u64;
            return unordered_range(base, desc);
        }
        View::Placed(true, resource, desc) if resource[0] == 1 => {
            let base = u64::from(desc[10]) | u64::from(desc[11]) << 32;
            return unordered_range(base, desc[..10].try_into().ok()?);
        }
        _ => return None,
    };
    let desc = srv_desc(words);
    let element = element_size(&desc);
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    Some((
        base + buffer.FirstElement * element,
        u64::from(buffer.NumElements) * element,
    ))
}

/// [`buffer_range`] of an unordered access view of memory at `base`.
fn unordered_range(base: u64, words: &[u32; 10]) -> Option<(u64, u64)> {
    let desc = uav_desc(words);
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    let element = if buffer.StructureByteStride > 0 {
        u64::from(buffer.StructureByteStride)
    } else if buffer.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0 {
        4
    } else {
        crate::resource::element(desc.Format.0 as u32)?.0
    };
    Some((
        base + buffer.FirstElement * element,
        u64::from(buffer.NumElements) * element,
    ))
}

/// The host resource a slot made by [`make`] views, if any.
type Kept = Option<ID3D12Resource>;

/// `D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING` when any 3-bit source is past
/// `FORCE_1` (5) or the mapping is zero (unset from title memory).
fn valid_component_mapping(m: u32) -> u32 {
    if m == 0 {
        return D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    }
    for i in 0..4 {
        if ((m >> (3 * i)) & 0x7) > 5 {
            return D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        }
    }
    m
}

/// Whether the host device supports a typed UAV of `format` (cached).
fn uav_format_supported(device: &ID3D12Device, format: DXGI_FORMAT) -> bool {
    static CACHE: Mutex<Option<HashMap<i32, bool>>> = Mutex::new(None);
    let mut guard = CACHE.lock().unwrap_or_else(|e| e.into_inner());
    let map = guard.get_or_insert_with(HashMap::new);
    if let Some(&ok) = map.get(&format.0) {
        return ok;
    }
    let mut support = D3D12_FEATURE_DATA_FORMAT_SUPPORT {
        Format: format,
        ..Default::default()
    };
    // SAFETY: CheckFeatureSupport writes `support` of the given size.
    let ok = unsafe {
        device.CheckFeatureSupport(
            D3D12_FEATURE_FORMAT_SUPPORT,
            (&raw mut support).cast(),
            size_of_val(&support) as u32,
        )
    }
    .is_ok()
        && (support.Support1.0 & D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW.0) != 0;
    map.insert(format.0, ok);
    ok
}

/// Clamps mip/array fields shared by 1D/2D/cube SRV descriptions.
fn clamp_srv_mips(most: &mut u32, levels: &mut u32, mip_count: u16) -> bool {
    let mips = u32::from(mip_count.max(1));
    if *most >= mips {
        return false;
    }
    if *levels != u32::MAX {
        *levels = (*levels).min(mips - *most);
        if *levels == 0 {
            return false;
        }
    }
    true
}

fn clamp_array(first: &mut u32, size: &mut u32, count: u16) -> bool {
    let n = u32::from(count.max(1));
    if *first >= n {
        return false;
    }
    if *size != u32::MAX {
        *size = (*size).min(n - *first);
        if *size == 0 {
            return false;
        }
    } else {
        *size = n - *first;
    }
    true
}

fn plane_ok(plane: u32, format: DXGI_FORMAT) -> bool {
    // R32G8X24_TYPELESS (19) and R24G8_TYPELESS (44) expose a second plane.
    plane == 0 || (plane == 1 && matches!(format.0, 19 | 44))
}

/// Rejects / clamps a texture SRV description against the host resource.
fn sanitize_texture_srv(
    desc: &mut D3D12_SHADER_RESOURCE_VIEW_DESC,
    d: &D3D12_RESOURCE_DESC,
) -> bool {
    desc.Shader4ComponentMapping = valid_component_mapping(desc.Shader4ComponentMapping);
    let msaa = d.SampleDesc.Count > 1;
    let dim = d.Dimension;
    match desc.ViewDimension {
        D3D12_SRV_DIMENSION_TEXTURE1D | D3D12_SRV_DIMENSION_TEXTURE1DARRAY => {
            if dim != D3D12_RESOURCE_DIMENSION_TEXTURE1D {
                return false;
            }
        }
        D3D12_SRV_DIMENSION_TEXTURE2D
        | D3D12_SRV_DIMENSION_TEXTURE2DARRAY
        | D3D12_SRV_DIMENSION_TEXTURECUBE
        | D3D12_SRV_DIMENSION_TEXTURECUBEARRAY => {
            if dim != D3D12_RESOURCE_DIMENSION_TEXTURE2D {
                return false;
            }
            if matches!(
                desc.ViewDimension,
                D3D12_SRV_DIMENSION_TEXTURECUBE | D3D12_SRV_DIMENSION_TEXTURECUBEARRAY
            ) && d.DepthOrArraySize < 6
            {
                return false;
            }
        }
        D3D12_SRV_DIMENSION_TEXTURE2DMS | D3D12_SRV_DIMENSION_TEXTURE2DMSARRAY => {
            if dim != D3D12_RESOURCE_DIMENSION_TEXTURE2D || !msaa {
                return false;
            }
        }
        D3D12_SRV_DIMENSION_TEXTURE3D => {
            if dim != D3D12_RESOURCE_DIMENSION_TEXTURE3D {
                return false;
            }
        }
        _ => return false,
    }
    if msaa
        && !matches!(
            desc.ViewDimension,
            D3D12_SRV_DIMENSION_TEXTURE2DMS | D3D12_SRV_DIMENSION_TEXTURE2DMSARRAY
        )
    {
        return false;
    }
    match desc.ViewDimension {
        D3D12_SRV_DIMENSION_TEXTURE1D => {
            // SAFETY: ViewDimension selects Texture1D.
            let t = unsafe { &mut desc.Anonymous.Texture1D };
            clamp_srv_mips(&mut t.MostDetailedMip, &mut t.MipLevels, d.MipLevels)
        }
        D3D12_SRV_DIMENSION_TEXTURE1DARRAY => {
            // SAFETY: ViewDimension selects Texture1DArray.
            let t = unsafe { &mut desc.Anonymous.Texture1DArray };
            clamp_srv_mips(&mut t.MostDetailedMip, &mut t.MipLevels, d.MipLevels)
                && clamp_array(&mut t.FirstArraySlice, &mut t.ArraySize, d.DepthOrArraySize)
        }
        D3D12_SRV_DIMENSION_TEXTURE2D => {
            // SAFETY: ViewDimension selects Texture2D.
            let t = unsafe { &mut desc.Anonymous.Texture2D };
            clamp_srv_mips(&mut t.MostDetailedMip, &mut t.MipLevels, d.MipLevels)
                && plane_ok(t.PlaneSlice, d.Format)
        }
        D3D12_SRV_DIMENSION_TEXTURE2DARRAY => {
            // SAFETY: ViewDimension selects Texture2DArray.
            let t = unsafe { &mut desc.Anonymous.Texture2DArray };
            clamp_srv_mips(&mut t.MostDetailedMip, &mut t.MipLevels, d.MipLevels)
                && clamp_array(&mut t.FirstArraySlice, &mut t.ArraySize, d.DepthOrArraySize)
                && plane_ok(t.PlaneSlice, d.Format)
        }
        D3D12_SRV_DIMENSION_TEXTURE2DMS => true,
        D3D12_SRV_DIMENSION_TEXTURE2DMSARRAY => {
            // SAFETY: ViewDimension selects Texture2DMSArray.
            let t = unsafe { &mut desc.Anonymous.Texture2DMSArray };
            clamp_array(&mut t.FirstArraySlice, &mut t.ArraySize, d.DepthOrArraySize)
        }
        D3D12_SRV_DIMENSION_TEXTURE3D => {
            // SAFETY: ViewDimension selects Texture3D.
            let t = unsafe { &mut desc.Anonymous.Texture3D };
            clamp_srv_mips(&mut t.MostDetailedMip, &mut t.MipLevels, d.MipLevels)
        }
        D3D12_SRV_DIMENSION_TEXTURECUBE => {
            // SAFETY: ViewDimension selects TextureCube.
            let t = unsafe { &mut desc.Anonymous.TextureCube };
            clamp_srv_mips(&mut t.MostDetailedMip, &mut t.MipLevels, d.MipLevels)
        }
        D3D12_SRV_DIMENSION_TEXTURECUBEARRAY => {
            // SAFETY: ViewDimension selects TextureCubeArray.
            let t = unsafe { &mut desc.Anonymous.TextureCubeArray };
            if !clamp_srv_mips(&mut t.MostDetailedMip, &mut t.MipLevels, d.MipLevels) {
                return false;
            }
            let faces = u32::from(d.DepthOrArraySize);
            if t.First2DArrayFace >= faces {
                return false;
            }
            let max_cubes = (faces - t.First2DArrayFace) / 6;
            if max_cubes == 0 {
                return false;
            }
            if t.NumCubes == 0 || t.NumCubes == u32::MAX {
                t.NumCubes = max_cubes;
            } else {
                t.NumCubes = t.NumCubes.min(max_cubes);
            }
            t.NumCubes > 0
        }
        _ => false,
    }
}

/// Rejects / clamps a texture UAV description against the host resource.
fn sanitize_texture_uav(
    desc: &mut D3D12_UNORDERED_ACCESS_VIEW_DESC,
    d: &D3D12_RESOURCE_DESC,
) -> bool {
    let dim = d.Dimension;
    match desc.ViewDimension {
        D3D12_UAV_DIMENSION_TEXTURE1D | D3D12_UAV_DIMENSION_TEXTURE1DARRAY => {
            if dim != D3D12_RESOURCE_DIMENSION_TEXTURE1D {
                return false;
            }
        }
        D3D12_UAV_DIMENSION_TEXTURE2D | D3D12_UAV_DIMENSION_TEXTURE2DARRAY => {
            if dim != D3D12_RESOURCE_DIMENSION_TEXTURE2D {
                return false;
            }
        }
        D3D12_UAV_DIMENSION_TEXTURE3D => {
            if dim != D3D12_RESOURCE_DIMENSION_TEXTURE3D {
                return false;
            }
        }
        _ => return false,
    }
    let mips = u32::from(d.MipLevels.max(1));
    match desc.ViewDimension {
        D3D12_UAV_DIMENSION_TEXTURE1D => {
            // SAFETY: ViewDimension selects Texture1D.
            let t = unsafe { &mut desc.Anonymous.Texture1D };
            t.MipSlice < mips
        }
        D3D12_UAV_DIMENSION_TEXTURE1DARRAY => {
            // SAFETY: ViewDimension selects Texture1DArray.
            let t = unsafe { &mut desc.Anonymous.Texture1DArray };
            t.MipSlice < mips
                && clamp_array(&mut t.FirstArraySlice, &mut t.ArraySize, d.DepthOrArraySize)
        }
        D3D12_UAV_DIMENSION_TEXTURE2D => {
            // SAFETY: ViewDimension selects Texture2D.
            let t = unsafe { &mut desc.Anonymous.Texture2D };
            t.MipSlice < mips && t.PlaneSlice == 0
        }
        D3D12_UAV_DIMENSION_TEXTURE2DARRAY => {
            // SAFETY: ViewDimension selects Texture2DArray.
            let t = unsafe { &mut desc.Anonymous.Texture2DArray };
            t.MipSlice < mips
                && t.PlaneSlice == 0
                && clamp_array(&mut t.FirstArraySlice, &mut t.ArraySize, d.DepthOrArraySize)
        }
        D3D12_UAV_DIMENSION_TEXTURE3D => {
            // SAFETY: ViewDimension selects Texture3D.
            let t = unsafe { &mut desc.Anonymous.Texture3D };
            if t.MipSlice >= mips {
                return false;
            }
            let depth_at_mip = (u32::from(d.DepthOrArraySize) >> t.MipSlice.min(31)).max(1);
            if t.FirstWSlice >= depth_at_mip {
                return false;
            }
            if t.WSize == u32::MAX {
                t.WSize = depth_at_mip - t.FirstWSlice;
            } else {
                t.WSize = t.WSize.min(depth_at_mip - t.FirstWSlice);
            }
            t.WSize > 0
        }
        _ => false,
    }
}

fn buffer_srv(
    address: u64,
    len: u64,
    mut desc: D3D12_SHADER_RESOURCE_VIEW_DESC,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
) -> Option<Kept> {
    if desc.ViewDimension != D3D12_SRV_DIMENSION_BUFFER {
        once!(
            dimension = desc.ViewDimension.0,
            "non-buffer SRV dimension on buffer"
        );
        return None;
    }
    desc.Shader4ComponentMapping = valid_component_mapping(desc.Shader4ComponentMapping);
    {
        // SAFETY: ViewDimension is BUFFER.
        let buffer = unsafe { &mut desc.Anonymous.Buffer };
        if buffer.StructureByteStride > 0 {
            if buffer.StructureByteStride > 2048 {
                once!(
                    stride = buffer.StructureByteStride,
                    "structured buffer SRV stride past 2048"
                );
                return None;
            }
            buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
            desc.Format = DXGI_FORMAT_UNKNOWN;
        } else if buffer.Flags.0 & D3D12_BUFFER_SRV_FLAG_RAW.0 != 0 {
            desc.Format = DXGI_FORMAT_R32_TYPELESS;
        } else if desc.Format == DXGI_FORMAT_UNKNOWN {
            return None;
        }
    }
    let shadow = crate::shadow::find(address, len)?;
    let element = element_size(&desc);
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { &mut desc.Anonymous.Buffer };
    let offset = shadow.offset(address);
    if element == 0 || offset % element != 0 {
        once!(
            offset,
            element,
            "buffer view not on an element of its host copy"
        );
        return None;
    }
    buffer.FirstElement += offset / element;
    // SAFETY: a live host buffer.
    let width = unsafe { shadow.buffer.GetDesc() }.Width;
    let cap = width / element;
    if buffer.FirstElement >= cap {
        once!(
            first = buffer.FirstElement,
            cap,
            "buffer SRV FirstElement past host width"
        );
        return None;
    }
    buffer.NumElements = u64::from(buffer.NumElements)
        .min(cap - buffer.FirstElement)
        .min(u64::from(u32::MAX)) as u32;
    if buffer.NumElements == 0 {
        return None;
    }
    let host = host()?;
    // SAFETY: a live buffer and a slot of our heap.
    unsafe {
        host.device
            .CreateShaderResourceView(&shadow.buffer, Some(&desc), at);
    }
    #[cfg(feature = "diagnostics")]
    {
        // SAFETY: ViewDimension is BUFFER.
        let b = unsafe { desc.Anonymous.Buffer };
        crate::host::check_device_alive(format_args!(
            "CreateShaderResourceView buffer fmt={} dim={} first={} num={} stride={} flags={:#x}",
            desc.Format.0,
            desc.ViewDimension.0,
            b.FirstElement,
            b.NumElements,
            b.StructureByteStride,
            b.Flags.0
        ));
    }
    Some(Some(shadow.buffer.clone()))
}

/// An unordered access view of title buffer memory at `address` (`len`
/// bytes) on its host copy, as [`buffer_srv`]; false if the copy cannot be
/// written by shaders (upload memory).
/// shortcut: what shaders write stays in the host copy; title memory (and
/// so the CPU, and copies run on the CPU) does not see it.
fn buffer_uav(
    address: u64,
    len: u64,
    mut desc: D3D12_UNORDERED_ACCESS_VIEW_DESC,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
) -> Option<Kept> {
    if desc.ViewDimension != D3D12_UAV_DIMENSION_BUFFER {
        once!(
            dimension = desc.ViewDimension.0,
            "non-buffer UAV dimension on buffer"
        );
        return None;
    }
    {
        // SAFETY: ViewDimension is BUFFER.
        let buffer = unsafe { &mut desc.Anonymous.Buffer };
        if buffer.StructureByteStride > 0 {
            if buffer.StructureByteStride > 2048 {
                once!(
                    stride = buffer.StructureByteStride,
                    "structured buffer UAV stride past 2048"
                );
                return None;
            }
            buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;
            desc.Format = DXGI_FORMAT_UNKNOWN;
        } else if buffer.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0 {
            desc.Format = DXGI_FORMAT_R32_TYPELESS;
        } else if desc.Format == DXGI_FORMAT_UNKNOWN {
            return None;
        }
    }
    let shadow = crate::shadow::find(address, len)?;
    // SAFETY: a live buffer.
    let host_desc = unsafe { shadow.buffer.GetDesc() };
    if host_desc.Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 == 0 {
        once!("unordered access to a buffer whose host copy shaders cannot write");
        return None;
    }
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { &mut desc.Anonymous.Buffer };
    let element = if buffer.StructureByteStride > 0 {
        u64::from(buffer.StructureByteStride)
    } else if buffer.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0 {
        4
    } else {
        crate::resource::element(desc.Format.0 as u32).map_or(0, |(bytes, _)| bytes)
    };
    let offset = shadow.offset(address);
    if element == 0 || offset % element != 0 {
        once!(offset, element, "unordered buffer view not on an element of its host copy");
        return None;
    }
    buffer.FirstElement += offset / element;
    buffer.CounterOffsetInBytes = 0;
    let cap = host_desc.Width / element;
    if buffer.FirstElement >= cap {
        once!(
            first = buffer.FirstElement,
            cap,
            "buffer UAV FirstElement past host width"
        );
        return None;
    }
    buffer.NumElements = u64::from(buffer.NumElements)
        .min(cap - buffer.FirstElement)
        .min(u64::from(u32::MAX)) as u32;
    if buffer.NumElements == 0 {
        return None;
    }
    let host = host()?;
    // SAFETY: a live buffer and a slot of our heap.
    unsafe {
        host.device
            .CreateUnorderedAccessView(&shadow.buffer, None::<&ID3D12Resource>, Some(&desc), at);
    }
    #[cfg(feature = "diagnostics")]
    {
        // SAFETY: ViewDimension is BUFFER.
        let b = unsafe { desc.Anonymous.Buffer };
        crate::host::check_device_alive(format_args!(
            "CreateUnorderedAccessView buffer fmt={} dim={} first={} num={} stride={} flags={:#x}",
            desc.Format.0,
            desc.ViewDimension.0,
            b.FirstElement,
            b.NumElements,
            b.StructureByteStride,
            b.Flags.0
        ));
    }
    Some(Some(shadow.buffer.clone()))
}

/// Whether the host can make a shader (or unordered access) view of
/// `texture` in `format` (none: the texture's own). The console allows any
/// view; the host refuses (and removes the device for) views without the
/// resource's permission, of typed depth, and in another format than a
/// typed resource's. A view of a typed texture in another format of its
/// family takes the texture's format.
/// shortcut: such a view (sRGB of UNORM, say) reads without its
/// conversion; cast with castable formats if colors come out wrong.
fn fits(texture: &ID3D12Resource, mut format: Option<&mut DXGI_FORMAT>, unordered: bool) -> bool {
    use crate::resource::{DEPTH_FORMATS, family};
    // SAFETY: a live resource.
    let d = unsafe { texture.GetDesc() };
    let allowed = if unordered {
        d.Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 != 0
    } else {
        d.Flags.0 & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE.0 == 0
    };
    let own = d.Format.0 as u32;
    if !allowed || DEPTH_FORMATS.iter().any(|&(typed, _)| typed == own) {
        once!(
            unordered,
            format = own,
            flags = d.Flags.0,
            "view the host does not allow"
        );
        return false;
    }
    if unordered && d.SampleDesc.Count > 1 {
        once!(samples = d.SampleDesc.Count, "MSAA texture UAV refused");
        return false;
    }
    // Every family has more than one format, its typeless one first.
    let typeless = family(own) == own && family(own + 1) == own;
    if let Some(format) = format.as_deref_mut()
        && unordered
        && let Some(&(_, linear)) = crate::resource::SRGB
            .iter()
            .find(|&&(s, _)| s == format.0 as u32)
    {
        format.0 = linear as i32;
    }
    let view = format.as_ref().map_or(0, |f| f.0 as u32);
    if view == 0 {
        if typeless {
            return false;
        }
    } else {
        let promoted = crate::resource::pc_target_format(view) == own;
        if family(view) != family(own) && !promoted {
            once!(view, own, "view format of another family");
            return false;
        }
        if let Some(format) = format.as_deref_mut()
            && (!typeless || promoted)
        {
            *format = d.Format;
        }
    }
    if unordered {
        let Some(device) = host().map(|h| &h.device) else {
            return false;
        };
        let mut view_fmt = format.as_ref().map_or(d.Format, |f| **f);
        if view_fmt.0 == 0 {
            view_fmt = d.Format;
        }
        if !uav_format_supported(device, view_fmt) {
            once!(
                format = view_fmt.0,
                "host device does not support UAV for texture format"
            );
            return false;
        }
    }
    true
}

/// Makes the host descriptor for `view` at `at`: the host resource it
/// views, or `None` if it cannot.
fn make(view: &View, at: D3D12_CPU_DESCRIPTOR_HANDLE) -> Option<Kept> {
    let host = host()?;
    let device = &host.device;
    match view {
        &View::Constant(address, size) => {
            let shadow = crate::shadow::find(address, u64::from(size.max(1)))?;
            let gpu = shadow.gpu(address);
            let cbv_bytes = size.max(1).next_multiple_of(256).min(65536);
            if gpu == 0 || !gpu.is_multiple_of(256) || u64::from(cbv_bytes) > shadow.left(address)
            {
                once!(
                    address = format_args!("{address:#x}"),
                    gpu,
                    size,
                    "constant buffer not 256-byte aligned on the host"
                );
                return None;
            }
            let desc = D3D12_CONSTANT_BUFFER_VIEW_DESC {
                BufferLocation: gpu,
                SizeInBytes: cbv_bytes,
            };
            // SAFETY: a slot of our heap; the range lies in the host copy.
            unsafe { device.CreateConstantBufferView(Some(&desc), at) };
            #[cfg(feature = "diagnostics")]
            crate::host::check_device_alive(format_args!(
                "CreateConstantBufferView gpu={gpu:#x} size={cbv_bytes}"
            ));
            Some(Some(shadow.buffer.clone()))
        }
        View::Shader(resource, desc) => {
            let resource = *resource as crate::object::This;
            if let Some(texture) = crate::resource::host_resource(resource) {
                let mut desc = desc.as_ref().map(srv_desc);
                if !fits(texture, desc.as_mut().map(|d| &mut d.Format), false) {
                    return None;
                }
                // SAFETY: a live texture.
                let d = unsafe { texture.GetDesc() };
                if let Some(ref mut desc) = desc
                    && !sanitize_texture_srv(desc, &d)
                {
                    once!(
                        dim = desc.ViewDimension.0,
                        res = d.Dimension.0,
                        "texture SRV ViewDimension/mip/array refused"
                    );
                    return None;
                }
                // SAFETY: a live texture and a slot of our heap.
                unsafe {
                    device.CreateShaderResourceView(
                        texture,
                        desc.as_ref().map(std::ptr::from_ref),
                        at,
                    );
                }
                #[cfg(feature = "diagnostics")]
                crate::host::check_device_alive(format_args!(
                    "CreateShaderResourceView tex {}x{}x{} f{} flags={:#x} view_dim={} view_fmt={}",
                    d.Width,
                    d.Height,
                    d.DepthOrArraySize,
                    d.Format.0,
                    d.Flags.0,
                    desc.as_ref().map(|x| x.ViewDimension.0).unwrap_or(0),
                    desc.as_ref().map(|x| x.Format.0).unwrap_or(0),
                ));
                Some(Some(texture.clone()))
            } else if let (Some((data, len)), Some(desc)) =
                (crate::resource::buffer_bytes(resource), desc)
            {
                buffer_srv(data as u64, len, srv_desc(desc), at)
            } else {
                None
            }
        }
        View::Unordered(resource, _counter, desc) => {
            let resource = *resource as crate::object::This;
            let Some(texture) = crate::resource::host_resource(resource) else {
                return match (crate::resource::buffer_bytes(resource), desc) {
                    (Some((data, len)), Some(desc)) => {
                        buffer_uav(data as u64, len, uav_desc(desc), at)
                    }
                    _ => None,
                };
            };
            let mut desc = desc.as_ref().map(uav_desc);
            if !fits(texture, desc.as_mut().map(|d| &mut d.Format), true) {
                return None;
            }
            // SAFETY: a live texture.
            let d = unsafe { texture.GetDesc() };
            if let Some(ref mut desc) = desc
                && !sanitize_texture_uav(desc, &d)
            {
                once!(
                    dim = desc.ViewDimension.0,
                    res = d.Dimension.0,
                    "texture UAV ViewDimension/mip/array refused"
                );
                return None;
            }
            // SAFETY: a live texture and a slot of our heap.
            unsafe {
                device.CreateUnorderedAccessView(
                    texture,
                    None::<&ID3D12Resource>,
                    desc.as_ref().map(std::ptr::from_ref),
                    at,
                );
            }
            #[cfg(feature = "diagnostics")]
            crate::host::check_device_alive(format_args!(
                "CreateUnorderedAccessView tex {}x{}x{} f{} flags={:#x} view_dim={} view_fmt={}",
                d.Width,
                d.Height,
                d.DepthOrArraySize,
                d.Format.0,
                d.Flags.0,
                desc.as_ref().map(|x| x.ViewDimension.0).unwrap_or(0),
                desc.as_ref().map(|x| x.Format.0).unwrap_or(0),
            ));
            Some(Some(texture.clone()))
        }
        View::Sampler(words) => {
            // SAFETY: 13 words in the host's layout.
            let desc = unsafe {
                std::ptr::from_ref(words)
                    .cast::<D3D12_SAMPLER_DESC>()
                    .read_unaligned()
            };
            // SAFETY: a slot of our sampler heap.
            unsafe { device.CreateSampler(&desc, at) };
            #[cfg(feature = "diagnostics")]
            crate::host::check_device_alive(format_args!(
                "CreateSampler filter={} addrU={}",
                desc.Filter.0, desc.AddressU.0
            ));
            Some(None)
        }
        View::ConsoleSampler(words) => {
            let desc = crate::sampler::console(words);
            // SAFETY: a slot of our sampler heap.
            unsafe { device.CreateSampler(&desc, at) };
            #[cfg(feature = "diagnostics")]
            crate::host::check_device_alive(format_args!(
                "CreateSampler console filter={} addrU={}",
                desc.Filter.0, desc.AddressU.0
            ));
            Some(None)
        }
        View::Placed(unordered, resource, desc) => {
            if let Some(view) = placed_texture(view) {
                return make(&view, at);
            }
            const BUFFER: u32 = 1;
            if resource[0] != BUFFER {
                // shortcut: unordered access to a texture the title placed
                // itself is not mirrored; give it a resource as shader
                // views of such textures get when a dispatch writes one.
                once!(dimension = resource[0], "placed unordered texture view not mirrored");
                return None;
            }
            // The resource description's width, after the alignment.
            let len = u64::from(resource[4]) | u64::from(resource[5]) << 32;
            let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
            let words: &[u32; 10] = desc[..10].try_into().unwrap_or(&[0; 10]);
            if *unordered {
                buffer_uav(address, len, uav_desc(words), at)
            } else {
                buffer_srv(address, len, srv_desc(words), at)
            }
        }
    }
}

/// A shader view of a texture the title placed in memory itself, as a view
/// of the resource made for that memory (on first use).
pub(crate) fn placed_texture(view: &View) -> Option<View> {
    const BUFFER: u32 = 1;
    let View::Placed(false, resource, desc) = view else {
        return None;
    };
    if resource[0] == BUFFER {
        return None;
    }
    let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
    let texture = crate::resource::texture_at(address, resource)?;
    Some(View::Shader(texture as usize, desc[..10].try_into().ok()))
}

/// A null view of the kind `view` is, for a slot whose view the host
/// cannot make: unordered access (to a buffer or a 2D texture) or a shader
/// view. Writes to it are dropped, reads give zeros.
fn null_for(view: &View, at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let is_buffer = |r: &usize| crate::resource::buffer_bytes(*r as crate::object::This).is_some();
    // Unordered views: whether a buffer, and the title's view dimension.
    let (buffer, dimension) = match view {
        View::Unordered(r, _, desc) => (
            is_buffer(r) || desc.is_some_and(|d| d[1] == 1),
            desc.map(|d| d[1]),
        ),
        View::Placed(true, resource, desc) => (resource[0] == 1, Some(desc[1])),
        // Shader views of buffers read nothing through a null buffer view:
        // a null 2D view where the shader expects a buffer is a mismatch
        // the hardware does not survive in general (GPU-based validation
        // reports it).
        View::Shader(r, desc) if is_buffer(r) || desc.is_some_and(|d| d[1] == 1) => {
            return null_buffer_view(at);
        }
        View::Placed(false, resource, _) if resource[0] == 1 => return null_buffer_view(at),
        // Textures: a null view of the title's dimension (1D, arrays, 3D,
        // cubes), not always 2D.
        View::Shader(_, Some(desc)) => return null_like(desc, at),
        View::Placed(false, _, desc) => {
            return null_like(desc[..10].try_into().unwrap_or(&[0; 10]), at);
        }
        View::Constant(..) => {
            let desc = D3D12_CONSTANT_BUFFER_VIEW_DESC::default();
            // SAFETY: a slot of our heap; a zero location makes a null view.
            unsafe { host.device.CreateConstantBufferView(Some(&desc), at) };
            #[cfg(feature = "diagnostics")]
            crate::host::check_device_alive(format_args!("CreateConstantBufferView null"));
            return;
        }
        _ => return null_view(at),
    };
    let desc = if buffer {
        D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_UINT,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            ..Default::default()
        }
    } else {
        null_uav_desc(dimension)
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device.CreateUnorderedAccessView(
            None::<&ID3D12Resource>,
            None::<&ID3D12Resource>,
            Some(&desc),
            at,
        );
    }
    #[cfg(feature = "diagnostics")]
    crate::host::check_device_alive(format_args!(
        "CreateUnorderedAccessView null dim={} fmt={}",
        desc.ViewDimension.0, desc.Format.0
    ));
}

/// A null texture UAV of the title's view dimension (1D, 1D or 2D arrays,
/// 3D), 2D if it names none: a 2D null view where the shader writes a 3D
/// texture is a mismatch GPU-based validation reports (the world's wind
/// field, CS_CopyWindFieldBeforeUpdateWithSmooth, 2026-11-24).
fn null_uav_desc(dimension: Option<u32>) -> D3D12_UNORDERED_ACCESS_VIEW_DESC {
    let mut desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
        Format: DXGI_FORMAT_R32_UINT,
        ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
        ..Default::default()
    };
    match dimension.map(|d| D3D12_UAV_DIMENSION(d as i32)) {
        Some(D3D12_UAV_DIMENSION_TEXTURE1D) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1D;
        }
        Some(D3D12_UAV_DIMENSION_TEXTURE1DARRAY) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1DARRAY;
            desc.Anonymous.Texture1DArray = D3D12_TEX1D_ARRAY_UAV {
                ArraySize: 1,
                ..Default::default()
            };
        }
        Some(D3D12_UAV_DIMENSION_TEXTURE2DARRAY) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
            desc.Anonymous.Texture2DArray = D3D12_TEX2D_ARRAY_UAV {
                ArraySize: 1,
                ..Default::default()
            };
        }
        Some(D3D12_UAV_DIMENSION_TEXTURE3D) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D;
            desc.Anonymous.Texture3D = D3D12_TEX3D_UAV {
                WSize: u32::MAX,
                ..Default::default()
            };
        }
        _ => {}
    }
    desc
}

/// A buffer view that reads zeros (raw, as translated shaders mostly read
/// buffers).
fn null_buffer_view(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R32_TYPELESS,
        ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Buffer: D3D12_BUFFER_SRV {
                NumElements: 1,
                Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                ..Default::default()
            },
        },
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device
            .CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), at);
    }
    #[cfg(feature = "diagnostics")]
    crate::host::check_device_alive(format_args!("CreateShaderResourceView null buffer"));
}

/// A texture view that reads zeros with the dimension (and mip and array
/// counts) of the title's description `words`; 2D if it names none.
fn null_like(words: &[u32; 10], at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    const TEXTURES: std::ops::RangeInclusive<u32> = 2..=10;
    if !TEXTURES.contains(&words[1]) {
        return null_view(at);
    }
    let Some(host) = host() else { return };
    // One mip, one slice or cube, plane 0: fields a null view still checks.
    let mut w = [0u32; 10];
    w[1] = words[1];
    match words[1] {
        6 => {}
        7 => w[5] = 1,
        3 | 5 | 10 => (w[5], w[7]) = (1, 1),
        _ => w[5] = 1,
    }
    let mut desc = srv_desc(&w);
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device
            .CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), at);
    }
    #[cfg(feature = "diagnostics")]
    crate::host::check_device_alive(format_args!(
        "CreateShaderResourceView null_like dim={}",
        desc.ViewDimension.0
    ));
}

/// A view that reads zeros, for slots whose view the host cannot make.
/// shortcut: slots with no view at all (kind unknown) get this null 2D
/// view; null buffer views there did not stop the world's low-address
/// page faults (2026-11-24, run w10), so they are not the cause.
fn null_view(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Texture2D: D3D12_TEX2D_SRV {
                MipLevels: 1,
                ..Default::default()
            },
        },
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device
            .CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), at);
    }
    #[cfg(feature = "diagnostics")]
    crate::host::check_device_alive(format_args!("CreateShaderResourceView null 2D"));
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A null texture UAV keeps the title's dimension (3D for the world's
    /// wind field), and is 2D when the title names none.
    #[test]
    fn null_uavs_keep_the_titles_dimension() {
        let d = null_uav_desc(Some(8));
        assert_eq!(d.ViewDimension, D3D12_UAV_DIMENSION_TEXTURE3D);
        // SAFETY: the member the dimension names.
        assert_eq!(unsafe { d.Anonymous.Texture3D.WSize }, u32::MAX);
        let d = null_uav_desc(Some(5));
        assert_eq!(d.ViewDimension, D3D12_UAV_DIMENSION_TEXTURE2DARRAY);
        // SAFETY: the member the dimension names.
        assert_eq!(unsafe { d.Anonymous.Texture2DArray.ArraySize }, 1);
        let dimension = |d| null_uav_desc(d).ViewDimension;
        assert_eq!(dimension(Some(2)), D3D12_UAV_DIMENSION_TEXTURE1D);
        assert_eq!(dimension(Some(4)), D3D12_UAV_DIMENSION_TEXTURE2D);
        assert_eq!(dimension(None), D3D12_UAV_DIMENSION_TEXTURE2D);
        // Multisampled UAVs need a cap; they stay 2D.
        assert_eq!(dimension(Some(6)), D3D12_UAV_DIMENSION_TEXTURE2D);
    }

    /// A null 3D UAV made on the real device leaves it alive.
    #[test]
    fn null_3d_uav_is_accepted_by_the_device() {
        let Some(host) = host() else {
            return; // no host GPU
        };
        let Some(heap) = HostHeap::new(false, 4) else {
            return;
        };
        let view = View::Unordered(0, 0, Some([28, 8, 0, 0, 0, 0, 0, 0, 0, 0]));
        null_for(&view, heap.cpu(0));
        // SAFETY: a plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }

    /// Dynamic slots stay taken until given back, and runs do not wrap.
    #[test]
    fn dynamic_slots_are_held_until_released() {
        let Some(heap) = HostHeap::new(false, 4) else {
            return; // no host GPU
        };
        let a = heap.dynamic(heap.dynamic - 100).unwrap();
        let b = heap.dynamic(60).unwrap();
        assert_eq!((a, b), (4, 4 + heap.dynamic - 100));
        // 40 left at the end: 50 do not fit until something is given back.
        assert_eq!(heap.dynamic(50), None);
        heap.release(a, 50);
        assert_eq!(heap.dynamic(50), Some(4));
        assert_eq!(heap.dynamic(40), Some(4 + heap.dynamic - 40));
    }

    #[test]
    fn dynamic_pool_word_boundaries_fast_fail_and_double_release() {
        let mut pool = DynamicPool::new(130);
        assert_eq!(pool.alloc(60), Some(0));
        assert_eq!(pool.alloc(10), Some(60)); // crosses 64-bit word boundary
        assert_eq!(pool.alloc(60), Some(70)); // reaches exact end 130
        assert_eq!(pool.free, 0);
        assert_eq!(pool.alloc(1), None);
        // Release two disjoint holes (5 + 5 = 10 free, largest contiguous run = 5)
        // plus test double-release safety.
        pool.release(60, 5);
        pool.release(60, 5);
        pool.release(70, 5);
        assert_eq!(pool.free, 10);
        // 6 <= free, so scan discovers no contiguous run of 6 and caches min_failed = 6.
        assert_eq!(pool.alloc(6), None);
        assert_eq!(pool.min_failed, 6);
        assert_eq!(pool.alloc(5), Some(60));
        assert_eq!(pool.alloc(5), Some(70));
        assert_eq!(pool.free, 0);
    }

    #[test]
    fn shared_tables_are_refcounted_until_last_release() {
        let Some(heap) = HostHeap::new(false, 4) else {
            return; // no host GPU
        };
        let key = vec![[VIEW_TAG, 1, 1], [VIEW_TAG, 2, 1]];
        assert_eq!(heap.acquire_shared(&key), None);
        let first = heap.dynamic(2).unwrap();
        assert_eq!(heap.publish_shared(key.clone(), first, 2), first);
        // Second list shares the exact same slot block without consuming new slots.
        assert_eq!(heap.acquire_shared(&key), Some(first));
        // Competing publish returns existing slot and frees the duplicate allocation.
        let dup = heap.dynamic(2).unwrap();
        assert_ne!(dup, first);
        assert_eq!(heap.publish_shared(key.clone(), dup, 2), first);
        // Three refs held; two releases keep it alive, third releases it.
        heap.release(first, 2);
        heap.release(first, 2);
        assert_eq!(heap.acquire_shared(&key), Some(first));
        heap.release(first, 2);
        heap.release(first, 2);
        assert_eq!(heap.acquire_shared(&key), None);
    }

    fn alive(host: &crate::host::Host) {
        assert!(
            // SAFETY: a plain query.
            unsafe { host.device.GetDeviceRemovedReason() }.is_ok(),
            "host device was removed"
        );
    }

    fn tex2d(
        host: &crate::host::Host,
        format: DXGI_FORMAT,
        flags: D3D12_RESOURCE_FLAGS,
    ) -> Option<ID3D12Resource> {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 4,
            Height: 4,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: format,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Flags: flags,
            ..Default::default()
        };
        let mut texture: Option<ID3D12Resource> = None;
        // SAFETY: plain committed texture for the test.
        unsafe {
            host.device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_COMMON,
                    None,
                    &mut texture,
                )
                .ok()?;
        }
        texture
    }

    /// Illegal / out-of-range views are refused or sanitized without removing the device.
    #[test]
    fn invalid_views_do_not_remove_the_device() {
        let Some(host) = host() else {
            return;
        };
        let Some(heap) = HostHeap::new(false, 8) else {
            return;
        };
        alive(host);

        // RAYTRACING_ACCELERATION_STRUCTURE (11) on a buffer → refused.
        let rtas = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_UNKNOWN,
            ViewDimension: D3D12_SRV_DIMENSION(11),
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            ..Default::default()
        };
        assert!(buffer_srv(0x2000_0000, 64, rtas, heap.cpu(0)).is_none());
        alive(host);

        // R9G9B9E5_SHAREDEXP texture UAV → refused by format support.
        let f67 = DXGI_FORMAT(67); // R9G9B9E5_SHAREDEXP
        if let Some(tex) = tex2d(host, f67, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) {
            let mut fmt = f67;
            assert!(!fits(&tex, Some(&mut fmt), true));
            alive(host);
        }

        // Out-of-bounds buffer SRV / UAV.
        crate::shadow::register(0xDEAD, 0x3000_0000, 64);
        let oob_srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_SRV {
                    FirstElement: 1_000_000,
                    NumElements: 4,
                    Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                    ..Default::default()
                },
            },
        };
        assert!(buffer_srv(0x3000_0000, 64, oob_srv, heap.cpu(1)).is_none());
        alive(host);
        let oob_uav = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_UAV {
                    FirstElement: 1_000_000,
                    NumElements: 4,
                    Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                    ..Default::default()
                },
            },
        };
        assert!(buffer_uav(0x3000_0000, 64, oob_uav, heap.cpu(2)).is_none());
        alive(host);

        // Structured buffer with a non-UNKNOWN format is sanitized (not a removal).
        let structured = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R32_UINT,
            ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_SRV {
                    FirstElement: 0,
                    NumElements: 4,
                    StructureByteStride: 16,
                    Flags: D3D12_BUFFER_SRV_FLAG_NONE,
                },
            },
        };
        assert!(buffer_srv(0x3000_0000, 64, structured, heap.cpu(3)).is_some());
        alive(host);
        let structured_uav = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_UINT,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_UAV {
                    FirstElement: 0,
                    NumElements: 4,
                    StructureByteStride: 16,
                    Flags: D3D12_BUFFER_UAV_FLAG_NONE,
                    ..Default::default()
                },
            },
        };
        assert!(buffer_uav(0x3000_0000, 64, structured_uav, heap.cpu(4)).is_some());
        alive(host);

        // TEXTURE3D SRV / UAV on a TEXTURE2D resource → refused.
        let Some(tex2) = tex2d(
            host,
            DXGI_FORMAT_R8G8B8A8_UNORM,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        ) else {
            return;
        };
        // SAFETY: live test texture.
        let d = unsafe { tex2.GetDesc() };
        let mut bad_srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE3D,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture3D: D3D12_TEX3D_SRV {
                    MipLevels: 1,
                    ..Default::default()
                },
            },
        };
        assert!(!sanitize_texture_srv(&mut bad_srv, &d));
        let mut bad_uav = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            ViewDimension: D3D12_UAV_DIMENSION_TEXTURE3D,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Texture3D: D3D12_TEX3D_UAV {
                    WSize: 1,
                    ..Default::default()
                },
            },
        };
        assert!(!sanitize_texture_uav(&mut bad_uav, &d));
        alive(host);

        // CBV size 0 and past 65536: clamped / null, device stays up.
        let (buf, _) = host.shared_buffer(256).expect("shared");
        // SAFETY: live shared buffer.
        let gpu = unsafe { buf.GetGPUVirtualAddress() };
        heap.constant_at(5, gpu, 0, Some(buf.clone()));
        heap.constant_at(6, gpu, 70_000, Some(buf));
        heap.constant_at(7, 0, 256, None); // null CBV for bad address
        alive(host);

        crate::shadow::unregister(0xDEAD, 0x3000_0000);
    }
}
