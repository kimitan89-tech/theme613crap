# theme613crap

fixing something (testing a future variation game).

Based on [VIctory](https://github.com/Marvisak/VIctory) (licensed under GNU GPL v3.0 — see [LICENSE](LICENSE)).

## Local Changes (on top of `ffe310e`)

- **NVIDIA dynamic descriptor handling (`crates/victory-gpu`)**: replaced the linear `Vec<bool>` dynamic slot scan with an O(1) fast-fail 64-bit bitset allocator, backed 256-byte-aligned CBVs directly by persistent `Shadow` buffers, and added reference-counted cross-list sharing for persistent dynamic tables.
- **CPU submission preparation (`crates/victory-gpu`)**: switched `Batch` buffer refreshes and `Shadow` UAV snapshots from whole-heap bounding boxes to sorted non-overlapping intervals, scoped `Step::GpuWrite` flushes to overlapping non-pending ranges, and deduplicated 4 KiB write-watch pages per 64 KiB block.
- **Pipeline compile backlog (`crates/victory-gpu`)**: prevented parallel recording threads from stalling simultaneously on deep pipeline queues and removed `THREAD_PRIORITY_BELOW_NORMAL` starvation on pipeline workers.
- **Series S multi-plane `PresentX` display (`crates/victory-gpu`)**: added a D3D12 premultiplied-alpha overlay compositor so multi-plane presentation (1080p base plane + 4K UI overlay) displays properly.

## Test Machine Specs

- **OS:** Windows 11 Home Single Language (64-bit)
- **CPU:** 12th Gen Intel Core i5-12450HX
- **RAM:** 16 GB (15.7 GB usable)
- **GPU:** NVIDIA GeForce RTX 3050 6 GB Laptop GPU
