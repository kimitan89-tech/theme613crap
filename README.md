# VIctory / ztrx

Runs Xbox Series title builds natively on Windows PCs with Direct3D 12
(shader model 6.6, mesh shaders), from title files you supply yourself.

**This repository contains no title files or assets.** You need your own copy
of the title. Not affiliated with or endorsed by its developer or publisher.

## Status

Integrates the **ztrx** (`90b529f`) world-rendering and performance core and
**victory-amd** (`070df42`) streaming/view-guard/shader-warming fixes on top
of VIctory, plus Series S multi-plane `PresentX` compositing and a bilinear
FFX upscaler for 1080p Series S presentation on modest GPUs (e.g. RTX 3050
laptop).

## Requirements

- Windows 10/11 x64, a CPU with AVX2
- A Direct3D 12 GPU with shader model 6.6 and mesh shaders (NVIDIA RTX 20 series or newer)
- Rust (the version in `rust-toolchain.toml`)
- Several GB of free Windows commit memory (raise the paging file if `crashes.log` reports error 1455)

## Build

```powershell
cargo build --release -p victory
```

Diagnostics build:

```powershell
cargo build --profile diag -p victory --features diagnostics
```

## Play

```powershell
powershell -ExecutionPolicy Bypass -File scripts\play.ps1 -Game "<your title folder>\<the title's exe>"
```

`scripts\play.ps1` keeps the shader cache and saves in folders you choose
(`-Cache`, `-Data`) and presses no buttons for you. Controller: XInput pads
(a DualShock/DualSense through DS4Windows works). With no pad connected the
keyboard acts as one while the title window has focus.

## Series S / performance & streaming settings

| Switch | Effect |
|---|---|
| `victory warm <title.exe>` | pre-translate shaders into the disk cache before play |
| `VICTORY_RELAY=2` | relay presentation / reduce CPU wait (off by default) |
| `VICTORY_LATE_WAIT=3` | defer GPU waits (off by default) |
| `VICTORY_WIDE_BACK=1` | wider copy-back windows (off by default) |
| `VICTORY_VIEW_GUARD=0` | disable cross-view page write tracking (on by default) |
| `VICTORY_CARD_REHASH_RUNS=0` | disable joining card hash passes (on by default) |
| `VICTORY_JOIN_COPIES=0` | disable joining consecutive title buffer copies (on by default) |
| `VICTORY_SWEEP_PRESSURE=0` | disable faster idle reclaim when RAM load ≥ 85% (on by default) |
| `VICTORY_INPLACE=1` / `=2` | place windows (1) or title buffers too (2) over title pages; skips view-guard on those pages |
| `VICTORY_HEAP_RECHECK=0` | disable delayed re-checks of indexed textures (on by default) |
| `VICTORY_TEXTURE_SWEEP=0` | disable recovery sweep for lost placed-texture views (on by default) |
| `VICTORY_RT_RESOLVE=1` | compile ray-tracing resolve shaders when BVH builder is off |
| (Series S 1080p) | multi-plane `PresentX` compositing + bilinear FFX upscaler are built in |

Set env vars in PowerShell before running the script. Example:

```powershell
$env:VICTORY_RELAY = "2"
# or:
$env:VICTORY_LATE_WAIT = "3"; $env:VICTORY_WIDE_BACK = "1"
```

## Debugging switches

| Switch | Effect |
|---|---|
| `VICTORY_CAPTURE=<dir>` + `VICTORY_CAPTURE_FRAMES=<list>` | save those frames as PNG (diagnostics build) |
| `VICTORY_DUMP_ON_UPSCALE=<N>` | dump the next N frames after an upscale (diagnostics build) |
| `VICTORY_KEYS=30:Enter,35:X` | press keys at those seconds (diagnostics build) |
| `VICTORY_D3D12_DRED=1` | on a GPU failure, report the commands in flight |
| `VICTORY_PIPELINE_WAIT_MS=24` | shorter per-frame wait for pipelines still being built |

## License

GPL-3.0-or-later (see `LICENSE`).

Credits (in no particular order of importance beyond lineage):

- **VIctory** — Marvisak & contributors (upstream base, incl. `#1122` by tightfall)
- **victory-fork** — fxceless (incl. `#115` by Spexivus with correction by joroyt)
- **ztrx** — Zettrox & contributors (commit `90b529f`: world rendering, Wave64-on-Wave32, shader opts, relay/late-wait/wide-back, pipeline prewarm, …)
- **victory-amd** — myokex & contributors (commit `070df42`: partial-mip detiling, true texture spans, split >65,535 dispatches, DirectStorage page-array reads, persistent early pipelines, cross-view page guard, indexed texture re-check, `warm` CLI)
- **XCL** — ivall & contributors
