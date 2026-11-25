# VIctory / ztrx

Runs Xbox Series title builds natively on Windows PCs with Direct3D 12
(shader model 6.6, mesh shaders), from title files you supply yourself.

**This repository contains no title files or assets.** You need your own copy
of the title. Not affiliated with or endorsed by its developer or publisher.

## Status

Integrates the **ztrx** (`90b529f`) world-rendering and performance core on
top of VIctory, plus Series S multi-plane `PresentX` compositing and a
bilinear FFX upscaler for 1080p Series S presentation on modest GPUs
(e.g. RTX 3050 laptop).

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

## Series S / performance settings (optional, off by default)

| Switch | Effect |
|---|---|
| `VICTORY_RELAY=2` | relay presentation / reduce CPU wait |
| `VICTORY_LATE_WAIT=3` | defer GPU waits |
| `VICTORY_WIDE_BACK=1` | wider copy-back windows |
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

- **VIctory** — Marvisak & contributors (upstream base)
- **victory-fork** — fxceless
- **ztrx** — Zettrox & contributors (commit `90b529f`: world rendering, Wave64-on-Wave32, shader opts, relay/late-wait/wide-back, pipeline prewarm, …)
- **XCL** — ivall & contributors
