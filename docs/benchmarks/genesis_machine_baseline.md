# Genesis Machine Baseline

All Neura-X baseline numbers are produced on the Founder's personal laptop.
It is the proof that the framework needs no privilege to work.

## Hardware Manifest

| Component | Specification |
|-----------|---------------|
| Device | Dell Latitude E7450 (2015) |
| CPU | Intel Core i7-5600U (Broadwell, 5th Gen) |
| Cores / Threads | 2 / 4 |
| SIMD | AVX2 (256-bit), SSE4.2 |
| RAM | 8 GB DDR3L-1600 |
| Storage | SATA SSD |
| GPU | Intel HD 5500 (unused) |
| OS | Linux (kernel 6.x), GCC 14, Rust stable, LLVM 19, Python 3.12 |

## Why This Machine

- **Availability:** millions of identical office laptops exist worldwide.
- **AVX2:** the minimum ISA for Holographic Memory and Fractal blooming.
- **The Zero-Excuse Threshold:** if Neura-X trains here, nobody can claim
  their computer is not good enough for AI.

## Baseline Tables (v1.0.0)

| Workload | Peak RSS | Wall-clock | Notes |
|----------|----------|------------|-------|
| Fractal bloom 4096×4096, K=1024 | < 50 MB | _ms | cache-resident |
| ShadowAdamW step 4096×4096, r=64 | < 60 MB | _ms | 64× less opt. memory |
| LiquidRouter route N=10⁶, =5% | < 80 MB | _ms | 95% asleep |
| .nex round-trip 70B-class seed set | < 40 MB | _s | magic NEXX verified |
| 1B from-scratch pretraining | ≤ 8 GB | _h | Challenge run |

(Values filled per release; raw logs under `logs/`.)

## Thermal Behavior

Thermal-Aware Circadian Scheduling logs pause events; baseline tables
include pause count and mean pause duration so throttling is visible,
never hidden.