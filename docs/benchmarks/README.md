
# Neura-X Benchmarks

Philosophy: every number must be reproducible on the weakest supported
machine. If it runs here, it runs anywhere.

| Document | Contents |
|----------|----------|
| `neura_x_challenge.md` | The public head-to-head protocol vs PyTorch |
| `native_metrics.md` | MIR, TPI, FR — Neura-X-native metric definitions |
| `industry_benchmarks.md` | MMLU, HumanEval, GSM8K, HellaSwag, TruthfulQA, ARC, WinoGrande |
| `genesis_machine_baseline.md` | Reference hardware specs and baseline tables |

## Reproducibility Rules

1. Hardware: Genesis Machine unless stated (see baseline doc).
2. Software: Neura-X v1.0.0, Python 3.12, GCC 14, Rust stable, LLVM 19.
3. Seeds: fixed and published with every result.
4. Raw logs committed under `docs/benchmarks/logs/`.
5. No cloud, no GPU, no thermal cheating (throttling reported, not hidden).