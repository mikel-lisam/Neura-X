# Neura-X Native Metrics

Industry metrics measure intelligence. Native metrics measure
**intelligence per unit of scarcity** — the thing Neura-X optimizes.

## MIR — Memory-to-Intelligence Ratio

    MIR = BenchmarkScore / PeakRAM_GB

Higher is better. A 70B model scoring 70.0 MMLU in 8 GB scores MIR 8.75;
the same score in 160 GB scores MIR 0.44. Neura-X targets ≥ 20× MIR uplift
over dense baselines at equal accuracy.

## TPI — Tokens Per Inference-second (CPU)

    TPI = tokens_generated / wall_seconds   (CPU only, no accelerator)

Measured on the Genesis Machine at 512-token context. Reports p50/p95.

## FR — Fractal Ratio

    FR = conceptual_parameters / nex_file_bytes

A 70B model in an 8 MB file: FR ≈ 8.75×10⁹ per byte. FR quantizes the
core claim: intelligence compressed to mathematical essence.

## SRR — Sparsity Realization Rate

    SRR = measured_asleep_fraction / configured_awake_threshold_complement

Verifies the Liquid Router delivers the sparsity it promises (target ≥ 0.98).

## CEE — Consolidation Energy Efficiency

    CEE = surprise_vectors_consolidated / joule_consumed

Sleep-phase efficiency; measured via RAPL where available.

## Reporting Format

Every release publishes: MIR, TPI, FR, SRR, CEE per model class, with
hardware manifest and seeds.