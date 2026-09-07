# The Neura-X Challenge

A falsifiable public claim: **algorithmic efficiency defeats hardware wealth.**

## Protocol

1. **Model.** 1B-parameter Transformer, identical architecture both arms.
2. **Data.** Identical 20 GB text corpus, identical tokenizer (BPE 32k).
3. **Arm A.** PyTorch 2.x on RTX 4090 (24 GB VRAM), batch 8, fp16.
4. **Arm B.** Neura-X v1.0.0 on Genesis Machine (8 GB RAM, CPU only), batch 8.
5. **Steps.** 100,000 optimizer steps, cosine schedule, AdamW.
6. **Report.** Final loss; MMLU; HumanEval; GSM8K; wall-clock; USD cost.

## Scorecard Template

| Metric | PyTorch / RTX 4090 | Neura-X / E7450 | Delta |
|--------|--------------------|------------------|-------|
| Final validation loss | _ | _ | _ |
| MMLU (5-shot) | _ | _ | _ |
| HumanEval pass@1 | _ | _ | _ |
| GSM8K (8-shot) | _ | _ | _ |
| Wall-clock | _ | _ | _ |
| Hardware cost | _ | _ | _ |
| Energy (kWh) | _ | _ | _ |

## Acceptance Criteria

- Neura-X loss within 5% of Arm A.
- Neura-X benchmark scores within 95% of Arm A (the stated guarantee).
- Peak RSS ≤ 8 GB on Arm B (verified via `/proc` sampling).

## Status

v1.0.0: protocol frozen; runs in progress. Results published with raw logs.