# Industry Benchmark Suite

Neura-X evaluates with `nx.EvalSuite()` against standard suites so results
are comparable with the published literature.

| Benchmark | Measures | Shots | Notes |
|-----------|----------|-------|-------|
| MMLU | Broad knowledge & reasoning | 5 | Multiple choice, 57 subjects |
| HumanEval | Code generation | 0 | pass@1, unit-test verified |
| GSM8K | Math word problems | 8 | Chain-of-thought prompted |
| HellaSwag | Commonsense inference | 0 | Sentence completion |
| TruthfulQA | Hallucination resistance | 0 | Truth + info scores |
| ARC-Challenge | Scientific reasoning | 25 | Multiple choice |
| WinoGrande | Coreference | 0 | Pronoun resolution |

## Usage

```python
import neura_x as nx
suite = nx.EvalSuite()
results = suite.run(model, benchmarks=["mmlu", "humaneval", "gsm8k"])
print(results.mmlu_score, results.memory_to_intelligence_ratio)
```
---

## Independent Grading

`nx.Judge` sends anonymized outputs to an external grader API for unbiased scoring; judge identity and prompts are published with each run.

---

## Policy
- No benchmark-specific tuning beyond published prompting conventions.
- Contamination checks: n-gram overlap scan against eval sets, reported.
- Full per-question outputs archived for audit.