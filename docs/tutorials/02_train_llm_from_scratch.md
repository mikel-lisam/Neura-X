# 02 — Train an LLM From Scratch on 8 GB RAM

## 1. Define the model

```python
import neura_x as nx

model = nx.LLM(
    conceptual_params=1_000_000_000,   # 1B conceptual parameters
    architecture="transformer",
)
```
Only Fractal Seeds (~MB) occupy RAM; the billion weights exist as formulas.

---

## 2. Stream the data

```python
loader = nx.StreamLoader(
    path="corpus/",
    chunk_size="64MB",
    tokenizer="bpe",
    skip_learned=True,       # Curriculum Sampler
    safety_filter=True,
)
```

---

## 3. Train

```python
engine = nx.PreTrainingEngine(model)
engine.train_from_scratch(
    dataset=loader,
    optimizer="adamw",
    lr=1e-4,
    scheduler="cosine",
    target_loss=2.1,
)
```

Internally: Wake logs Surprise Vectors; Sleep consolidates them; the Shadow Optimizer updates rank-64 subspaces; the Liquid Router keeps 95% asleep.

---

## 4. Watch it dream

```python
nx.DreamDashboard(model).open()
```

---

## 5. Export

```python
model.export("my-first-llm.nex")   # a few megabytes, Founder's Lock sealed
```

---

## 6. Verify memory

```python
nx.xray(model)   # peak RSS should stay far below your RAM budget
```

You just pretrained a billion-parameter model on a laptop. Welcome to intelligence without limits.

---