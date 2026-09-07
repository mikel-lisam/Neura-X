# 03 — Fine-Tuning

## Load a base model

```python
import neura_x as nx
model = nx.load("base-llm.nex")
```

---

## Advanced loop (full control, standard math)

```python
opt = nx.optim.AdamW(model.parameters(), lr=2e-5)
sched = nx.optim.CosineAnnealingLR(opt, T_max=1000)
loss_fn = nx.loss.CrossEntropyLoss()

for batch in nx.StreamLoader("instructions.jsonl", batch_size=4):
    opt.zero_grad()
    out = model(batch.inputs)
    loss = loss_fn(out, batch.targets)
    loss.backward()
    opt.step()
    sched.step()
```

---

## Privacy-preserving fine-tune

```python
from neura_x.privacy import PrivacyEngine
pe = PrivacyEngine(epsilon=1.0, delta=1e-5)
# gradients are clipped and noised inside the Shadow Optimizer
```

---

## Save the adapter-style result

```python
model.export("my-finetune.nex")
```

