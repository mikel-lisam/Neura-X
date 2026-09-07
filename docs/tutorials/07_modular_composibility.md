# 07 — Modular Composability

Build custom intelligence from capability modules. No restrictions.

## Compose

```python
import neura_x as nx

model = nx.compose(
    brain=nx.LLM(conceptual_params=8_000_000_000),
    modules=[
        nx.module.Prediction(task="time_series"),
        nx.module.ImageGen(resolution=512),
        nx.module.AudioGen(sample_rate=32000),
        nx.module.Tools([nx.tools.web_search, nx.tools.run_code]),
        nx.module.Skills(["swahili", "medical", "coding_python"]),
    ],
)
```

---

## Train joinly or per-module

```python
nx.TrainingEngine(model).train_from_scratch(dataset="combined/")
# or
nx.TrainingEngine(model.brain).train_from_scratch(dataset="text/")
nx.TrainingEngine(model.get_module("imagegen")).train(dataset="images/")
model.fine_tune_together(dataset="multimodal/")
```

---

## Hot-swap runtime

```python
model.attach(nx.module.VideoGen(resolution=256, fps=8))
model.detach("audiogen")
```

---

## Write your own module

```python
@nx.module(name="kenyan_law", description="Kenyan legal expert")
class KenyanLaw(nx.Module):
    def __init__(self):
        super().__init__()
        self.knowledge = nx.HolographicMemory(capacity=50_000)
        self.fractal = nx.FractalLayer(4096, 4096, conceptual_params=500_000_000)
    def train(self, dataset): ...
    def forward(self, x): ...

nx.hub.push_module("kenyan_law", version="1.0")
```

The Liquid Router wakes only the modules each input needs: ten attached, one or two resident.