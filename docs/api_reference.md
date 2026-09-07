# Neura-X API Reference — v1.0

`import neura_x as nx`

---

## 1. Package Metadata

| Attribute | Value |
|-----------|-------|
| `nx.__version__` | 1.0.0 |
| `nx.__author__` | Edusei Mikel Lisamba |
| `nx.__tagline__` | Intelligence Without Limits. |
| `nx._core_available` | True when the Rust core is loaded |
| `nx._core` | Native module (version, founder, lock, pager, .nex) |

---

## 2. Core Building Blocks

### `nx.Parameter(data, requires_grad=True)`
Trainable tensor wrapper. `.data`, `.grad`, `.shape`, `.size`, `.zero_grad()`.

### `nx.Module`
Base class. `forward()`, `parameters()`, `named_parameters()`, `modules()`,
`train()`, `eval()`, `zero_grad()`, `parameter_count()`, `state_dict()`,
`load_state_dict()`.

### `nx.FractalLayer(in_features, out_features, conceptual_params=None, K=1024, bias=True, seed=None)`
Weight matrix generated from a harmonic seed.
`bloom(rows, cols)`, `forward(x)`, `memory_usage() -> dict`.

### `nx.FractalSeed(a, w1, w2, phi)`
`.K`, `.num_parameters`, `.memory_bytes`, `.random(K, seed)`,
`.from_hash(text, K)`, `.to_dict()`, `.from_dict()`.

### `nx.FractalTensor(rows, cols, seed=None, K=1024)`
`.bloom_element(i,j)`, `.bloom_row(i)`, `.bloom_block(a,b)`, `.bloom_all()`,
`.matmul(x, block_size=64)`, `.compression_ratio`.

### `nx.LiquidRouter(num_neurons, awake_threshold=0.05, temperature=1.0, temperature_decay=0.999)`
`.route(x, hard=False) -> mask`, `.step()`, `.get_stats()`, `.sparsity`,
`.effective_awake`, `.memory_savings()`, `.reset_stats()`.

---

## 3. Optimizers and Losses

### Neura-X optimizers
- `nx.ShadowAdamW(params, lr=1e-3, rank=64, betas=(0.9,0.999), eps=1e-8, weight_decay=0.01, rotation_interval=100)` — `.step()`, `.rotate_subspace()`, `.memory_usage()`.
- `nx.ShadowSGD(params, lr=0.01, rank=64)`
- `nx.CircadianOptimizer(model, base_optimizer=None, sleep_interval=100, critic_lr=1e-3)` — `.wake_step(loss, hidden)`, `.sleep()`, `.log_surprise(v)`.

### Standard optimizers (`nx.optim`)
`AdamW`, `Adam`, `SGD`, `RMSprop`, `CosineAnnealingLR`, `LinearLR` — same
mathematics as PyTorch equivalents.

### Losses (`nx.loss`)
`CrossEntropyLoss(reduction)`, `MSELoss(reduction)`, `MAELoss(reduction)`,
`L1Loss(reduction)`.

---

## 4. Model Types

All accept `conceptual_params` and expose `forward`, `export`, `infer`/task methods.

| Class | Key arguments | Generation method |
|-------|---------------|-------------------|
| `nx.LLM` | `architecture="transformer\|mamba\|hybrid"` | `.infer(prompt)` |
| `nx.ImageGen` | `architecture="latent_diffusion\|dit\|gan\|vae"`, `resolution` | `.generate(prompt, steps)` |
| `nx.VideoGen` | `architecture="temporal_diffusion"`, `resolution`, `fps`, `duration_seconds` | `.generate(prompt, frames)` |
| `nx.AudioGen` | `architecture="autoregressive_transformer"`, `sample_rate`, `duration_seconds` | `.generate(prompt)` |
| `nx.SpeechSynthesis` | `architecture="vits"`, `languages` | `.synthesize(text)` |
| `nx.SpeechRecognition` | `architecture="whisper_style"`, `languages` | `.transcribe(audio)` |
| `nx.Predictive` | `task="time_series\|regression\|classification"`, `input_features`, `horizon` | `.predict(data)` |
| `nx.MoE` | `num_experts`, `experts_per_token`, `conceptual_params_per_expert`, `dynamic_expert_growth` | routed `forward` |
| `nx.Vision` | task-specific heads | `.predict(images)` |

---

## 5. Training Engines

```python
engine = nx.TrainingEngine(model)          # standard training
engine = nx.PreTrainingEngine(model)       # from scratch
engine = nx.FineTuneEngine(model)          # adaptation

engine.train(dataset=..., optimizer="adamw", lr=1e-4, scheduler="cosine", epochs=10)
engine.train_from_scratch(dataset=..., target_loss=2.1)
engine.fine_tune(dataset="instructions.jsonl")
```
Advanced loop (PyTorch-style) is fully supported:

`nx.optim.AdamW(model.parameters(), lr)`, `nx.loss.CrossEntropy(...)`,
`loss.backward()`, `optimizer.step()`, `scheduler.step()`.

---

## 6. Data

> nx.StreamLoader(path, chunk_size="64MB", tokenizer="bpe", skip_learned=True, safety_filter=True, batch_size=8, shuffle=True, seed=None)

Zero-copy chunked streaming. `.stats()`, `.should_skip(id, loss)`,
`.from_pandas(df)`. Iterates batches without loading the dataset into RAM.

---

## 7. Cmposability

```python
model = nx.compose(brain=nx.LLM(...), modules=[
    nx.module.Prediction(task="time_series"),
    nx.module.ImageGen(resolution=512),
    nx.module.AudioGen(sample_rate=32000),
    nx.module.VideoGen(resolution=256, fps=8),
    nx.module.Tools([nx.tools.web_search, nx.tools.run_code]),
    nx.module.Skills(["swahili", "medical"]),
    nx.module.Memory(), nx.module.Reasoning(), nx.module.AgentCore(),
])
model.attach(module)      # hot-swap in
model.detach(name)        # hot-swap out
model.list_modules(); model.get_module(name)
model.fine_tune_together(dataset=...)
```

Available:
>Prediction, ImageGen, ImageUnderstand, VideoGen, AudioGen, SpeechSynthesis, SpeechRecognition, Tools, Skills, Memory, Reasoning, CodeExecution, MultiModal, AgentCore, Custom.

---

## 8. Tools, Skills, Agents

```python
@nx.tool(description="...")
def my_tool(arg: str) -> str: ...

model.attach_tools([my_tool, nx.tools.web_search, nx.tools.run_code,
                    nx.tools.query_db, nx.tools.read_file, nx.tools.file_write])
response = model.infer(prompt, use_tools=True)

model.install_skill("swahili")
model.install_skill(nx.hub.pull_skill("edusei/nairobi_guide"))

agent = nx.Agent(brain="model.nex", skills=[...], tools=[...],
                 memory=nx.HolographicMemory(capacity=10_000), circadian=True)
agent.run(task)
agent.memory.store(concept=..., data=..., importance=...)
agent.memory.recall(query)

team = nx.AgentTeam(agents=[a, b, c], coordinator="model.nex")
team.run(task)
```

**Custom skills:** decorate a class with `@nx.skill(name=..., description=...)`,
implement `ingest()`, `compress_to_fractal()`, `forward()`.

---

## 9. Evaluation, Serving, Conversion, Hub

```python
results = nx.EvalSuite().run(model, benchmarks=["mmlu","humaneval","gsm8k",
                            "hellaswag","truthfulqa","arc","winogrande"])
results.mmlu_score; results.memory_to_intelligence_ratio

nx.serve("model.nex", port=8000)        # OpenAI-compatible /v1 endpoints

nx.convert.from_pytorch("model.pt"); nx.convert.from_gguf("m.gguf")
nx.convert.from_huggingface("org/model"); model.export_gguf(...); model.export_onnx(...)
model.export("model.nex"); nx.load("model.nex")

nx.hub.push / pull / push_module / pull_module / push_skill / pull_skill
```

---

## 10. Observability
- `nx.xray(model)` — bloom latency, router sparsity, shadow rank, memory.
- `nx.DreamDashboard(model).open()` — Wake/Sleep visualization, surprise patterns.
- `nx.BenchmarkSuite()`.run(model, iterations) — RAM, CPU, tokens/s.

---

## 11. Safety and Privacy
- `nx.SafetyFilter(strictness)` — `.filter(text)`, `.is_safe(text)`.
- `nx.PrivacyEngine(epsilon, delta, max_grad_norm)` — `.add_noise(grad)` (differential privacy on Shadow gradients).

---

## 12. Math and Utils Namespaces
- `nx.math.genesis_gaussian(K=1024, seed=None)` — CLT-based seed.
- `nx.math.fibonacci_spiral(K=1024)` — deterministic golden-ratio seed.
- `nx.utils.search / sandbox / sql` — tool backends.

---

## 13. Rust Core (nx._core)
`get_version()`, `get_founder()`, `get_tagline()`, `get_cpu_info()`,
`create_pager(ram_budget_mb, backing_dir)`,
`serialize_nex(model_type, conceptual_params, path)`,
`get_founders_lock_signature()`.

---

## 14. CLI

`neura-x` / `nx` entry points (defined in `pyproject.toml`).