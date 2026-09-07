# 08 — Serving & Conversion

## Serve an OpenAI-compatible API

```python
import neura_x as nx
nx.serve("my-model.nex", port=8000)
```

Endpoints: `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/models`. Point any OpenAI client at `http://localhost:8000/v1` with any API key zero code changes.


```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="neura-x")
client.chat.completions.create(model="my-model", messages=[...])
```

---

## Convert existing models

```python
model = nx.convert.from_pytorch("model.pt")
model = nx.convert.from_gguf("model.gguf")
model = nx.convert.from_huggingface("meta-llama/Llama-3-8B")
model.export("converted.nex")
```

---

## Export outwrd

```python
model.export_gguf("model.gguf")
model.export_onnx("model.onnx")
```

---

## Share on the Hub

```python
nx.hub.push("my-model.nex", name="my-model-v1")
model = nx.hub.pull("mikeledusei/my-model-v1")
```

---

## Ecosystem interop

```python
import numpy as np, pandas as pd
arr = tensor.to_numpy()
loader = nx.StreamLoader.from_pandas(df)
from langchain.llms import NeuraXLLM   # drop-in LangChain backend
```

Local. Private. Free. Forever.