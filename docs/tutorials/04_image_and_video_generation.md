# 04 — Image & Video Generation on CPU

## Image model

```python
import neura_x as nx

img = nx.ImageGen(architecture="latent_diffusion",
                  conceptual_params=2_500_000_000,
                  resolution=512)

nx.TrainingEngine(img).train_from_scratch(dataset="images/")

image = img.generate("A sunset over Nairobi", steps=20)
image.save("sunset.png")
```
Fractal Tensors compress the UNet/DiT; the Liquid Router sparsifies attention; Ghost Tensors offload denoising activations.

---

## Video model

```python
vid = nx.VideoGen(architecture="temporal_diffusion",
                  conceptual_params=5_000_000_000,
                  resolution=256, fps=8, duration_seconds=4)

nx.TrainingEngine(vid).train_from_scratch(dataset="videos/")

clip = vid.generate("A cheetah running across the savanna", frames=32)
clip.save("cheetah.mp4")
```

Predictive Coding stores frame 1 fully and frames 2–32 as surprise deltas, ≈90% less temporal memory.

## Memory expectations

| **Model** | **Standard VRAM** | **Neura-X RAM** |
| :--- | :--- | :--- |
| 512px latent diffusion | 16 GB | 1.5 GB |
| 4s 256px video | 40 GB | 3 GB |

---