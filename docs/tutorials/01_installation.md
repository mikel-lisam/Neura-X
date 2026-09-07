# 01 — Installation & First Import

## Install from PyPI

```bash
pip install neura-x
```
## Install from source

```bash
git clone https://github.com/mikeledusei/Neura-X.git
cd Neura-X
make all          # C → C++ → Rust → wheel → install
```
make all compiles the C HAL, the C++ engine, the Rust crates, builds the wheel with maturin, installs it, and copies the native core into the source package so local development works too.

## Verify

```python
import neura_x as nx
print(nx.__version__, nx._core_available)
```
Expected banner:

```bash
⚡ Neura-X v1.0.0
   Intelligence Without Limits.
────────────────────────────────────────────────
🧠 Founded by Edusei Mikel Lisamba
🌍 Built in Kenya | Open University of Kenya
────────────────────────────────────────────────
CPU:     Intel(R) Core(TM) i7-5600U CPU @ 2.60GHz
RAM:     7.6 GB
Engine:  Fractal Tensor + Circadian Learning
Core:    Rust/C/C++ Native ✅
Status:  All systems operational. Ready to train.
```
## Run the test suites

```bash
cd python && python -m pytest tests/ -v     # 28 unit tests
python verify_real_tests.py                  # 7 real-computation proofs
```

## Troubleshooting

| **Symptom** | **Fix** |
| :--- | :--- |
| `Core: Python Fallback` | Run `make all` again |
| pip can't reach PyPI | `pip install wheel --no-deps` |
| Old CPU without AVX2 | Scalar fallbacks engage automatically |

---