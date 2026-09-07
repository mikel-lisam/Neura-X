# Deploying Neura-X to PyPI
## The Official Release Guide — v1.0

- **Author:** Edusei Mikel Lisamba — Open University of Kenya
- **Applies to:** neura-x ≥ 1.0.0
- **Build system:** maturin (mixed Rust + Python + native C/C++ payload)

---

## 0. How Neura-X Is Distributed

```text
PyPI artifacts = platform wheels ONLY (no sdist in v1.0.0)

  neura_x-1.0.0-cp312-cp312-manylinux_2_34_x86_64.whl
  neura_x-1.0.0-cp312-cp312-macosx_11_0_arm64.whl
  neura_x-1.0.0-cp312-cp312-macosx_10_12_x86_64.whl
  neura_x-1.0.0-cp312-cp312-win_amd64.whl

Why wheels-only:
  - The package embeds compiled Rust (_core) plus native C/C++ libraries.
  - Building from source requires gcc + g++ + cargo + make; that is a
    developer workflow (`make all`), not an end-user workflow.
  - sdist support is on the roadmap for v1.1.
```

---

## 1. Phase 0 — One-Time Account & Security Setup

### 1.1 Create accounts (2FA is mandatory on PyPI since 2024)

```text
1. https://pypi.org/account/register/     → production account
2. https://test.pypi.org/account/register/ → separate test account
3. Enable TOTP two-factor authentication on BOTH.
4. Add a recovery code set and store it offline.
```

### 1.2 Check name availability

```bash
curl -s -o /dev/null -w "%{http_code}" https://pypi.org/project/neura-x/
# 404 → name is FREE.  200 → name taken; fallbacks:
#   neura-x-framework, neurax-ai, neura-x-core
```

### 1.3 Choose an upload authentication method

| Method | Best for | Setup |
|--------|----------|-------|
| **Trusted publishing (OIDC)** — recommended | CI releases | No secrets stored; PyPI trusts your GitHub workflow |
| API token | Manual/local uploads | Scoped token in `~/.pypirc` or CI secret |

**Trusted publishing setup (production):**

```text
PyPI → Account settings → Publishing → Add a new publisher
  Project name:   neura-x            (must match, or "pending" for new project)
  Owner:          mikeledusei
  Repository:     Neura-X
  Workflow name:  build_wheels.yml
  Environment:    pypi
```

Repeat on **TestPyPI** with environment name `test-pypi`.

**API token setup (manual):**

```text
PyPI → Account settings → API tokens → Create
  Scope: project "neura-x" (or entire account for first publish)
Store the token ONCE; it is shown only once.
```

`~/.pypirc`:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
repository = https://upload.pypi.org/legacy/
username = __token__
password = <YOUR_PYPI_TOKEN>

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = <YOUR_TESTPYPI_TOKEN>
```

```bash
chmod 600 ~/.pypirc     # never commit this file (gitignore already covers secrets)
```

---

## 2. Phase 1 — Pre-Flight (run before every release)

### 2.1 Version lives in FOUR places — keep them in sync

```text
pyproject.toml                     version = "1.0.0"
python/neura_x/__init__.py         __version__ = "1.0.0"
core/rust/pager/Cargo.toml         version = "1.0.0"
core/rust/nex_format/Cargo.toml    version = "1.0.0"
core/rust/serve/Cargo.toml         version = "1.0.0"
core/rust/python_bridge/Cargo.toml version = "1.0.0"
```

```bash
./scripts/bump_version.sh 1.0.0     # edits all six atomically
```

### 2.2 Make the README render on PyPI

Relative image paths do NOT render on PyPI. Use absolute raw URLs:

```markdown
<p align="center">
  <img src="https://raw.githubusercontent.com/mikeledusei/Neura-X/main/assets/logo.png"
       alt="Neura-X" width="400"/>
</p>
```

### 2.3 License correctness on PyPI

```text
Neura-X is dual-licensed (Community + Commercial). On PyPI:
  - Keep classifier:  License :: Other/Proprietary License
  - NEVER add an OSI classifier (MIT/Apache) — that would be false.
  - The full LICENSE file ships inside the wheel (maturin includes it).
```

### 2.4 Run the release gate

```bash
pip install twine auditwheel
./scripts/release_check.sh
```

The gate verifies: clean git tree, version sync, pytest + verification
suites, distribution-mode build, native library self-containment, wheel
size, `twine check`, and a fresh-venv import with the Rust core alive.

### 2.5 Native self-containment rule (critical)

Wheels must not depend on libraries users do not have:

```text
ALLOWED runtime deps of bundled .so files:
  linux-vdso, libc, libm, libpthread/libdl, libgcc_s, libstdc++

FORBIDDEN (breaks manylinux / user machines):
  libopenblas, liblapack, libLLVM-*
```

Build in distribution mode to guarantee this:

```bash
NEX_DIST=1 make all        # disables optional BLAS/LAPACK/LLVM linking
ldd build/libneura_c.so    # inspect
ldd build/libneura_cpp.so
```

(The C layer falls back to built-in BLAS kernels; performance-critical
users can install OpenBLAS locally and build from source.)

---

## 3. Phase 2 — Build & Inspect

### 3.1 Local single-platform build

```bash
NEX_DIST=1 make all
ls -la dist/
python -m auditwheel show dist/*.whl     # confirms manylinux tag
python -m twine check dist/*.whl         # README/metadata render check
```

### 3.2 All-platform build (the real release path)

Push a tag; CI builds the matrix:

```bash
git add -A
git commit -m "release: v1.0.0 — Intelligence Without Limits"
git tag -a v1.0.0 -m "Neura-X v1.0.0"
git push origin main --tags
```

`.github/workflows/build_wheels.yml` then produces:
linux-x86_64, macos-arm64, macos-x86_64, windows-x64 wheels as artifacts.

### 3.3 Wheel anatomy check (once per release)

```bash
unzip -l dist/neura_x-1.0.0-*.whl | grep -E "_core|libneura|LICENSE|__init__"
# must show:
#   neura_x/_core.cpython-*.so          ← Rust core
#   neura_x/libneura_c.so               ← C payload (optional payload)
#   neura_x/libneura_cpp.so             ← C++ payload
#   neura_x-1.0.0.dist-info/LICENSE     ← dual license shipped
```

---

## 4. Phase 3 — TestPyPI Dry Run (never skip)

### 4.1 Upload

```bash
# Trusted publishing via CI (environment: test-pypi), or manually:
python -m twine upload --repository testpypi dist/*.whl
```

### 4.2 Install from TestPyPI into a clean venv

```bash
python -m venv /tmp/nxtest && source /tmp/nxtest/bin/activate
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ \
            neura-x
python -c "import neura_x as nx; print(nx.__version__, nx._core_available)"
python -m pytest --pyargs neura_x   # if you ship tests; else run smoke script
deactivate
```

`--extra-index-url` is required because `numpy` resolves from real PyPI.

### 4.3 TestPyPI acceptance criteria

```text
[ ] Banner prints with Core: Rust/C/C++ Native ✅
[ ] nx._core.get_founders_lock_signature() returns 64-hex chars
[ ] verify_real_tests.py passes against the installed package
[ ] Project page renders logo, README tables, and license text
```

---

## 5. Phase 4 — Production Release

### 5.1 Via CI trusted publishing (recommended)

The `publish` job in `build_wheels.yml` runs only on `v*` tags, downloads
all wheel artifacts, and uploads with OIDC — no token stored anywhere:

```yaml
publish:
  runs-on: ubuntu-latest
  needs: build
  if: startsWith(github.ref, 'refs/tags/v')
  environment: pypi
  permissions:
    id-token: write
  steps:
    - uses: actions/download-artifact@v4
      with: { pattern: "wheels-*", path: dist, merge-multiple: true }
    - uses: pypa/gh-action-pypi-publish@release/v1
      # no password: OIDC trusted publishing
```

### 5.2 Manual upload (fallback)

```bash
python -m twine upload --repository pypi dist/*.whl
```

### 5.3 Immediate post-upload verification

```bash
pip install --upgrade neura-x            # from a clean venv
python -c "import neura_x as nx; print(nx.__version__)"
pip show neura-x                          # license, size, files
curl -s https://pypi.org/pypi/neura-x/json | python -m json.tool | head -40
```

### 5.4 Announce

```text
1. GitHub Release from the v1.0.0 tag (notes = CHANGELOG section).
2. Pin the release wheels as artifacts on the Release page.
3. Update README badge: version-1.0.0 → new version.
```

---

## 6. Phase 5 — Post-Release Operations

### 6.1 Monitoring (first 72 hours)

```text
- PyPI project page → download stats trend
- GitHub Issues filtered by label:bug + version
- `pip install neura-x` on: Ubuntu 24.04, macOS 14 (ARM), Windows 11
- Genesis Machine smoke run (the zero-excuse threshold)
```

### 6.2 Fixing a bad release — YANK, never delete

```text
PyPI → Manage project → Release → Yank   (hides from new installs,
keeps existing pins reproducible)
Then publish v1.0.1 with the fix. Deleted files can never be re-uploaded
with the same name+version — always bump.
```

### 6.3 Hotfix cadence

```text
patch  (1.0.x) : bug fixes, no API change   → same week
minor  (1.x.0) : new modules/model types    → scheduled
major  (x.0.0) : .nex format change         → with migration guide
```

### 6.4 Rotating credentials

```text
- API tokens: rotate every 12 months or on any contributor change.
- Trusted publishing: review publisher list quarterly.
- Never place tokens in Makefiles, workflows, or docs.
```

---

## 7. Appendix A — Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `403 Invalid or non-existent authentication` | Wrong/missing token or 2FA enforcement | Re-create scoped token; use trusted publishing |
| `File already exists` | Version already published | Bump patch version; yank old if broken |
| Wheel tagged `linux_x86_64` (not manylinux) | Non-portable libs linked | Build with `NEX_DIST=1`; rerun `auditwheel show` |
| `auditwheel` reports libopenblas | BLAS linked into wheel | `NEX_DIST=1 make all` |
| README images missing on PyPI | Relative paths | Use raw.githubusercontent absolute URLs |
| TestPyPI install fails on numpy | Dep not on TestPyPI | Add `--extra-index-url https://pypi.org/simple/` |
| `_core_available False` after install | Wheel built without Rust module | Check `maturin` bindings + `module-name = "neura_x._core"` |
| Upload > 100 MiB rejected | Wheel too large | Strip debug symbols; ship payload libs compressed |

---

## 8. Appendix B — Release Checklist (copy per release)

```text
[ ] ./scripts/bump_version.sh X.Y.Z
[ ] CHANGELOG.md updated; README badge updated
[ ] git tree clean; all tests green (pytest + verify_real_tests)
[ ] NEX_DIST=1 make all  → ldd clean (no openblas/llvm)
[ ] twine check + auditwheel show pass
[ ] Tag vX.Y.Z pushed → CI matrix wheels built
[ ] TestPyPI upload + clean-venv install verified
[ ] Production publish (OIDC) verified from clean venv
[ ] GitHub Release published with notes + artifacts
[ ] 72-hour monitoring window opened
```

---

*Neura-X: Intelligence Without Limits.*
*From a Dell Latitude E7450 in Kenya — to every `pip install` on Earth.*