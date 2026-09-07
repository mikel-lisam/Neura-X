# ==========================================================
# Neura-X: Intelligence Without Limits.
# Copyright (c) 2026 Edusei Mikel Lisamba. All Rights Reserved.
# Neura Hub — Model Repository
# ==========================================================

"""Neura Hub: pluggable model repository.

The hub is structured around an :class:`HubBackend` interface so the
local-filesystem backend (the default, dependency-free) can be swapped
for a remote HTTP backend without touching call sites.

Extension points:

* :func:`set_backend` — install a custom backend implementation.
* :meth:`HubBackend.list` — enumerate artifacts the backend knows about.
* :meth:`HubBackend.delete` — remove an artifact.
* :meth:`HubBackend.publish` — upload to a remote hub.
* :meth:`HubBackend.fetch` — download from a remote hub.
"""

import os
import shutil
import urllib.error
import urllib.request
from typing import Dict, List, Optional, Tuple


_HUB_ROOT = os.path.join(os.path.expanduser("~"), ".neura_x", "hub")


def _ensure_root() -> str:
    os.makedirs(_HUB_ROOT, exist_ok=True)
    return _HUB_ROOT


# ──────────────────────────────────────────────────────────────────────
# Backend interface
# ──────────────────────────────────────────────────────────────────────


class HubBackend:
    """Abstract base for hub backends.

    Subclasses override :meth:`push`, :meth:`pull`, :meth:`list`,
    :meth:`delete`, :meth:`publish`, and :meth:`fetch`.  All methods
    return ``None`` on missing artifacts and raise ``FileNotFoundError``
    on hard failures.
    """

    def push(self, path: str, name: str, version: str) -> str:  # pragma: no cover
        raise NotImplementedError

    def pull(self, model_id: str, version: str) -> Optional[str]:  # pragma: no cover
        raise NotImplementedError

    def list(self) -> List[Tuple[str, str]]:
        return []

    def delete(self, model_id: str, version: str = None) -> bool:
        return False

    def publish(self, path: str, name: str, version: str, remote_url: str) -> bool:
        """Upload to a remote hub via HTTP PUT.  Requires a configured URL."""
        if not remote_url:
            return False
        url = remote_url.rstrip("/") + f"/{name}/{version}"
        try:
            with open(path, "rb") as f:
                data = f.read()
            req = urllib.request.Request(
                url, data=data, method="PUT",
                headers={"Content-Type": "application/octet-stream"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                return 200 <= resp.status < 300
        except (urllib.error.URLError, OSError):
            return False

    def fetch(self, model_id: str, version: str, remote_url: str) -> Optional[str]:
        """Download from a remote hub via HTTP GET."""
        if not remote_url:
            return None
        url = remote_url.rstrip("/") + f"/{model_id}/{version}"
        try:
            target = os.path.join(_ensure_root(), model_id, version)
            os.makedirs(target, exist_ok=True)
            out_path = os.path.join(target, os.path.basename(model_id))
            with urllib.request.urlopen(url, timeout=30) as resp, open(out_path, "wb") as out:
                shutil.copyfileobj(resp, out)
            return out_path
        except (urllib.error.URLError, OSError):
            return None


class LocalHubBackend(HubBackend):
    """File-system backed hub.  Default backend."""

    def __init__(self, root: str = _HUB_ROOT):
        self.root = root

    def push(self, path: str, name: str, version: str) -> str:
        if not os.path.exists(path):
            raise FileNotFoundError(f"No such artifact: {path}")
        os.makedirs(self.root, exist_ok=True)
        target = os.path.join(self.root, name, version)
        os.makedirs(target, exist_ok=True)
        dest = os.path.join(target, os.path.basename(path))
        if os.path.isdir(path):
            if os.path.exists(dest):
                shutil.rmtree(dest)
            shutil.copytree(path, dest)
        else:
            shutil.copy2(path, dest)
        return dest

    def pull(self, model_id: str, version: str) -> Optional[str]:
        candidate = os.path.join(self.root, model_id, version)
        if os.path.isdir(candidate):
            files = [f for f in os.listdir(candidate) if not f.startswith(".")]
            return candidate if files else None
        return None

    def list(self) -> List[Tuple[str, str]]:
        if not os.path.isdir(self.root):
            return []
        result: List[Tuple[str, str]] = []
        for name in sorted(os.listdir(self.root)):
            for version in sorted(os.listdir(os.path.join(self.root, name))):
                result.append((name, version))
        return result

    def delete(self, model_id: str, version: str = None) -> bool:
        target = os.path.join(self.root, model_id, version or "")
        if version is None:
            target = os.path.join(self.root, model_id)
        if not os.path.isdir(target):
            return False
        shutil.rmtree(target)
        return True


# ──────────────────────────────────────────────────────────────────────
# Public namespace
# ──────────────────────────────────────────────────────────────────────


_BACKEND: HubBackend = LocalHubBackend()
_REMOTE_URL: Optional[str] = None


def set_backend(backend: HubBackend) -> None:
    """Install a custom hub backend (e.g., for tests or remote deployments)."""
    global _BACKEND
    _BACKEND = backend


def set_remote_url(url: Optional[str]) -> None:
    """Configure the remote hub URL used by :py:meth:`publish` and :py:meth:`fetch`."""
    global _REMOTE_URL
    _REMOTE_URL = url


def get_backend() -> HubBackend:
    """Return the currently-installed backend."""
    return _BACKEND


class _HubNamespace:
    """Namespace for ``nx.hub.*`` access."""

    @staticmethod
    def push(path: str, name: str = None, version: str = "1.0") -> str:
        """Copy a local artifact into the Neura Hub."""
        artifact_name = name or os.path.splitext(os.path.basename(path))[0]
        dest = _BACKEND.push(path, artifact_name, version)
        print(f"[Neura Hub] Pushed: {artifact_name} v{version} -> {dest}")
        return dest

    @staticmethod
    def pull(model_id: str, version: str = "1.0") -> Optional[str]:
        """Locate a previously-pushed model in the local Hub."""
        local = _BACKEND.pull(model_id, version)
        if local:
            print(f"[Neura Hub] Pulled: {model_id} v{version} -> {local}")
            return local
        if _REMOTE_URL:
            fetched = _BACKEND.fetch(model_id, version, _REMOTE_URL)
            if fetched:
                print(f"[Neura Hub] Fetched: {model_id} v{version} -> {fetched}")
                return fetched
        print(f"[Neura Hub] Pulled: {model_id} v{version} (not found locally)")
        return None

    @staticmethod
    def publish(path: str, name: str = None, version: str = "1.0", remote_url: str = None) -> bool:
        """Upload an artifact to a remote hub."""
        if not _REMOTE_URL and remote_url is None:
            print("[Neura Hub] No remote URL configured; publish skipped")
            return False
        url = remote_url or _REMOTE_URL
        artifact_name = name or os.path.splitext(os.path.basename(path))[0]
        ok = _BACKEND.publish(path, artifact_name, version, url)
        print(f"[Neura Hub] Publish {artifact_name} v{version} -> {'OK' if ok else 'FAILED'}")
        return ok

    @staticmethod
    def list() -> List[Tuple[str, str]]:
        """Enumerate all artifacts known to the backend."""
        return _BACKEND.list()

    @staticmethod
    def delete(model_id: str, version: str = None) -> bool:
        """Remove an artifact (or all versions) from the hub."""
        ok = _BACKEND.delete(model_id, version)
        print(f"[Neura Hub] Delete {model_id} v{version or '*'}: {'OK' if ok else 'NOT FOUND'}")
        return ok

    @staticmethod
    def push_module(name: str, version: str = "1.0") -> str:
        """Push a module directory to the Hub (creates a stub dir)."""
        target = os.path.join(_ensure_root(), "__module__", name, version)
        os.makedirs(target, exist_ok=True)
        with open(os.path.join(target, "MODULE.md"), "w") as f:
            f.write(f"# {name}\nVersion: {version}\n")
        print(f"[Neura Hub] Pushed module: {name} v{version} -> {target}")
        return target

    @staticmethod
    def pull_module(module_id: str, version: str = "1.0") -> Optional[str]:
        """Locate a previously-pushed module in the local Hub."""
        candidate = os.path.join(_ensure_root(), "__module__", module_id, version)
        if os.path.isdir(candidate):
            print(f"[Neura Hub] Pulled module: {module_id} v{version} -> {candidate}")
            return candidate
        print(f"[Neura Hub] Pulled module: {module_id} v{version} (not found locally)")
        return None

    @staticmethod
    def push_skill(name: str, version: str = "1.0") -> str:
        """Push a skill directory to the Hub (creates a stub dir)."""
        target = os.path.join(_ensure_root(), "__skill__", name, version)
        os.makedirs(target, exist_ok=True)
        with open(os.path.join(target, "SKILL.md"), "w") as f:
            f.write(f"# {name}\nVersion: {version}\n")
        print(f"[Neura Hub] Pushed skill: {name} v{version} -> {target}")
        return target

    @staticmethod
    def pull_skill(skill_id: str, version: str = "1.0") -> Optional[str]:
        """Locate a previously-pushed skill in the local Hub."""
        candidate = os.path.join(_ensure_root(), "__skill__", skill_id, version)
        if os.path.isdir(candidate):
            print(f"[Neura Hub] Pulled skill: {skill_id} v{version} -> {candidate}")
            return candidate
        print(f"[Neura Hub] Pulled skill: {skill_id} v{version} (not found locally)")
        return None


hub = _HubNamespace()