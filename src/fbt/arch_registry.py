"""Architecture registry.

Maps an architecture name to ``archs/<name>/model.py`` and loads it by file
path, so the lookup does not depend on ``sys.path`` state. Checkpoints store
{"arch": ARCH, "model_cfg": CFG-dict}; ``build_from_ckpt`` rebuilds the exact
trained architecture. Checkpoints without an "arch" key are rejected.
"""
from __future__ import annotations

import importlib.util
import os
import sys

# Repo root: this file is src/fbt/arch_registry.py, so 3x dirname.
VL_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
ARCHS_DIR = os.path.join(VL_ROOT, "archs")

ARCH_NAMES = ("fftconv",)


def load_arch_module(name: str):
    assert name in ARCH_NAMES, f"unknown arch {name!r}"
    path = os.path.join(ARCHS_DIR, name, "model.py")
    key = f"fbt_arch_{name}"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod          # dataclass/annotation resolution safety
    spec.loader.exec_module(mod)
    return mod


def build_arch(name: str, cfg: dict | None = None):
    return load_arch_module(name).build(cfg)


def build_from_ckpt(ck: dict, device="cuda"):
    """Rebuild the trained model from a checkpoint dict. Returns (model, arch_name).
    The ckpt must carry an "arch" key (fftconv)."""
    if "arch" not in ck:
        raise ValueError(
            "checkpoint does not name a known architecture: it has no "
            "'arch' key. Save the checkpoint with an 'arch' entry naming the "
            "architecture it was trained with.")
    m = build_arch(ck["arch"], ck.get("model_cfg"))
    m.load_state_dict(ck["model"])
    return m.to(device).eval(), ck["arch"]
