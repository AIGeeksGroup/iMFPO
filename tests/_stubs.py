"""Make ``isaaclab_fpo`` importable without Isaac Lab / Isaac Sim (pure-torch tests).

Only ``isaaclab.utils.configclass`` is needed by ``rl_cfg`` / ``task_cfgs``. The stub keeps
the configclass semantics the configs rely on: class attributes (also unannotated
overrides in subclasses) are per-instance deep-copied defaults, kwargs override them,
and ``__post_init__`` runs last.
"""

from __future__ import annotations

import copy
import pathlib
import sys
import types
from dataclasses import MISSING

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _configclass(cls):
    def _fields(klass):
        names: dict[str, None] = {}
        for base in reversed(klass.__mro__):
            if base is object:
                continue
            for name in getattr(base, "__annotations__", {}):
                names[name] = None
            for name, value in vars(base).items():
                if name.startswith("_") or isinstance(
                    value, (types.FunctionType, staticmethod, classmethod, property)
                ):
                    continue
                names[name] = None
        return list(names)

    def __init__(self, **kwargs):
        for name in _fields(type(self)):
            value = getattr(type(self), name, MISSING)
            setattr(self, name, copy.deepcopy(value))
        for name, value in kwargs.items():
            setattr(self, name, value)
        post_init = getattr(self, "__post_init__", None)
        if post_init is not None:
            post_init()

    cls.__init__ = __init__
    return cls


def install() -> None:
    if "isaaclab.utils" not in sys.modules:
        isaaclab = types.ModuleType("isaaclab")
        utils = types.ModuleType("isaaclab.utils")
        utils.configclass = _configclass
        isaaclab.utils = utils
        sys.modules["isaaclab"] = isaaclab
        sys.modules["isaaclab.utils"] = utils
    src = str(ROOT / "source" / "isaaclab_fpo")
    if src not in sys.path:
        sys.path.insert(0, src)
