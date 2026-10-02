"""Configuration loading: packaged defaults + optional profile overlay + optional
user overrides from ./config/*.toml in the working directory."""

from __future__ import annotations

import re
import tomllib
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

PACKAGED_CONFIG_DIR = Path(__file__).resolve().parent / "config"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def load_config(profile: str | None = None, extra_path: str | Path | None = None) -> dict[str, Any]:
    """Load default.toml, overlay a named profile (e.g. 'pi5'), then an explicit file.

    Named files are looked up first in ./config/ (working directory, so users can
    edit without touching the installed package), then in the packaged config dir.
    """
    config = _read_toml(PACKAGED_CONFIG_DIR / "default.toml")

    def resolve(name: str) -> Path:
        candidates = (Path.cwd() / "config" / f"{name}.toml", PACKAGED_CONFIG_DIR / f"{name}.toml")
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(f"config profile '{name}' not found in ./config/ or packaged")

    if profile:
        config = _deep_merge(config, _read_toml(resolve(profile)))
    if extra_path:
        config = _deep_merge(config, _read_toml(Path(extra_path)))
    return config


def resolve_training_cfg(config: dict, track: str, warm: bool = False,
                         agent: str | None = None) -> dict:
    """[training] merged with [training_presets.<track>], then (when ``agent``
    is given) [training_presets_<agent>.<track>], and, when ``warm``,
    [training_warm] (+ [training_warm_gp] on top for gp).

    The one precedence rule for per-track training recipes, shared by the
    server and headless training (kept here so it needs no server imports).
    The per-agent layer exists because the two agents want different
    recipes: e.g. the exploration floor that keeps the circuit lapping on gp
    stops the MLP from learning it.
    """
    cfg = dict(config["training"])
    cfg.update(config.get("training_presets", {}).get(track, {}))
    if agent:
        cfg.update(config.get(f"training_presets_{agent}", {}).get(track, {}))
    if warm:
        cfg.update(config.get("training_warm", {}))
        if track == "gp":
            cfg.update(config.get("training_warm_gp", {}))
    return cfg


_BARE_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]*")


def parse_override(spec: str) -> tuple[str, Any]:
    """Split one ``section.key=value`` override into (dotted key, value).

    The value is parsed as a TOML literal (``6``, ``0.5``, ``true``,
    ``"huber"``, ``[1, 2]``, ``{ head = 0.1 }``); a bare word that is not
    valid TOML (``huber``) is taken as a string, sparing the shell quoting.
    """
    key, sep, raw = spec.partition("=")
    key, raw = key.strip(), raw.strip()
    if not sep or not raw or len(key.split(".")) < 2 or not all(key.split(".")):
        raise ValueError(
            f"bad config override '{spec}': expected section.key=value "
            "(e.g. training.gamma=0.99)"
        )
    if raw.lower() in ("true", "false"):  # accept Python-style True/False too
        return key, raw.lower() == "true"
    try:
        doc = tomllib.loads(f"value = {raw}")
    except tomllib.TOMLDecodeError:
        if not _BARE_WORD.fullmatch(raw):
            raise ValueError(
                f"bad config override '{spec}': '{raw}' is not a TOML literal "
                "(strings need quotes, e.g. training.loss='\"huber\"')"
            ) from None
        return key, raw
    if set(doc) != {"value"}:  # further lines after the literal: not ONE value
        raise ValueError(
            f"bad config override '{spec}': expected a single TOML literal after '='"
        )
    return key, doc["value"]


def apply_overrides(
    config: dict[str, Any], overrides: Mapping[str, Any] | Iterable[str] | None
) -> dict[str, Any]:
    """Set dotted-key overrides on ``config`` IN PLACE (and return it).

    ``overrides`` is a ``{"section.key": value}`` mapping or an iterable of
    ``"section.key=value"`` strings (see :func:`parse_override`); missing
    intermediate tables are created, so ``training.lr_groups.head=0.1`` works.
    """
    if not overrides:
        return config
    if not isinstance(overrides, Mapping):
        overrides = dict(parse_override(spec) for spec in overrides)
    for dotted, value in overrides.items():
        *parents, leaf = dotted.split(".")
        table = config
        for name in parents:
            child = table.get(name)
            if not isinstance(child, dict):
                child = table[name] = {}
            table = child
        table[leaf] = value
    return config
