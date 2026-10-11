"""YAML experiment config loader, validator, sweep expansion (SRS FR-CFG, FR-RUN).

Supports:
  - single run:  configs/poisson_fedavg.yaml
  - sweeps: lists of values produce Cartesian product
  - validation-gate command
"""

from __future__ import annotations

import copy
import itertools
import json
from pathlib import Path
from typing import Any, Dict, List

import yaml

VALID_AGGREGATORS = [
    "fedavg",
    "fedprox",
    "fedadam",
    "fedadagrad",
    "fedyogi",
    "scaffold",
    "feddyn",
    "median",
    "trimmed_mean",
    "krum",
]
VALID_WEIGHTINGS = ["uniform", "data_size", "quality"]
VALID_NOISE = ["none", "noisy", "adversarial"]
VALID_CLIP = ["none", "value", "norm"]
VALID_COMPRESSION = ["none", "int8", "topk", "int8_topk"]
VALID_PROBLEMS_PREFIX = None  # validated against registry lazily


def _ensure_list(v: Any) -> List[Any]:
    if isinstance(v, list):
        return v
    return [v]


def load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Config {p} must be a mapping, got {type(data)}")
    return data


def validate_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a single concrete config (no sweep lists)."""
    cfg = dict(cfg)
    # required
    if "problem" not in cfg:
        raise ValueError("Config missing required field 'problem'")
    # aggregator
    agg = str(cfg.get("aggregator", "fedavg")).lower().strip()
    if agg not in VALID_AGGREGATORS and agg not in ("fedagrad",):
        raise ValueError(f"Unknown aggregator '{agg}'. Options: {VALID_AGGREGATORS}")
    # weighting
    w = str(cfg.get("aggregation_weights", cfg.get("weighting", "data_size"))).lower().strip()
    if w not in VALID_WEIGHTINGS:
        raise ValueError(f"Unknown weighting '{w}'. Options: {VALID_WEIGHTINGS}")
    # K
    n_clients = cfg.get("n_clients", cfg.get("K", 2))
    if n_clients is not None:
        n = int(n_clients)
        if n < 2:
            raise ValueError("n_clients must be >=2 (SRS FR-AGG-1)")
        cfg["n_clients"] = n
    # rounds
    for k in ("global_rounds", "total_rounds"):
        if k in cfg and int(cfg[k]) < 1:
            raise ValueError(f"{k} must be >=1")
    if "global_rounds" in cfg and "total_rounds" not in cfg:
        cfg["total_rounds"] = int(cfg["global_rounds"])
    if "total_rounds" in cfg and "global_rounds" not in cfg:
        cfg["global_rounds"] = int(cfg["total_rounds"])
    # noise
    noise_mode = str(cfg.get("noise_mode", "none")).lower().strip()
    if noise_mode not in VALID_NOISE:
        raise ValueError(f"Unknown noise_mode '{noise_mode}'. Options: {VALID_NOISE}")
    nf = float(cfg.get("noise_fraction", cfg.get("noise_frac", 0.0)))
    if not (0.0 <= nf <= 1.0):
        raise ValueError("noise_fraction must be in [0,1]")
    cfg["noise_fraction"] = nf
    cfg["noise_mode"] = noise_mode
    # clip
    clip = str(cfg.get("gradient_clip", "none")).lower().strip()
    if clip not in VALID_CLIP:
        raise ValueError(f"Unknown gradient_clip '{clip}'. Options: {VALID_CLIP}")
    cfg["gradient_clip"] = clip
    # heterogeneity
    if "heterogeneity" in cfg and cfg["heterogeneity"] is not None and not isinstance(cfg["heterogeneity"], dict):
        raise ValueError("heterogeneity must be a mapping e.g. {mode: 1d_partition, n_pieces: 10}")
    # model_family (SRS FR-CFG-2) — optional, defaults per problem; validated if provided
    if "model_family" in cfg and cfg["model_family"] is not None:
        mf = str(cfg["model_family"]).lower().strip()
        if mf not in ("fnn", "mlp", "deeponet", "operator", "pinn"):
            raise ValueError(f"Unknown model_family '{mf}'. Options: fnn, deeponet (or pinn/operator aliases)")
        cfg["model_family"] = mf
    # learning rate
    if "learning_rate" in cfg:
        lr = float(cfg["learning_rate"])
        if not (0 < lr < 10):
            raise ValueError("learning_rate out of range (0,10)")
    # seed (support alias 'seeds' per Appendix A.4)
    if "seeds" in cfg and "seed" not in cfg:
        cfg["seed"] = cfg.pop("seeds")
    if "seed" in cfg and cfg["seed"] is not None:
        # if list (sweep), leave as list for expansion; validate will handle single
        if not isinstance(cfg["seed"], list):
            cfg["seed"] = int(cfg["seed"])
    # aggregator params
    if "aggregator_params" in cfg and not isinstance(cfg["aggregator_params"], dict):
        raise ValueError("aggregator_params must be a mapping")
    # Framework v2: sampling, compression, DP
    cf = cfg.get("client_fraction", cfg.get("C", 1.0))
    if cf is not None:
        cf = float(cf)
        if not (0 < cf <= 1.0):
            raise ValueError("client_fraction must be in (0,1]")
        cfg["client_fraction"] = cf
    if "round_timeout" in cfg and cfg["round_timeout"] is not None:
        rt = int(cfg["round_timeout"])
        if rt < 0:
            raise ValueError("round_timeout must be >=0")
        cfg["round_timeout"] = rt
    comp = str(cfg.get("compression", "none")).lower().strip()
    if comp not in VALID_COMPRESSION:
        raise ValueError(f"Unknown compression '{comp}'. Options: {VALID_COMPRESSION}")
    cfg["compression"] = comp
    if "topk_ratio" in cfg:
        tr = float(cfg["topk_ratio"])
        if not (0 < tr <= 1.0):
            raise ValueError("topk_ratio must be in (0,1]")
        cfg["topk_ratio"] = tr
    if "dp_noise_multiplier" in cfg:
        dpm = float(cfg["dp_noise_multiplier"])
        if dpm < 0:
            raise ValueError("dp_noise_multiplier must be >=0")
        cfg["dp_noise_multiplier"] = dpm
        # DP requires norm clipping
        if dpm > 0 and clip != "norm":
            raise ValueError("DP (dp_noise_multiplier>0) requires gradient_clip='norm'")
    if "dp_delta" in cfg:
        dd = float(cfg["dp_delta"])
        if not (0 < dd < 1):
            raise ValueError("dp_delta must be in (0,1)")
        cfg["dp_delta"] = dd
    return cfg


def expand_sweeps(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Expand sweep syntax: any list-valued leaf produces Cartesian product.

    Special key 'sweep' (mapping of param->list) is also supported per Appendix A.4.
    Returns list of concrete validated configs.
    """
    # Handle top-level 'sweep' mapping
    if "sweep" in cfg and isinstance(cfg["sweep"], dict):
        sweep_map = cfg["sweep"]
        base = {k: v for k, v in cfg.items() if k != "sweep"}
        keys = list(sweep_map.keys())
        values = [_ensure_list(sweep_map[k]) for k in keys]
        out: List[Dict[str, Any]] = []
        for combo in itertools.product(*values):
            concrete = dict(base)
            for k, v in zip(keys, combo):
                # handle nested heterogeneity etc.
                concrete[k] = v
            out.append(validate_config(concrete))
        return out

    # Otherwise expand any list-valued fields as sweep dimensions
    sweep_keys = [k for k, v in cfg.items() if isinstance(v, list)]
    if not sweep_keys:
        return [validate_config(cfg)]
    # Cartesian product
    combos = itertools.product(*[cfg[k] for k in sweep_keys])
    configs: List[Dict[str, Any]] = []
    for combo in combos:
        concrete = dict(cfg)
        for k, v in zip(sweep_keys, combo):
            concrete[k] = v
        configs.append(validate_config(concrete))
    return configs


def config_to_run_start(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Translate validated YAML config to RunStartRequest dict."""
    # Normalize keys
    problem = cfg.get("problem") or cfg.get("problem_name") or "poisson"
    aggregator = str(cfg.get("aggregator", "fedavg")).lower().strip()
    if aggregator == "fedagrad":
        aggregator = "fedadagrad"
    weighting = str(cfg.get("aggregation_weights", cfg.get("weighting", cfg.get("aggregation_weights", "data_size")))).lower().strip()
    total_rounds = int(cfg.get("total_rounds", cfg.get("global_rounds", 10)))
    local_epochs = int(cfg.get("local_epochs", 5))
    lr = float(cfg.get("learning_rate", cfg.get("lr", 1e-3)))
    agg_params = dict(cfg.get("aggregator_params") or {})
    # promote top-level mu/server_lr etc into aggregator_params
    for k in ("mu", "alpha", "beta", "server_lr", "beta1", "beta2", "tau", "trim_ratio", "n_byzantine", "multi_k", "local_epochs"):
        if k in cfg:
            agg_params[k] = cfg[k]
    heterogeneity = cfg.get("heterogeneity")
    model_family = cfg.get("model_family")
    noise_mode = str(cfg.get("noise_mode", "none"))
    noise_fraction = float(cfg.get("noise_fraction", 0.0))
    gradient_clip = str(cfg.get("gradient_clip", "none"))
    max_norm = float(cfg.get("max_norm", 1.0))
    clip_value = float(cfg.get("clip_value", 0.5))
    optimizer = str(cfg.get("optimizer", "adam"))
    client_fraction = float(cfg.get("client_fraction", cfg.get("C", 1.0)) or 1.0)
    round_timeout = int(cfg.get("round_timeout", 0) or 0)
    compression = str(cfg.get("compression", "none")).lower().strip()
    topk_ratio = float(cfg.get("topk_ratio", 0.01) or 0.01)
    dp_noise_multiplier = float(cfg.get("dp_noise_multiplier", 0.0) or 0.0)
    dp_delta = float(cfg.get("dp_delta", 1e-5) or 1e-5)
    return {
        "problem": problem,
        "aggregator": aggregator,
        "total_rounds": total_rounds,
        "local_epochs": local_epochs,
        "learning_rate": lr,
        "weighting": weighting,
        "aggregator_params": agg_params,
        "heterogeneity": heterogeneity,
        "model_family": model_family,
        "noise_mode": noise_mode,
        "noise_fraction": noise_fraction,
        "gradient_clip": gradient_clip,
        "max_norm": max_norm,
        "clip_value": clip_value,
        "optimizer": optimizer,
        "seed": cfg.get("seed"),
        "n_clients": cfg.get("n_clients"),
        "client_fraction": client_fraction,
        "round_timeout": round_timeout,
        "compression": compression,
        "topk_ratio": topk_ratio,
        "dp_noise_multiplier": dp_noise_multiplier,
        "dp_delta": dp_delta,
    }
