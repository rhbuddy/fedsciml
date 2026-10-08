"""Structured run storage: ``results/<problem>/<aggregator>/<run_id>/``.

Artifacts (SRS v1.4 §8.2 + FR-STORE-1..5, FR-SERVERAPP-6, backward-compat):
  - ``config.json``              run configuration (+ run_id, created_at)
  - ``config.yaml``              exact YAML snapshot (reproducibility FR-CFG-6)
  - ``metrics.jsonl``            one JSON record per completed round
  - ``metrics.json``             aggregated summary (final)
  - ``final_model.safetensors``  final global weights (wire format)
  - ``server_final.pth``        torch.save final weights (SRS §8.2)
  - ``server_best.pth``         best (lowest L2) weights
  - ``loss.npz``                 training loss curve
  - ``l2_error.npz``             L2 error over rounds
  - ``weight_divergence.npz``    per-layer divergence vs initial/central proxy
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml

from .weights import save_state_dict


class RunStorage:
    def __init__(self, root: str | Path, problem: str, aggregator: str) -> None:
        # Include a short random suffix so two runs started in the same second do
        # not collide and append metrics into the same directory.
        run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
        self.root = Path(root)
        self.dir = self.root / problem / aggregator / run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.dir / "metrics.jsonl"
        # best tracking for server_best.pth
        self._best_l2: float | None = None
        self._best_state: Optional[Dict[str, torch.Tensor]] = None
        self._l2_history: List[float] = []
        self._loss_history: List[float] = []
        self._wd_history: List[Dict[str, float]] = []
        self._initial_state: Optional[Dict[str, torch.Tensor]] = None

    @property
    def run_id(self) -> str:
        return self.dir.name

    def write_config(self, config: Dict[str, Any]) -> None:
        payload = dict(config)
        payload["run_id"] = self.run_id
        payload["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        (self.dir / "config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        # SRS FR-CFG-6: save exact YAML snapshot
        try:
            # preserve original yaml if provided
            if "_raw_yaml" in payload:
                (self.dir / "config.yaml").write_text(payload["_raw_yaml"], encoding="utf-8")
            else:
                # dump json-like config as yaml for reproducibility
                safe = {k: v for k, v in payload.items() if k != "_raw_yaml"}
                (self.dir / "config.yaml").write_text(yaml.safe_dump(safe, sort_keys=False), encoding="utf-8")
        except Exception:
            pass

    def set_initial_state(self, state: Dict[str, torch.Tensor]) -> None:
        if self._initial_state is None:
            self._initial_state = {k: v.detach().cpu().clone() for k, v in state.items()}

    def log_round(self, record: Dict[str, Any]) -> None:
        with self.metrics_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        # track histories for SRS artifacts
        try:
            if "l2_relative_error" in record and record["l2_relative_error"] is not None:
                v = float(record["l2_relative_error"])
                if np.isfinite(v):
                    self._l2_history.append(v)
            if "mean_local_loss" in record and record["mean_local_loss"] is not None:
                self._loss_history.append(float(record["mean_local_loss"]))
            if "weight_divergence" in record and isinstance(record["weight_divergence"], dict):
                self._wd_history.append(dict(record["weight_divergence"]))
        except Exception:
            pass

    def save_model(self, state_dict, is_best: bool = False) -> None:
        # Always save final via safetensors (wire compat)
        save_state_dict(state_dict, self.dir / "final_model.safetensors")
        # SRS: torch.save variants
        try:
            torch.save(state_dict, self.dir / "server_final.pth")
        except Exception:
            pass
        # Update best tracking externally via notify_best

    def notify_best(self, l2: float, state_dict: Dict[str, torch.Tensor]) -> None:
        """Track best model (lowest L2) for server_best.pth."""
        try:
            if l2 is None or not np.isfinite(l2):
                return
            if self._best_l2 is None or float(l2) < self._best_l2:
                self._best_l2 = float(l2)
                self._best_state = {k: v.detach().cpu().clone() for k, v in state_dict.items()}
        except Exception:
            pass

    def finalize(self, final_state: Dict[str, torch.Tensor], extra_metrics: Optional[Dict[str, Any]] = None) -> None:
        """Write SRS artifacts at run completion: metrics.json, npz files, best model."""
        # Save best
        if self._best_state is not None:
            try:
                save_state_dict(self._best_state, self.dir / "server_best.safetensors")
                torch.save(self._best_state, self.dir / "server_best.pth")
            except Exception:
                pass
        else:
            # no best tracked, duplicate final as best
            try:
                save_state_dict(final_state, self.dir / "server_best.safetensors")
                torch.save(final_state, self.dir / "server_best.pth")
            except Exception:
                pass
        # loss.npz
        try:
            np.savez_compressed(self.dir / "loss.npz", loss=np.array(self._loss_history, dtype=np.float64))
        except Exception:
            pass
        try:
            np.savez_compressed(self.dir / "l2_error.npz", l2=np.array(self._l2_history, dtype=np.float64))
        except Exception:
            pass
        try:
            # weight_divergence.npz: save per-round mean divergence plus per-layer stacks
            if self._wd_history:
                # collect keys
                keys = set()
                for d in self._wd_history:
                    keys.update(d.keys())
                # remove summary keys for per-layer array? keep all
                wd_arr = {}
                for k in keys:
                    vals = [float(d.get(k, np.nan)) for d in self._wd_history]
                    wd_arr[k] = np.array(vals, dtype=np.float64)
                np.savez_compressed(self.dir / "weight_divergence.npz", **wd_arr)
            else:
                np.savez_compressed(self.dir / "weight_divergence.npz", mean=np.array([], dtype=np.float64))
        except Exception:
            pass
        # metrics.json (summary)
        try:
            summary: Dict[str, Any] = {
                "run_id": self.run_id,
                "best_l2": self._best_l2,
                "final_l2": self._l2_history[-1] if self._l2_history else None,
                "l2_history": self._l2_history,
                "loss_history": self._loss_history,
                "total_rounds": len(self._l2_history),
            }
            if extra_metrics:
                summary.update(extra_metrics)
            (self.dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        except Exception:
            pass

    def read_metrics(self) -> List[Dict[str, Any]]:
        if not self.metrics_path.exists():
            return []
        records: List[Dict[str, Any]] = []
        for line in self.metrics_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
        return records
