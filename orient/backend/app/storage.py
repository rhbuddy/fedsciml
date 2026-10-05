"""Structured run storage: ``results/<problem>/<aggregator>/<run_id>/``.

Artifacts (SRS v1.4 FR-SERVERAPP-6):
  - ``config.json``          the run configuration
  - ``metrics.jsonl``        one JSON record per completed round
  - ``final_model.safetensors``  final global weights
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .weights import save_state_dict


class RunStorage:
    def __init__(self, root: str | Path, problem: str, aggregator: str) -> None:
        run_id = time.strftime("%Y%m%d-%H%M%S")
        self.root = Path(root)
        self.dir = self.root / problem / aggregator / run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.dir / "metrics.jsonl"

    @property
    def run_id(self) -> str:
        return self.dir.name

    def write_config(self, config: Dict[str, Any]) -> None:
        payload = dict(config)
        payload["run_id"] = self.run_id
        payload["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        (self.dir / "config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def log_round(self, record: Dict[str, Any]) -> None:
        with self.metrics_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    def save_model(self, state_dict) -> None:
        save_state_dict(state_dict, self.dir / "final_model.safetensors")

    def read_metrics(self) -> List[Dict[str, Any]]:
        if not self.metrics_path.exists():
            return []
        records: List[Dict[str, Any]] = []
        for line in self.metrics_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
        return records
