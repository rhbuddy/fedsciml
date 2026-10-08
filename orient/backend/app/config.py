"""Server settings, overridable with environment variables (prefix ``ORIENT_``)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class Settings:
    host: str = field(default_factory=lambda: os.environ.get("ORIENT_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.environ.get("ORIENT_PORT", "8000")))
    results_dir: str = field(default_factory=lambda: os.environ.get("ORIENT_RESULTS_DIR", "results"))
    seed: int = field(default_factory=lambda: int(os.environ.get("ORIENT_SEED", "0")))
    round_timeout_seconds: int = field(
        default_factory=lambda: int(os.environ.get("ORIENT_ROUND_TIMEOUT", "0"))
    )
    # URL that locally-spawned demo clients should connect to.
    public_url: str = field(
        default_factory=lambda: os.environ.get(
            "ORIENT_PUBLIC_URL",
            f"http://127.0.0.1:{os.environ.get('ORIENT_PORT', '8000')}",
        )
    )
    # Drop a client that has not polled within this many seconds (0 = never).
    client_ttl_seconds: int = field(
        default_factory=lambda: int(os.environ.get("ORIENT_CLIENT_TTL", "300"))
    )


settings = Settings()
