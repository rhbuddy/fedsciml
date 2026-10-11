"""Server settings, overridable with environment variables (prefix ``ORIENT_``)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class Settings:
    host: str = field(default_factory=lambda: os.environ.get("ORIENT_HOST", "0.0.0.0"))  # nosec B104 - intentional for docker; override with ORIENT_HOST=127.0.0.1 for lockdown
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
    # Optional API key for admin endpoints (if set, clients must send X-API-Key header).
    # When empty, auth is disabled to keep local/dev and existing tests frictionless.
    api_key: str = field(default_factory=lambda: os.environ.get("ORIENT_API_KEY", ""))


settings = Settings()
