"""Server entry point.

Run from the ``backend/`` directory::

    python -m app.main
    # or
    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import uvicorn

from .api import app
from .config import settings


def main() -> None:
    uvicorn.run(app, host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
