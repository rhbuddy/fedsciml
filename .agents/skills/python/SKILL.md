---
name: python
description: >-
  Use this skill when the user asks to build, scaffold, modify, or debug a
  Python backend application or service. Covers project structure, dependency
  management (pip/uv/poetry), APIs (FastAPI, Flask, Django), async patterns,
  testing, packaging, and deployment.
---

# Python Backend Skill

## When to Use

Use this skill for any Python backend work: building REST/GraphQL APIs, data
pipelines, services, CLI tools, or scientific computing backends.

## Project Setup

### Modern Dependency Management
```bash
# uv (fast, recommended)
uv init --lib my-project
uv add fastapi uvicorn pydantic

# Or poetry
poetry new my-project
poetry add fastapi uvicorn pydantic

# Or pip + venv
python -m venv .venv
source .venv/bin/activate
pip install fastapi uvicorn pydantic
```

### Recommended Structure
```
my-project/
├── src/
│   ├── __init__.py
│   ├── main.py          # app entrypoint
│   ├── api/             # route handlers
│   │   ├── __init__.py
│   │   └── users.py
│   ├── services/        # business logic
│   ├── models/          # Pydantic / data models
│   └── core/            # config, db, deps
├── tests/
├── pyproject.toml
└── README.md
```

## Core Frameworks

### FastAPI (Recommended for new APIs)
```python
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel

app = FastAPI(title="My API", version="1.0.0")

class UserCreate(BaseModel):
    name: str
    email: str

@app.post("/users/", response_model=UserCreate, status_code=201)
async def create_user(user: UserCreate) -> UserCreate:
    # persist to DB
    return user
```

- Use `async def` for I/O-bound handlers.
- Use `Depends` for shared dependencies (DB sessions, auth).
- Auto-generated OpenAPI docs at `/docs` and `/redoc`.

### Flask (Microservices / small APIs)
```python
from flask import Flask, jsonify, request

app = Flask(__name__)

@app.get("/health")
def health():
    return jsonify(status="ok")
```

### Django (Full-featured, admin, ORM)
- Use for complex apps needing built-in admin, auth, ORM.
- Prefer `django-ninja` or `djangorestframework` for APIs.

## Data & Databases
- **SQL**: SQLAlchemy (`sqlalchemy` + `asyncpg` for async).
- **NoSQL**: `motor` (MongoDB async), `aiomysql`, `aiopg`.
- **ORM patterns**: Repository pattern or service layer; keep routes thin.

```python
from sqlalchemy.ext.asyncio import AsyncSession

async def get_session() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session
```

## Async & Concurrency
- Use `asyncio` for I/O-bound concurrency.
- Use `concurrent.futures.ProcessPoolExecutor` or `multiprocessing` for CPU-bound work.
- Avoid blocking calls in async handlers; use `asyncio.to_thread` or `run_in_executor`.

## Testing
```bash
pip install pytest pytest-asyncio httpx
```

```python
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}
```

## Packaging & Deployment
- Use `pyproject.toml` with PEP 621 metadata.
- Build: `python -m build` (produces sdist + wheel).
- Containerize with Docker:
  ```dockerfile
  FROM python:3.12-slim
  WORKDIR /app
  COPY . .
  RUN pip install --no-cache-dir -e .
  CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
  ```
- Run with `uvicorn main:app --reload` for dev, or via Gunicorn/Uvicorn in prod.

## Best Practices

1. Use **type hints** everywhere (PEP 484/585).
2. Use **Pydantic** for validation and settings (`pydantic-settings`).
3. Separate **config** from code (env vars, `.env`, `pydantic-settings`).
4. Write **tests** for services and endpoints; aim for coverage.
5. Use **logging** (`structlog` or stdlib `logging`) with JSON in prod.
6. Add **openapi/docs** and health check endpoints.
7. Pin dependency versions in production; use lock files.
