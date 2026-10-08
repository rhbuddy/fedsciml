"""Static import-graph check (stdlib only, no runtime dependencies needed).

Parses every module in ``backend/app`` and ``client/app`` with :mod:`ast` and
verifies that each ``from .module import name`` reference actually resolves to a
top-level name defined in (or re-exported by) that module. Catches typos and
renames before the heavy scientific stack is installed.

Run directly::

    python tests/test_static_imports.py
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = [ROOT / "backend" / "app", ROOT / "client" / "app"]
# Entry points that import the client package as `app.*` (UI + examples)
ABSOLUTE_ENTRY_POINTS = [
    ROOT / "client" / "ui" / "app.py",
    ROOT / "client" / "examples" / "make_datasets.py",
]


def defined_names(path: Path) -> Set[str]:
    """Top-level names defined or imported by a module."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: Set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
    return names


def relative_imports(path: Path) -> List[Tuple[str, str]]:
    """Return ``(module, name)`` pairs for every ``from .module import name``."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    pairs: List[Tuple[str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
            for alias in node.names:
                pairs.append((node.module, alias.name))
    return pairs


def absolute_app_imports(path: Path) -> List[Tuple[str, str]]:
    """Return ``(module, name)`` pairs for every ``from app.module import name``."""
    if not path.exists():
        return []
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    pairs: List[Tuple[str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module == "app" or node.module.startswith("app."):
                parts = node.module.split(".")
                module = parts[1] if len(parts) > 1 else "__init__"
                for alias in node.names:
                    pairs.append((module, alias.name))
    return pairs


def main() -> int:
    problems: List[str] = []
    checked = 0

    for package in PACKAGES:
        modules = {p.stem: p for p in package.glob("*.py")}
        exports: Dict[str, Set[str]] = {stem: defined_names(p) for stem, p in modules.items()}

        for stem, path in sorted(modules.items()):
            label = f"{package.parent.name}/app/{stem}.py"
            for module, name in relative_imports(path):
                if module not in modules:
                    problems.append(f"{label}: relative import of unknown module '.{module}'")
                    continue
                if name not in exports[module]:
                    problems.append(
                        f"{label}: 'from .{module} import {name}' — '{name}' not defined in {module}.py"
                    )
                checked += 1

    # UI / examples import the client package as `app.*`
    client_modules = {p.stem: p for p in PACKAGES[1].glob("*.py")}
    client_exports: Dict[str, Set[str]] = {
        stem: defined_names(p) for stem, p in client_modules.items()
    }
    for entry in ABSOLUTE_ENTRY_POINTS:
        for module, name in absolute_app_imports(entry):
            if module not in client_modules:
                problems.append(f"{entry.name}: import of unknown module 'app.{module}'")
                continue
            if name not in client_exports[module]:
                problems.append(
                    f"{entry.name}: 'from app.{module} import {name}' — "
                    f"'{name}' not defined in {module}.py"
                )
            checked += 1

    print(f"Checked {checked} imports across {len(PACKAGES)} packages + entry points.")
    if problems:
        print(f"\n{len(problems)} PROBLEM(S):")
        for item in problems:
            print(f"  - {item}")
        return 1
    print("All relative imports resolve to defined names. OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
