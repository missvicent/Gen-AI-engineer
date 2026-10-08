#!/usr/bin/env python3
"""Checks the generated workshop layers.

  python3 verify_layers.py            # snapshots, owners, compose path
  python3 verify_layers.py --pytest   # also: the README's first failure is the first failure

The lesson verifier already runs each layer (after/ green, before/ red for the
right reason). This catches the drift that run cannot see.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import build_layers as build  # noqa: E402

ASSISTANT = ROOT.parent
COMPOSE = (
    ROOT.parents[2] / "phase8-deploy" / "01-compose" / "after" / "docker-compose.yml"
)
SHARED_PY = ASSISTANT / "after" / ".venv" / "bin" / "python"


def fail(problems: list[str], message: str) -> None:
    problems.append(message)


def check_owners(problems: list[str]) -> None:
    spec = build.load()
    owned: dict[str, str] = {}
    for layer in spec["layer"]:
        for name in layer.get("edit", []):
            owned[name] = layer["id"]
    shipped = set(spec.get("shipped_complete", []))
    for path in sorted((ROOT / "before" / "src" / "assistant").glob("*.py")):
        if "NotImplementedError" not in path.read_text():
            continue
        if path.name in shipped:
            continue
        if path.name not in owned:
            fail(
                problems, f"{path.name} raises NotImplementedError and no layer owns it"
            )


def check_snapshots(problems: list[str]) -> None:
    if build.check(build.render()) != 0:
        fail(problems, "phases/ does not match build_layers.py — regenerate it")


def check_identity(problems: list[str]) -> None:
    spec = build.load()
    for layer in spec["layer"]:
        base = ASSISTANT / "phases" / layer["id"]
        edit = set(layer.get("edit", []))
        before_src = base / "before" / "src" / "assistant"
        after_src = base / "after" / "src" / "assistant"
        if not before_src.is_dir():
            fail(problems, f"{layer['id']} has no before/src")
            continue
        names = {p.name for p in before_src.glob("*.py")} | {
            p.name for p in after_src.glob("*.py")
        }
        for name in sorted(names):
            b, a = before_src / name, after_src / name
            if not b.is_file() or not a.is_file():
                fail(problems, f"{layer['id']} is missing {name} on one side")
                continue
            same = b.read_bytes() == a.read_bytes()
            if name in edit and same:
                fail(
                    problems,
                    f"{layer['id']} asks you to edit {name}, but before and after match",
                )
            if name not in edit and not same:
                fail(
                    problems,
                    f"{layer['id']} changed {name}, which this layer does not own",
                )
        readme = (base / "before" / "README.md").read_text()
        if layer["first_failure"] not in readme:
            fail(
                problems, f"{layer['id']} README does not name {layer['first_failure']}"
            )


def check_compose(problems: list[str]) -> None:
    if not COMPOSE.is_file():
        fail(problems, f"compose file missing: {COMPOSE}")
        return
    text = COMPOSE.read_text()
    if "workshops/assistant/after" not in text:
        fail(problems, "compose context no longer builds workshops/assistant/after")


def first_failure(layer_id: str) -> str | None:
    before = ASSISTANT / "phases" / layer_id / "before"
    py = before / ".venv" / "bin" / "python"
    if not py.is_file():
        py = SHARED_PY
    if not py.is_file():
        return None
    proc = subprocess.run(
        [str(py), "-m", "pytest", "-x", "--tb=line", "-q", "-p", "no:cacheprovider"],
        cwd=before,
        env={
            **dict(**{k: v for k, v in __import__("os").environ.items()}),
            "PYTHONPATH": "src",
        },
        capture_output=True,
        text=True,
    )
    for line in (proc.stdout + proc.stderr).splitlines():
        if line.startswith(("FAILED ", "ERROR ")):
            return line
    return None


def check_pytest(problems: list[str]) -> None:
    for layer in build.load()["layer"]:
        line = first_failure(layer["id"])
        if line is None:
            fail(problems, f"{layer['id']}: could not run pytest (no venv)")
            continue
        if layer["first_failure"] not in line:
            fail(
                problems,
                f"{layer['id']}: first failure was {line.strip()} — README says {layer['first_failure']}",
            )


def main() -> int:
    problems: list[str] = []
    check_snapshots(problems)
    check_owners(problems)
    check_identity(problems)
    check_compose(problems)
    if "--pytest" in sys.argv:
        check_pytest(problems)
    if problems:
        print(f"phases: {len(problems)} problem(s)")
        for item in problems:
            print(f"  {item}")
        return 1
    print("phases: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
