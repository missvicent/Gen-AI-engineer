#!/usr/bin/env python3
"""Write workshops/assistant/phases from layers.toml.

Students do not run this. It turns one package into one folder per workshop:
finished code for what earlier workshops already taught, stubs for the files
this workshop owns, and only the tests those files can answer.

  python3 build_layers.py          # write phases/
  python3 build_layers.py --check  # exit 1 if phases/ would change
"""

from __future__ import annotations

import ast
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ASSISTANT = ROOT.parent
SRC = ROOT / "before" / "src" / "assistant"
AFTER = ASSISTANT / "after" / "src" / "assistant"
TESTS = ASSISTANT / "after" / "tests"
LAYERS = ASSISTANT / "phases"
RUFF = ASSISTANT / "after" / ".venv" / "bin" / "ruff"
INCLUDE = "../../../../../_lesson.mk"


def load() -> dict:
    return tomllib.loads((ROOT / "layers.toml").read_text())


def module_name(path: Path) -> str:
    return path.name


def imports_of(path: Path) -> set[str]:
    """Assistant modules this file imports at module level."""
    tree = ast.parse(path.read_text())
    found: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "assistant":
                found.update(alias.name for alias in node.names)
            elif node.module.startswith("assistant."):
                found.add(node.module.split(".", 2)[1])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "assistant":
                    continue
                if alias.name.startswith("assistant."):
                    found.add(alias.name.split(".", 2)[1])
    return {
        name
        for name in found
        if (AFTER / f"{name}.py").is_file() or (SRC / f"{name}.py").is_file()
    }


def owner_map(layers: list[dict]) -> dict[str, str]:
    owned: dict[str, str] = {}
    for layer in layers:
        for name in layer.get("edit", []):
            if name in owned:
                raise SystemExit(
                    f"{name} is owned by both {owned[name]} and {layer['id']}"
                )
            owned[name] = layer["id"]
    return owned


def version_for(name: str, layer: dict, owned: dict[str, str], side: str) -> Path:
    """Which file lands in this layer: the stub, or the finished module."""
    here = layer["id"]
    if name in layer.get("edit", []) and side == "before":
        return SRC / name
    if name in layer.get("supply", []):
        return AFTER / name
    owner = owned.get(name)
    if owner == here and side == "before":
        return SRC / name
    # Earlier and later owners, and anything shipped finished, are the reference.
    # A later stub inside this folder is how a student starts chasing the wrong file.
    return AFTER / name


def as_file(name: str) -> str:
    return name if name.endswith(".py") else f"{name}.py"


def closure(seeds: set[str], layer: dict, owned: dict[str, str], side: str) -> set[str]:
    seen: set[str] = set()
    stack = [as_file(name) for name in seeds]
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        if not (AFTER / name).is_file() and not (SRC / name).is_file():
            continue
        seen.add(name)
        path = version_for(name, layer, owned, side)
        for dep in imports_of(path):
            dep_file = as_file(dep)
            if dep_file not in seen:
                stack.append(dep_file)
    return seen


def used_names(nodes: list[ast.AST]) -> set[str]:
    found: set[str] = set()
    for node in nodes:
        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                found.add(child.id)
            elif isinstance(child, ast.Attribute) and isinstance(child.value, ast.Name):
                found.add(child.value.id)
    return found


def snippet(source: str, node: ast.AST) -> str:
    """Original lines, so a slice stays formatted the way the suite already is.

    A decorator sits above `def` and is not included in the function's own
    lineno. Dropping it turns a parametrized test into a missing fixture.
    """
    lines = source.splitlines()
    decos = getattr(node, "decorator_list", None) or []
    start = decos[0].lineno if decos else getattr(node, "lineno", None)
    end = getattr(node, "end_lineno", None)
    if start is None or end is None:
        return ""
    return "\n".join(lines[start - 1 : end])


def slice_tests(source: str, keep: set[str] | None, drop: set[str]) -> str:
    """Keep the named tests plus the helpers and imports they use."""
    tree = ast.parse(source)
    functions: dict[str, ast.AST] = {}
    assigns: dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions[node.name] = node
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assigns[target.id] = node
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            assigns[node.target.id] = node

    wanted: set[str] = set()
    for name in functions:
        if not name.startswith("test_"):
            continue
        if keep is not None and name not in keep:
            continue
        if name in drop:
            continue
        wanted.add(name)

    changed = True
    while changed:
        changed = False
        names = used_names([functions[n] for n in wanted if n in functions])
        for name in list(functions):
            if name in wanted or name.startswith("test_"):
                continue
            if name in names:
                wanted.add(name)
                changed = True
        for name in assigns:
            if name in names and name not in wanted:
                wanted.add(name)
                changed = True

    body_names = used_names(
        [functions[n] for n in wanted if n in functions]
        + [assigns[n] for n in wanted if n in assigns]
    )
    doc = ""
    imports: list[str] = []
    body: list[str] = []
    if (
        tree.body
        and isinstance(tree.body[0], ast.Expr)
        and isinstance(tree.body[0].value, ast.Constant)
    ):
        doc = snippet(source, tree.body[0])

    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            imports.append(snippet(source, node))
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            bound = [alias.asname or alias.name.split(".")[0] for alias in node.names]
            if any(name in body_names for name in bound):
                imports.append(snippet(source, node))
            continue
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in wanted
        ):
            body.append(snippet(source, node))
            continue
        if isinstance(node, ast.Assign):
            ids = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if any(i in wanted for i in ids):
                body.append(snippet(source, node))
            continue
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id in wanted
        ):
            body.append(snippet(source, node))

    if not any(name.startswith("test_") and name in wanted for name in functions):
        raise SystemExit("a slice kept no tests")
    chunks = [part for part in (doc, "\n".join(imports), "\n\n".join(body)) if part]
    return "\n\n".join(chunks) + "\n"


def render_slice(spec: dict) -> str:
    text = (TESTS / spec["source"]).read_text()
    if (
        "keep" not in spec
        and "drop" not in spec
        and spec.get("dest", spec["source"]) == spec["source"]
    ):
        return text
    keep = set(spec["keep"]) if "keep" in spec else None
    drop = set(spec.get("drop", []))
    if keep is None and not drop:
        return text
    return slice_tests(text, keep, drop)


def pyproject(layer_id: str, side: str) -> str:
    text = (ASSISTANT / "after" / "pyproject.toml").read_text()
    return text.replace(
        'name = "assistant-after"', f'name = "assistant-{layer_id}-{side}"', 1
    )


def makefile(previous: list[str]) -> str:
    lines = [
        f"include {INCLUDE}",
        "",
        ".PHONY: adopt-mine",
        "adopt-mine:  ## copy the files you finished in the previous layer",
    ]
    if not previous:
        lines.append(
            '\t@echo "This is the first layer. There is nothing to copy." && exit 1'
        )
    else:
        lines.append(
            '\t@test -n "$(FROM)" || { echo "usage: make adopt-mine FROM=../<previous>/before"; exit 1; }'
        )
        for name in previous:
            lines.append(f'\tcp "$(FROM)/src/assistant/{name}" src/assistant/{name}')
    lines.append("")
    return "\n".join(lines)


def readme(layer: dict, previous_id: str | None) -> str:
    edit = ", ".join(f"`src/assistant/{name}`" for name in layer["edit"])
    supply = layer.get("supply", [])
    supply_line = (
        "Supplied, already finished: " + ", ".join(f"`{name}`" for name in supply) + "."
        if supply
        else "Nothing else in this folder is yours to write."
    )
    carry = ""
    if previous_id:
        carry = f"""
## Carry your work forward

From this folder, once the previous layer is green:

```bash
make adopt-mine FROM=../../{previous_id}/before
```

That copies only the files the previous layer asked you to write.
"""
    hints = "\n".join(f"{i}. {hint}" for i, hint in enumerate(layer["hints"], 1))
    return f"""# {layer["title"]}

Edit {edit}. {supply_line}

Files that belong to a later workshop are not in this folder, or they are already
finished because this one imports them. If a test fails in a file that is not
listed above, you are in the wrong folder.

## Do this

```bash
make setup
{layer["command"]}
make test
```

`make test` runs every test in this folder, including the ones earlier workshops
already made green.

## What the first failure means

`{layer["first_failure"]}` — {layer["meaning"]}

## Done when

- [ ] `make test` exits 0.
- [ ] The failure you started from is gone, and no new failure appeared in a file you do not own.

## Stuck?

{hints}

Diff only the file you were asked to edit, against this layer's solution:

```bash
diff -u src/assistant/{layer["edit"][0]} ../after/src/assistant/{layer["edit"][0]}
```
{carry}"""


def seeds_from_tests(texts: list[str]) -> set[str]:
    found: set[str] = set()
    for text in texts:
        for node in ast.parse(text).body:
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module == "assistant":
                    found.update(alias.name for alias in node.names)
                elif node.module.startswith("assistant."):
                    found.add(node.module.split(".", 2)[1])
    return found


def layer_slices(layer: dict) -> list[tuple[str, str]]:
    rendered = []
    for spec in layer.get("slice", []):
        dest = spec.get("dest", spec["source"])
        rendered.append((dest, render_slice(spec)))
    return rendered


def layer_files(
    layer: dict,
    owned: dict[str, str],
    own_tests: list[tuple[str, str]],
    earlier: list[tuple[str, str]],
) -> dict[str, str]:
    """Relative path -> text, for both sides. Tests are identical on both sides."""
    files: dict[str, str] = {}
    tests = own_tests + [
        (name, text) for name, text in earlier if name not in {n for n, _ in own_tests}
    ]
    seeds = set(layer.get("edit", [])) | set(layer.get("supply", []))
    seeds |= seeds_from_tests([text for _, text in tests])
    if layer.get("mode") == "full":
        modules = {p.name for p in AFTER.glob("*.py")}
    else:
        modules = closure(seeds, layer, owned, "before") | closure(
            seeds, layer, owned, "after"
        )
    for side in ("before", "after"):
        base = f"{layer['id']}/{side}"
        for name in sorted(modules):
            path = version_for(name, layer, owned, side)
            files[f"{base}/src/assistant/{name}"] = path.read_text()
        init = AFTER / "__init__.py"
        if init.is_file():
            files[f"{base}/src/assistant/__init__.py"] = init.read_text()
        for dest, text in tests:
            files[f"{base}/tests/{dest}"] = text
        files[f"{base}/pyproject.toml"] = pyproject(layer["id"], side)
        files[f"{base}/README.md"] = (
            readme(layer, None)
            if side == "before"
            else f"# {layer['title']} — reference\n\nThis is the finished layer. The exercise is `../before`.\n"
        )
    return files


def makefiles(layers: list[dict]) -> dict[str, str]:
    files: dict[str, str] = {}
    previous: list[str] = []
    for layer in layers:
        text = makefile(previous)
        for side in ("before", "after"):
            files[f"{layer['id']}/{side}/Makefile"] = text
        previous = list(layer.get("edit", []))
    return files


def readmes(layers: list[dict]) -> dict[str, str]:
    files: dict[str, str] = {}
    prev_id: str | None = None
    for layer in layers:
        files[f"{layer['id']}/before/README.md"] = readme(layer, prev_id)
        prev_id = layer["id"]
    return files


def index_page(layers: list[dict]) -> str:
    rows = [
        "| Phase | You edit | First test |",
        "|-------|----------|------------|",
    ]
    for layer in layers:
        edit = ", ".join(f"`{name}`" for name in layer["edit"])
        rows.append(
            f"| [`{layer['id']}`]({layer['id']}/before/) | {edit} | `{layer['first_failure']}` |"
        )
    body = "\n".join(rows)
    return f"""# Workshop phases

Open one folder. Edit the files it names. Run `make test` there.

`workshops/assistant/generator` writes these folders. It is not where you work.
`workshops/assistant/after` is the finished assistant the compose file builds.
Do not rename it.

{body}

The defect lab is not another copy of the service. It runs in
[`../after`](../after/) — see [`09-defect-lab/WORKSHOP-DEFECT-LAB.md`](09-defect-lab/WORKSHOP-DEFECT-LAB.md).
"""


def defect_readme() -> str:
    return """# Workshop 9 — defect lab

This lab does not have its own copy of the service. The service it breaks is
the finished one in `workshops/assistant/after`, because that is the tree the
Dockerfile builds.

## Do this

```bash
cd ../../after
mv defects/test_regressions.py defects/test_regressions.reference.py
cp ../phases/09-defect-lab/test_regressions.py defects/test_regressions.py
make defect-lab
```

The first run is not green. That is the start. Do not edit `variants.py`.
When you are stuck, diff your tests against `defects/test_regressions.reference.py`.
"""


def render() -> dict[str, str]:
    spec = load()
    layers = spec["layer"]
    owned = owner_map(layers)
    files: dict[str, str] = {}
    carried: list[tuple[str, str]] = []
    for layer in layers:
        own = layer_slices(layer)
        files.update(layer_files(layer, owned, own, carried))
        names = {name for name, _ in own}
        carried = own + [(name, text) for name, text in carried if name not in names]
    files.update(makefiles(layers))
    files.update(readmes(layers))
    files["README.md"] = index_page(layers)
    files["09-defect-lab/README.md"] = defect_readme()
    starter = ROOT / "before" / "defects" / "test_regressions.py"
    files["09-defect-lab/test_regressions.py"] = starter.read_text()
    return files


def format_tests(root: Path) -> None:
    """Drop imports a slice no longer uses. Do not reformat: the lines are the originals."""
    if not RUFF.is_file():
        return
    tests = [str(p) for p in root.rglob("tests/*.py")]
    if not tests:
        return
    subprocess.run(
        [
            str(RUFF),
            "check",
            "--fix",
            "--quiet",
            "--no-cache",
            "--select",
            "F401,I001",
            *tests,
        ],
        check=False,
    )


def is_brief(path: Path) -> bool:
    """Workshop briefs live beside a layer and are not generated."""
    return path.name.startswith("WORKSHOP-") and path.suffix == ".md"


def write(files: dict[str, str], root: Path) -> None:
    preserved: dict[str, str] = {}
    if root == LAYERS and root.exists():
        for path in root.rglob("WORKSHOP-*.md"):
            if path.is_file():
                preserved[str(path.relative_to(root))] = path.read_text()
        shutil.rmtree(root)
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    for rel, text in preserved.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    format_tests(root)


def check(files: dict[str, str]) -> int:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        rendered = Path(tmp)
        write(files, rendered)

        def tracked(root: Path) -> dict[str, str]:
            found = {}
            for path in root.rglob("*"):
                if not path.is_file():
                    continue
                if any(
                    part in {".venv", "__pycache__", ".ruff_cache", ".pytest_cache"}
                    for part in path.parts
                ):
                    continue
                if is_brief(path):
                    continue
                found[str(path.relative_to(root))] = path.read_text(errors="replace")
            return found

        expected = tracked(rendered)
        actual = tracked(LAYERS)
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        changed = sorted(
            p for p in set(expected) & set(actual) if expected[p] != actual[p]
        )
        if not missing and not extra and not changed:
            print("phases: up to date")
            return 0
        print(
            f"phases: drift ({len(missing)} missing, {len(extra)} extra, {len(changed)} changed)"
        )
        for name in (missing + extra + changed)[:30]:
            print(f"  {name}")
        return 1


def main() -> int:
    files = render()
    if "--check" in sys.argv:
        return check(files)
    write(files, LAYERS)
    print(f"wrote {len(files)} files under {LAYERS.relative_to(ASSISTANT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
