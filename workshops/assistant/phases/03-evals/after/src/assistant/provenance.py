"""Which system produced this number — derived, never typed.

Two consumers, one answer: `core.py` stamps the prompt version onto every
compose span, and `report.py` stamps model/prompt/corpus/dataset onto the report
the merge gate reads. They agree because they call the same functions.

The rule this file exists to enforce: **a version stamp is computed from the
thing it describes.** The alternative — `PROMPT_VERSION = "v3"` next to the
prompt — rots the first time someone edits the prompt in a hurry, and it rots
silently. Nothing fails, no test goes red; the label just stops being true, and
every comparison across that boundary quietly becomes a comparison between two
different systems wearing the same name.

Hashes are truncated to eight hex characters. That is 32 bits — plenty to notice
a change, not enough to be a cryptographic claim, and short enough that a person
can read one out in a review.
"""

from __future__ import annotations

import hashlib
import inspect
import os
import subprocess
from collections.abc import Iterable
from pathlib import Path

STAMP_LENGTH = 8
#: Git's conventional short form. Long enough to be unique in any repo a person
#: will work on, short enough to read out loud during an incident.
SHA_LENGTH = 12

#: A file that exists at the course source root and nowhere else above this one.
#: Shipped in the companion ZIP too (`git archive HEAD -- src`), so the extracted
#: lane finds the same landmark a checkout does.
ROOT_MARKER = "verify-lessons.sh"

#: Everything the full-fidelity release numbers are a measurement of, under `SRC`.
#:
#: Two entries because the red team is one dataset shared with the lesson that
#: maintains it, so a row added there changes what a containment number means
#: without touching a line of capstone code.
#:
#: Both are deliberately outside `release/evidence/`, where the published artifact
#: is committed. If the evidence lived inside what it measures, committing it would
#: change the answer, and no evidence could ever match the release carrying it.
MEASURED_SOURCE = (
    "workshops/assistant/after",
    "phase6-design-defend/01-red-team/after/evals/redteam.jsonl",
)

#: Everything a RELEASE is made of, as paths from the repository root — a wider
#: circle than `MEASURED_SOURCE`, which covers only what the numbers measure.
#:
#: This exists because the attestation used to carry a commit sha and the tag gate
#: required it to equal the commit being tagged. That is unsatisfiable by
#: construction: the run records the commit it measured, committing the record
#: produces a NEW commit, and the record now names its own parent. Every numeric
#: gate passed and publication still exited 1, forever.
#:
#: A tree object id is the fix, for exactly the reason `source_id` already uses one
#: a level down. `release/evidence/` is deliberately absent: it is the only place
#: generated evidence is committed (`src/workshops/assistant/*/evidence/` is
#: gitignored), so `HEAD:src` and `HEAD:app` cannot move when the evidence lands.
#: The binding is a fixed point rather than a race.
#:
#: `release/README.md` is named individually rather than taking `release/` whole,
#: because taking the directory would put the evidence back inside the thing it is
#: bound to and restore the circularity this replaced.
RELEASE_INPUTS = ("src", "app", "release/README.md")


def source_root() -> Path | None:
    """The course source root above this file, or `None` when there is not one.

    Searched for by landmark rather than counted to. The version this replaced was
    `Path(__file__).resolve().parents[5]`, which is right in the checkout — thirteen
    parents — and `IndexError` inside the image, where `COPY src/ src/` under
    `WORKDIR /app` leaves four. The API imports this module for `build_version`, so
    a source-root calculation that only the release lane needs crashed Uvicorn
    before startup and Docker restarted the container for as long as anyone let it.

    A fixed depth is an assumption about a layout with no way to check itself, and
    it fails where the code is hardest to run. A landmark either is there or is not.

    `None` is therefore a real answer, not a failure: inside the image there is no
    course tree, and callers are expected to say so — `source_id()` returns
    `unbound` — rather than to guess at a directory.
    """
    for parent in Path(__file__).resolve().parents:
        if (parent / ROOT_MARKER).is_file():
            return parent
    return None


def build_version() -> str:
    """Which commit is serving this request.

    Baked into the image at build time (`ARG GIT_SHA`) and read back from the
    environment, because by then there is no `.git` in the container — and the
    whole point is a stamp the running process can report about ITSELF.

    This is what a post-deploy smoke check compares against
    (`phase8-deploy/03-deploy-observe`). Without it the nastiest deploy failure
    is invisible: a rollout that half-finished, leaving an old machine in the
    pool. It is genuinely healthy, it answers correctly, and every check you
    have passes against it — because it is a working service, just not the one
    you shipped.

    Falls back to the local checkout so a developer's `/health` is honest too,
    and to `dev` outside a repo. It never raises: a version stamp that can take
    down the health endpoint is worse than no version stamp.
    """
    if stamped := os.getenv("GIT_SHA"):
        return stamped[:SHA_LENGTH]
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=2, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return "dev"
    return out.stdout.strip()[:SHA_LENGTH] if out.returncode == 0 else "dev"


def digest(*parts: str) -> str:
    """Content, hashed. Parts are NUL-joined so `["ab", "c"]` and `["a", "bc"]`
    are different inputs — a separator that cannot appear in text is the cheap
    way to avoid a stamp collision that would be very confusing to debug."""
    joined = "\u0000".join(parts).encode()
    return hashlib.sha256(joined).hexdigest()[:STAMP_LENGTH]


def prompt_version() -> str:
    """A stamp for the prompt the composers actually send.

    It hashes the SOURCE of `grounded_prompt`, which means the spans and the
    report both change the moment the prompt does — including the day someone
    changes it without thinking of themselves as changing anything."""
    from assistant import composers

    return f"grounded-{digest(inspect.getsource(composers.grounded_prompt))}"


def corpus_version(docs: Iterable[str]) -> str:
    """A stamp for a set of documents, count included so a human can read it."""
    items = [str(d) for d in docs]
    return f"corpus-{len(items)}-{digest(*items)}"


def dataset_version(label: str, questions: Iterable[str]) -> str:
    """A stamp for an eval set: its name, its size, and its content."""
    items = [str(q) for q in questions]
    return f"{label}-{len(items)}-{digest(*items)}"


def source_id() -> str:
    """A stable name for the exact code and data a measurement was taken against.

    The same rule as every other stamp here, applied one level up: `build_version`
    says which commit is *serving*, and this says which source was *measured*. The
    release evidence carried a date and a list of model versions, and a date cannot
    say whether numbers describe the code about to be published or the tree from a
    fortnight ago — so `release.yml` had nothing to check and published anyway.

    Git object ids rather than a commit sha, because the evidence has to be
    committed *to* the release it certifies: a sha changes when the evidence file
    lands, a tree hash of the measured paths does not. So the maintainer can run the
    measurement, commit the result, and tag — and the binding still holds.

    Four answers, all of them honest:

      ``<8 hex>``         a clean checkout; this is the measured source.
      ``dirty-<8 hex>``   a checkout with uncommitted changes under those paths.
                          Prefixed rather than reported as a separate flag: a gate
                          that compares ids cannot forget to also check a boolean,
                          and this id matches nothing.
      ``release-<sha12>`` unpacked from the ZIP, which has no git. `package.sh`
                          writes the commit into `src/RELEASE_COMMIT`, so the
                          measurement is still bound to something — the release it
                          came from rather than a tree.
      ``unbound``         no git and no stamp — including inside the image, where
                          there is no course tree at all. Says so, rather than
                          inventing a value that would compare equal to something.
    """
    if not _git("rev-parse", "--git-dir"):
        root = source_root()
        stamp = root / "RELEASE_COMMIT" if root else None
        if stamp and stamp.is_file() and (sha := stamp.read_text().strip()):
            return f"release-{sha[:SHA_LENGTH]}"
        return "unbound"

    ids = []
    for path in MEASURED_SOURCE:
        # `HEAD:<path>` is always read from the repo root, whatever the cwd.
        oid = _git("rev-parse", f"HEAD:src/{path}")
        if not oid:
            return "unbound"
        ids.append(f"{path}={oid}")

    # A pathspec, unlike a revision, IS resolved against the cwd — and the cwd here
    # is `src/`, so a plain `src/workshops/...` would look for `src/src/workshops/...`,
    # match nothing, and report a dirty tree as clean. `:/` pins it to the root.
    pending = _git("status", "--porcelain", "--", *(f":/src/{p}" for p in MEASURED_SOURCE))
    return f"dirty-{digest(*ids)}" if pending else digest(*ids)


def release_inputs_id() -> str:
    """A stable name for every input a release is built from.

    `source_id` answers "which code produced these numbers". This answers "which
    code is being published", and the two are different questions: the workbook
    under `app/` is in the release and in none of the measurements, and the compose
    stack and this verifier live under `src/` outside the capstone.

    The reason it is a separate function rather than a wider `MEASURED_SOURCE` is
    that widening the measured set would make the evidence claim things it never
    measured. So there are two bindings and the gate checks both.

    Same four answers as `source_id`, and for the same reasons — see `RELEASE_INPUTS`
    for why committing evidence cannot move this value.
    """
    if not _git("rev-parse", "--git-dir"):
        root = source_root()
        stamp = root / "RELEASE_COMMIT" if root else None
        if stamp and stamp.is_file() and (sha := stamp.read_text().strip()):
            return f"release-{sha[:SHA_LENGTH]}"
        return "unbound"

    ids = []
    for path in RELEASE_INPUTS:
        oid = _git("rev-parse", f"HEAD:{path}")
        if not oid:
            return "unbound"
        ids.append(f"{path}={oid}")

    pending = _git("status", "--porcelain", "--", *(f":/{p}" for p in RELEASE_INPUTS))
    return f"dirty-{digest(*ids)}" if pending else digest(*ids)


def _git(*args: str) -> str:
    """Git's answer, or "" when git cannot answer — including not being installed,
    and including having nowhere to ask from, which is the container's case."""
    root = source_root()
    if root is None:
        return ""
    try:
        done = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return done.stdout.strip() if done.returncode == 0 else ""
