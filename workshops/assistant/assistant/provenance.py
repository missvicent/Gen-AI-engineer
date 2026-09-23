"""TODO: which system produced this number — derived, never typed.

Two consumers, one answer: `core.py` stamps the prompt version onto every
compose span, and `report.py` stamps model/prompt/corpus/dataset onto the report
the merge gate reads. They agree because they call the same functions.

The rule this file exists to enforce: **a version stamp is computed from the
thing it describes.** The alternative — `PROMPT_VERSION = "v3"` next to the
prompt — rots the first time someone edits the prompt in a hurry, and it rots
silently. Nothing fails, no test goes red; the label just stops being true, and
every comparison across that boundary quietly becomes a comparison between two
different systems wearing the same name.

Reference: ../../after/src/assistant/provenance.py.
"""
from __future__ import annotations

import hashlib
import subprocess
from collections.abc import Iterable
from pathlib import Path

STAMP_LENGTH = 8
#: Git's conventional short form.
SHA_LENGTH = 12

#: A file that exists at the course source root and nowhere else above this one.
#: Shipped in the companion ZIP too, so the extracted lane finds the same landmark
#: a checkout does.
ROOT_MARKER = "verify-lessons.sh"

#: Everything the full-fidelity release numbers are a measurement of, under the root.
#:
#: Two entries because the red team is one dataset shared with the lesson that
#: maintains it, so a row added there changes what a containment number means
#: without touching a line of capstone code.
MEASURED_SOURCE = (
    "workshops/assistant/after",
    "phase6-design-defend/01-red-team/after/evals/redteam.jsonl",
)

#: Everything a RELEASE is made of, as paths from the repository ROOT — a wider
#: circle than MEASURED_SOURCE, which covers only what the numbers measure.
#:
#: `release/evidence/` is deliberately absent, and that absence is the whole design.
#: It is the only place generated evidence is committed, so `HEAD:src` and `HEAD:app`
#: cannot move when evidence lands — which is what makes the binding below a fixed
#: point instead of the unsatisfiable commit-sha comparison it replaced.
#:
#: `release/README.md` is named individually rather than taking `release/` whole:
#: taking the directory would put the evidence back inside the thing it is bound to.
RELEASE_INPUTS = ("src", "app", "release/README.md")


def build_version() -> str:
    """TODO 1: which commit is serving this request.

    Read `GIT_SHA` from the environment (the Dockerfile bakes it in at build
    time — by then there is no `.git` in the container, and the whole point is a
    stamp the running process can report about ITSELF). Fall back to
    `git rev-parse HEAD` for a local checkout, then to `"dev"`.

    `/health` reports it and a post-deploy smoke check compares it to the SHA it
    just shipped. Without it the nastiest deploy failure is invisible: a rollout
    that half-finished, leaving an old machine in the pool. It is healthy, it
    answers correctly, and every other check passes against it.

    Never raise. A version stamp that can take down the health endpoint is worse
    than no version stamp — wrap the subprocess call and return `"dev"`.
    """
    raise NotImplementedError


def digest(*parts: str) -> str:
    """Eight hex characters of sha256 over the parts.

    Join with a separator that cannot appear in the text (`"\u0000"`) before
    hashing, so `["ab", "c"]` and `["a", "bc"]` are different inputs. Truncate to
    STAMP_LENGTH: 32 bits is plenty to notice a change, not a cryptographic
    claim, and short enough to read out in a review.
    """
    # Join parts with a null separator
    joined = "\u0000".join(parts)
    sha = hashlib.sha256(joined.encode("utf-8")).hexdigest()
    # Truncate to STAMP_LENGTH (8 hex characters = 32 bits)
    return sha[:8]


def prompt_version() -> str:
    """TODO 3: a stamp for the prompt the composers actually send.

    Hash `inspect.getsource(composers.grounded_prompt)` and return
    `f"grounded-{...}"`. Hashing the SOURCE is the point: the spans and the
    report both change the moment the prompt does — including the day someone
    changes it without thinking of themselves as changing anything.

    Import composers inside the function; core.py imports this module, and a
    module-level import would close the circle.
    """
    raise NotImplementedError


def corpus_version(docs: Iterable[str]) -> str:
    """TODO 4: a stamp for a set of documents — `corpus-<count>-<digest>`, with
    the count included so a human can read it."""
    raise NotImplementedError


def dataset_version(label: str, questions: Iterable[str]) -> str:
    """TODO 5: a stamp for an eval set: `<label>-<count>-<digest>`."""
    raise NotImplementedError


def source_id() -> str:
    """TODO 6: a stable name for the code and data a measurement was taken against.

    The same rule as every stamp above, one level up: `build_version` says which
    commit is *serving*, this says which source was *measured*. `release.yml` reads
    it out of the committed evidence and refuses to publish a tag whose tree answers
    differently — so a release cannot quote numbers taken against other code.

    Return four different things, and the distinctions are the exercise:

      ``<8 hex>``         `digest` over `"<path>=<oid>"` for each MEASURED_SOURCE
                          entry, where the oid is `git rev-parse HEAD:src/<path>`.
                          Hash the git object ids, NOT the commit sha — the evidence
                          has to be committed to the release it certifies, and a
                          commit sha changes when the evidence file lands while a
                          tree hash of the measured paths does not.
      ``dirty-<8 hex>``   when `git status --porcelain` reports anything under those
                          paths. Prefix the same digest rather than returning a
                          separate flag: a gate that compares ids cannot forget to
                          also check a boolean, and this id matches nothing.
                          Careful — a pathspec resolves against the cwd, so use the
                          `:/` root-relative form or a clean tree will read dirty.
      ``release-<sha12>`` no git, but `source_root() / "RELEASE_COMMIT"` exists —
                          the ZIP lane, where `package.sh` wrote the commit.
      ``unbound``         neither, including `source_root()` returning None. Never
                          invent a value here: an id that compares equal to a real
                          one is worse than admitting you have none.

    `_git` and `source_root` below are written for you; the judgement being tested
    is what to hash.
    """
    raise NotImplementedError


def release_inputs_id() -> str:
    """TODO 7: a stable name for every input a release is built from.

    Same four answers and the same construction as `source_id`, over RELEASE_INPUTS
    instead of MEASURED_SOURCE — note the paths are root-relative here, so the
    revision is `HEAD:<path>` with no `src/` prefix.

    Why a SECOND id rather than a wider first one: the two answer different
    questions. `source_id` says which code produced the numbers; this says which
    code is being published, and the release contains things no measurement touched
    (the workbook under `app/`, the compose stack, the verifier). Widening
    MEASURED_SOURCE to cover them would make the evidence claim it measured them.

    The attestation this feeds used to carry a commit sha, and the tag gate required
    it to equal the commit being tagged. Work out why that can never pass before you
    write this — the answer is the reason the function exists.
    """
    raise NotImplementedError


def source_root() -> Path | None:
    """The course source root above this file, or `None` when there is not one.

    Searched for by landmark rather than counted to. `parents[5]` would be right in
    the checkout and `IndexError` inside the image, where `COPY src/ src/` under
    `WORKDIR /app` leaves four parents — and because `api.py` imports this module
    for `build_version`, that crashed the serving API for a calculation only the
    release lane needs.

    `None` is a real answer, not a failure: in the container there is no course
    tree, and callers say so rather than guess at a directory.
    """
    for parent in Path(__file__).resolve().parents:
        if (parent / ROOT_MARKER).is_file():
            return parent
    return None


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
