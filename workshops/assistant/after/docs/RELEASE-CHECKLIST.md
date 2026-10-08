# Release checklist

What has to be true before the numbers on the portfolio page are allowed to be
quoted as measurements of the deployed system, and the order to establish it in.

The short version: `make report` measures a proxy and says so; `make
release-evidence` measures the thing that ships and refuses to run on anything
else. Only the second one produces a number you may put next to the word
"production".

## Which lane may claim what

| Lane | Retrieval | Judge | Red team | May be quoted as |
|------|-----------|-------|----------|------------------|
| `make report` | in-memory BM25 | `KeywordJudge` (lexical) | 3 inline probes | "the offline tier, on this commit" |
| `make release-evidence` | Qdrant, semantic embedder, hybrid RRF, rerank on | RAGAS 0.4 against the pinned judge | all 58 rows, 11 benign controls included | "the deployed configuration, on this commit" |

Both write their instrument into the header of their own page. A number lifted
out of one of those pages and pasted somewhere without the header is the failure
this table exists to prevent.

## Before the run

1. **The fast gates are green.** `make check` here, and `./src/verify-lessons.sh`
   from the repo root. A release measurement of code that does not pass its own
   tests measures nothing worth having.
2. **Your Ollama is up, with both models warm.** `./src/preflight-ollama.sh`. The
   model runs on this machine, not in the stack, so it is the one dependency
   compose cannot start for you or wait on.
3. **The stack boots and every check passes.** `./src/verify-e2e.sh --reset --down`
   — the unqualified form, which is the release gate. It runs the preflight as
   check 0 and attests the Ollama version and both model digests, so the evidence
   names the bytes that answered rather than just a tag.
4. **The judge model is pulled.** `ollama pull qwen3-coder:30b`. It is 18 GB and
   it is not the chat model; on a 16 GB machine, use a hosted judge and say so in
   the page rather than quietly swapping in the chat model.

## The run

The measurement builds its own assistant in the host process — the release judge
and the full dataset are not in the image — and points it at the same Qdrant the
stack answers from and the same Ollama, which is already here. Qdrant has to be
reachable from the host and the base compose file publishes exactly one port on
purpose, so boot with the release overlay:

```bash
cd src/phase8-deploy/01-compose/after
docker compose -f docker-compose.yml -f docker-compose.release.yml up -d --build

cd ../../../workshops/assistant/after
make release-evidence     # OLLAMA_HOST defaults to localhost:11434 — your daemon
```

Output lands in `evidence/`: `RELEASE-EVIDENCE.md` for people and
`release-report.json` for the gate. The run takes a while and most of it is the
judge; that is the price of the row in the table above.

If the tier is wrong, the run exits and names every mismatch — `rag=`, `brain=`,
`retrieval=`, the embedder, the reranker, and anything in `degraded`. Every one
of those is something that fails open in normal operation, which is right for a
service and wrong for a measurement. Fix the tier; do not lower the bar.

The same check runs again after the last question, and that one is the one that
matters. Failing open means a composer can stop answering on question nine and the
suite finishes anyway, scored across two tiers, with the pre-flight check's
"no component fell back" still printed at the top of the page. A run that trips
the second check cannot be repaired afterwards — nobody can say which answers came
from which tier — so it has to be re-run once the stack is stable.

## After the run

- [ ] The header of `RELEASE-EVIDENCE.md` names the source, the tier, the judge,
      the dataset size, and whether the token counts were reported by the
      provider or estimated from a word split.
- [ ] The source id in that header is a bare hash. `dirty-…` means the run
      measured uncommitted changes and `unbound` means it is tied to nothing —
      either way the numbers describe a tree nobody else can check out, and the
      publication gate refuses them. Commit first, then measure.
- [ ] The red-team section shows the benign controls, not only the attacks. A
      containment rate without a false-positive rate is half a result.
- [ ] Anything quoted elsewhere — the README, a portfolio page, a talk — carries
      the lane it came from.
- [ ] `./package.sh && ./verify-dist.sh` from the repo root, and the tag push
      that runs both again in CI (`.github/workflows/release.yml`). That workflow
      cannot run this measurement — a hosted runner has no GPU — so it reads the
      committed report instead and will not publish a tag whose source id differs
      from the one it is building.
