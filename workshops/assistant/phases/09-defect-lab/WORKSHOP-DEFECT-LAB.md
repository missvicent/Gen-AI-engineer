# Workshop · The defect lab  (ends Phase 8)

Three vulnerabilities were in this codebase. Reviewed, tested, green, deployed
in the compose stack, walked past by everyone including the person who wrote
them — until an audit read the code with a different question in mind.

They are still here, in `after/defects/variants.py`, running. Your job is to
write the regression tests that catch them, and to prove those tests can fail.

**Where you work.** The lab runs in `workshops/assistant/after`, because that
tree has the Dockerfile and the finished `approvals.py` and `output_gate.py`.

The file you fill in starts as `phases/09-defect-lab/test_regressions.py`. The
copy already in `after/defects/test_regressions.py` is the reference. Move it
aside before you replace it, or you will be editing the solution.

```bash
cd src/workshops/assistant/after
mv defects/test_regressions.py defects/test_regressions.reference.py
cp ../phases/09-defect-lab/test_regressions.py defects/test_regressions.py
make defect-lab
```

Do not edit `defects/variants.py`. That file is the bug, on purpose.

The first `make defect-lab` says the lab is not green yet. That message is the
start. Write the tests named in the TODO file. Green against the current code
comes first. Then the same tests must go red when each defect is seeded back
in. When you are stuck, diff your file against
`defects/test_regressions.reference.py`.

## Why the defects were kept

Deleting a fixed vulnerability deletes the evidence. What is instructive is not
the patch — you can read that in a diff in thirty seconds — it is that all three
of these looked *fine*. None announces itself at the call site. Each passed a
test suite written by someone who had thought about the problem. A vulnerability
you have only read about in a postmortem is one you will write again, because
you will not recognise it in the shape it arrives in.

So they are preserved as drop-in replacements with the real interfaces, and the
lab seeds them back in to interrogate your tests.

## The mechanism, and why the order matters

`make defect-lab` runs your regression file twice per defect:

1. **against the current code — it must pass.**
2. **with the defect seeded in (`DEFECT=...`) — it must fail.**

Green first, then red. That order is the whole gate. "My test went red" is only
evidence if the test was capable of being green; otherwise an unwritten test, a
typo'd import and a collection error all read as a caught defect. And a test
that only ever passes is a test you have no information about — it has never
demonstrated it can detect anything, and most of the ones written after the fix
are quietly asserting something beside the point. They look identical to the
good ones. This is the only thing that tells them apart.

## The three defects

### 1. `unbound-approval` — the counter that authorized everyone

The original grant was `counts["send_telegram"] += 1`. A human said yes, so a
send may proceed, once. Four separate holes hide in that sentence:

| what the grant never recorded | the incident |
|---|---|
| the subject | Alice approves; Bob's request finds the count and sends |
| the arguments | approval for `{"chat_id": "team"}` also sends to the press |
| an expiry | last Tuesday's yes fires today's send |
| an atomic spend | check-then-decrement: two callers both read `1`, both send |

`CountingApprovalStore` even takes a lock, which is what makes it convincing —
it protects the integer from corruption, which was never the risk. The risk is
two callers making the same decision from the same value, and a mutex around
each half of a read-modify-write does nothing about that.

Four tests, one per hole, named `test_unbound_approval_*`.

**On the concurrency one:** in production the gap between read and decide held a
database round trip, so it was milliseconds wide and lost every race it entered.
In-process, CPython's GIL usually carries a thread through both statements, and
a naive test passes nine runs in ten. The seeded variant makes the gap explicit
so the race is deterministic. A flaky test that misses the defect nine times out
of ten is worse than no test — it is an alibi.

### 2. `pre-gate-stream` — the gate that was really a notification

Streaming yielded each chunk as it arrived and screened the joined answer at the
end. By the time the gate fired, the PII was rendered in the caller's browser
and sitting in their proxy logs; the `blocked` event that followed announced a
decision that had already been overtaken.

This one has no seeded variant file, because the vulnerable path is still in the
codebase as `output_gate.RAW` — labelled local-only, kept because seeing the two
side by side is the lesson, and reachable by any operator who sets an env var.
Your test's job is therefore to prove the *default* is safe, not that the unsafe
code was deleted.

Then prove the fix is not "buffer everything". Trading a real vulnerability for
a fake fix that destroys the feature is the most common repair, and it passes
the first test perfectly. Named `test_pre_gate_stream_*`.

### 3. `lockless-build` — the image that was not the image you tested

`uv sync` with no lockfile re-resolves dependencies at build time. Rebuild an
unchanged commit two weeks later and you get different code, with nothing in the
diff, the tag or the logs to say so. The first symptom is a transitive
dependency's patch release breaking production on a commit nobody touched.

Prove the build installs from a committed lock, and that the base images are
pinned — a floating base is the same defect one layer down. Named
`test_lockless_build_*`.

## Assert the property, not the fix

Do not count SQL statements. Do not check that `ApprovalStore` has a `_lock`.
State what a user cares about: Bob cannot spend Alice's approval; two racing
requests send one message; nothing crosses the boundary before the gate has seen
it; the image is built from a pinned lockfile.

A test written against the *fix* rather than the *property* goes green on the
next refactor — and stays green through the next regression, which is the only
time it mattered.

## Done when

- [ ] `make defect-lab` prints `every defect is caught, and the fix is green`.
- [ ] You can say, for each defect, what a reviewer would have had to ask to
      catch it. "Read more carefully" is not an answer; "who is this grant for?"
      is.
- [ ] You wrote each test before reading `after/defects/test_regressions.py`,
      and you watched it go red before you saw it go green.

## The point of the whole exercise

`after/` is a **teaching reference, not a production authority**, and this lab
is the proof. A codebase confident enough to ship its own vulnerabilities as
coursework is telling you something true about review: these three survived it.
The question worth carrying out of here is not "how do I avoid these three" —
it is "what is currently green in my repo for a reason nobody has tested?"
