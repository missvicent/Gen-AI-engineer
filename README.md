# GenAI Engineer — Learning Log

A working repo where I, a senior engineer, building systems *on top
of* language models — retrieval, evaluation, agents, guardrails, deployment — rather
than the ML math underneath them.

This README is a show-and-tell, not a changelog. It captures the **core ideas** I've
actually put into code so far, and how they map to the engineering instincts I already
had.

---

## The engineering ideas that stuck

Coming from other languages, the domain is new but the good patterns are the old ones.
The reps so far keep pointing back to the same handful of principles:

- **Normalize at the boundary.** Wrap an unreliable, vendor-specific dependency in a
  single clean shape so the rest of the code never learns the mess behind it — the
  adapter pattern, applied to a messy new domain.
- **Configuration decides behavior, not branching logic.** Swapping one backend for
  another should be a config change, never a code change.
- **Structured output is a contract, not a hope.** Validate what a model returns and
  fail loudly when it doesn't hold up. An LLM is a nondeterministic dependency, so I
  treat its output like untrusted input from any other service.
- **Streaming is just an iterator.** Text arrives in chunks; the awkward parts get
  filtered out before they reach the caller.
- **Secrets stay out of code.** Configuration comes from the environment, never the
  source.

## Python I picked up along the way

- **Type hints as design** — they document the contract and the linter enforces it.
- **Frozen dataclasses** for small immutable value objects — records without boilerplate.
- **Modern union syntax** and future annotations.
- **Generators** as the natural shape for streaming.
- A fast, opinionated modern toolchain for env, dependencies, formatting, and linting.

---

## What I take away so far

The domain is new, but the engineering isn't. The good patterns are the old ones:
**put an adapter in front of an unreliable dependency, normalize its output at the
boundary, validate what it returns, and drive behavior from config.** That's the lens
I'm carrying forward.
