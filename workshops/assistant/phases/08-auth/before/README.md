# Workshop 8 — auth

Edit `src/assistant/auth.py`, `src/assistant/oauth.py`. Nothing else in this folder is yours to write.

Files that belong to a later workshop are not in this folder, or they are already
finished because this one imports them. If a test fails in a file that is not
listed above, you are in the wrong folder.

## Do this

```bash
make setup
uv run pytest -q tests/test_auth.py::test_without_a_key_source_the_zero_key_demo_path_stays_open
make test
```

`make test` runs every test in this folder, including the ones earlier workshops
already made green.

## What the first failure means

`test_without_a_key_source_the_zero_key_demo_path_stays_open` — With no key configured, the service is the local demo and this test is the one that fails first. The next failures are the tokens: a token with no `exp` must be rejected, and PyJWT will not do that unless you require the claim.

## Done when

- [ ] `make test` exits 0.
- [ ] The failure you started from is gone, and no new failure appeared in a file you do not own.

## Stuck?

1. require exp, sub, aud, and scope. verify_exp alone does not reject a token that omits exp.
2. Pin the algorithm from the key source. HS256 for a shared secret, RS256 for JWKS. Never read it from the token header.

Diff only the file you were asked to edit, against this layer's solution:

```bash
diff -u src/assistant/auth.py ../after/src/assistant/auth.py
```

## Carry your work forward

From this folder, once the previous layer is green:

```bash
make adopt-mine FROM=../../08-service/before
```

That copies only the files the previous layer asked you to write.
