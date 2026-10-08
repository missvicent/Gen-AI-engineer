# ADR-0009 — The gate requires claims, and the issuer is pluggable

**Status:** accepted

## Context

The first gate called `jwt.decode(token, key, algorithms=["HS256"], audience=...)`
and checked `scope` afterwards. Every test passed, because every token the tests
minted had the claims in it.

PyJWT verifies a claim if it is present and shrugs if it is not. `verify_exp`
does not mean "there must be an `exp`", it means "if there is one, honour it".
So a token with no `exp` and no `sub` was **accepted** — and `identity_from`
mapped the missing subject to the shared anonymous identity. The result looked
exactly like a working request. What it actually was is a credential that never
expires, spending the memory and approvals of everyone at once. ADR-0008 had
just made the subject the partition key; this quietly routed unauthenticated
traffic into the shared partition.

The second problem was structural rather than a bug: HS256 with a shared secret
means every verifier is also a forger. That is acceptable for one service whose
operator also mints the tokens, and it is not what a deployment with an identity
provider looks like.

## Decision

An `AuthPolicy` object carries the whole configuration, and `require=` states
out loud which claims must be **present** — `exp`, `aud`, `sub`, plus `iss` when
an issuer is configured. A present-but-blank `sub` is rejected separately,
because it satisfies `require` and still names nobody.

Two key sources share one decode path. `ASSISTANT_JWT_SECRET` verifies HS256.
`ASSISTANT_JWKS_URL` verifies RS256 against the issuer's published keys, cached
in-process so the identity provider's bad afternoon does not become ours. The
accepted algorithms are pinned by the policy and never read from the token's own
header, which is the classic forgery: pick `none`, or downgrade RS256 to HS256
and sign with the public key everyone can fetch.

Clock skew is a stated policy, not an accident: `ASSISTANT_JWT_LEEWAY`, default
60 seconds, applied to `exp`/`nbf`.

`oauth.py` implements the other half — authorization code + PKCE (S256 only),
discovery, exact redirect matching, `state` compared with `compare_digest` — so
the course can show how a token gets minted, not only how it gets checked.
`docker-compose.oauth.yml` runs it against a real Keycloak.

## Alternatives considered

Keep the verify flags and document the requirement (documentation is not a
control, and the failure mode is silent acceptance). Reject only missing `exp`
and keep anonymous fallback for missing `sub` (leaves the shared-partition hole
open). Make JWKS the only mode (a course that needs an identity provider running
before lesson one teaches nobody). Ship an OAuth client library (the flow is one
POST and a SHA-256; hiding it behind a dependency hides exactly the four
defences the reader is here to see). Introspection endpoints instead of local
verification (a network round trip per request, and an availability coupling to
the IdP that JWKS caching specifically avoids).

## Consequences

Tokens minted by the old code are rejected, which is the point and is still a
breaking change. Leeway is a real window: for 60 seconds a just-expired token
still works, so revocation is not instant and the secure profile narrows it to
30. RS256 pulls in `pyjwt[crypto]` and therefore `cryptography` — a compiled
dependency the offline tier would otherwise not need. The JWKS cache means a
rotated key is picked up on the next unseen `kid`, not on a schedule; an issuer
that rotates without changing `kid` will break, and that is the issuer's bug,
not something this service can detect.
