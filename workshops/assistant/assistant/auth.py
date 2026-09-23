"""Optional Bearer-JWT auth for the service — the phase 7.3 checks, applied here.

Off by default: without a key source the service stays a zero-key local demo and
every request runs as the anonymous "local" identity. With one configured, every
mutating endpoint demands a token that survives six checks:

  1. signature — minted with a key we trust, using an algorithm we chose
  2. exp       — present AND not expired
  3. aud       — issued FOR THIS service, not passed through from another
  4. iss       — minted by the issuer we expect, when one is configured
  5. sub       — names a caller, because everything downstream is scoped by it
  6. scope     — grants the specific capability being exercised

Checks 2, 4 and 5 are the ones the first version of this file got wrong, and they
failed in the most comfortable way possible: a token with no `exp` and no `sub`
was *accepted* and mapped to the shared anonymous identity. It looked like a
working request. What it actually was is a credential that never expires,
spending memory and approvals that belong to everyone at once.

Sit with that before you fix it. PyJWT verifies a claim if it is there and shrugs
if it is not — so "required" has to be said out loud. `require=` and
`options={"verify_..."}` are not the same setting, and only one of them makes a
*missing* claim an error. A validator that only sets the verify flags passes
every test you would think to write, because every token you mint by hand has
the claims in it.

Two key sources, one code path:

**Shared secret (HS256).** `ASSISTANT_JWT_SECRET`. Fine for a single service you
also mint tokens for; it means every verifier is also a forger.

**Issuer JWKS (RS256).** `ASSISTANT_JWKS_URL`. What an OAuth 2.1 deployment
actually looks like: the identity provider signs with a private key nobody else
holds, this service fetches the public half and only ever verifies. Rotation is
the issuer's problem, and a leaked verifier cannot mint anything.

Clock skew is a policy, not an accident. Two machines disagreeing by a second
should not 401 a valid token, so `ASSISTANT_JWT_LEEWAY` (default 60s) widens
`exp`/`nbf` — deliberately small, because leeway is also the window in which a
just-expired token still works.

The verified `sub` claim becomes the caller's identity — the hook that memory
namespacing, approval grants and the audit log all hang off.

Reference: ../../after/src/assistant/auth.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jwt

ANONYMOUS = "local"

#: claims that must be PRESENT, not merely valid if present
REQUIRED_CLAIMS = ("exp", "aud", "sub")

#: seconds of clock disagreement tolerated on exp/nbf
DEFAULT_LEEWAY = 60.0


@dataclass(frozen=True)
class AuthPolicy:
    """Everything the gate needs to judge a token. Absent `secret` and
    `jwks_url` both mean auth is off — the zero-key demo path."""

    secret: str | None = None
    jwks_url: str | None = None
    audience: str = "assistant"
    issuer: str | None = None
    leeway: float = DEFAULT_LEEWAY
    algorithms: tuple[str, ...] = ()

    @property
    def enabled(self) -> bool:
        return bool(self.secret or self.jwks_url)

    def signing_algorithms(self) -> list[str]:
        """TODO 1: the algorithms this policy will accept, pinned by US.

        Honour an explicit `algorithms` if one was configured; otherwise choose
        from the key source — a JWKS deployment verifies RS256, a shared secret
        verifies HS256. Never derive this from the token's own header: that is
        the classic JWT forgery, where the attacker picks `none`, or downgrades
        RS256 to HS256 and signs with the public key everybody can fetch."""
        raise NotImplementedError


@dataclass
class JwksKeys:
    """The issuer's public keys, fetched once and cached.

    A cache is not an optimisation here, it is a dependency decision: without
    one, every request to this service becomes a request to the identity
    provider, and the IdP's bad afternoon becomes ours. `refresh()` exists so a
    key rotation does not need a restart — PyJWT's client re-fetches when it is
    asked for a `kid` it has not seen.
    """

    url: str
    _client: Any = field(default=None, repr=False)

    def client(self) -> Any:
        if self._client is None:
            self._client = jwt.PyJWKClient(self.url, cache_keys=True)
        return self._client

    def refresh(self) -> None:
        self._client = None


def _decode(token: str, policy: AuthPolicy, keys: JwksKeys | None) -> dict:
    """TODO 2: verify and decode, or let PyJWT's error out.

    The two key sources differ only in where the key comes from — for JWKS, ask
    the (cached) client for the signing key named by the token's `kid`; for a
    shared secret, the key is the secret. Keep every check after that identical:
    duplicating them is how one source quietly ends up laxer than the other.

    Pass `algorithms`, `audience`, `issuer` and `leeway` from the policy, and
    make the REQUIRED_CLAIMS actually required — plus `iss` when the policy
    names an issuer."""
    raise NotImplementedError


def validate_token(
    token: str,
    *,
    policy: AuthPolicy,
    required_scope: str,
    keys: JwksKeys | None = None,
) -> tuple[bool, str, dict]:
    """TODO 3: (ok, reason, claims).

    Turn each PyJWT failure into a reason that names the check that failed
    without echoing the token back: expired, missing required claim, wrong
    audience, wrong issuer, anything else invalid — and a JWKS fetch that simply
    could not happen, which is not the caller's fault but is still a refusal.

    Then two checks PyJWT will not do for you:
      - a present-but-EMPTY `sub` satisfies `require` and names nobody, which
        would quietly land the caller in the shared anonymous namespace;
      - `scope` is one space-separated string, so membership is a split, not a
        substring test — `"assistant:ask" in "assistant:asking"` is True."""
    raise NotImplementedError


def identity_from(
    authorization: str,
    *,
    policy: AuthPolicy,
    required_scope: str,
    keys: JwksKeys | None = None,
) -> tuple[bool, str, str]:
    """TODO 4: resolve a request's identity. Returns (ok, reason, subject).

    With auth disabled everyone is ANONYMOUS and always allowed — the demo path.
    With auth enabled, a missing or failing `Authorization: Bearer <token>`
    header is rejected, and the caller's identity is the token's verified `sub`
    (never a value taken from anywhere else in the request)."""
    raise NotImplementedError
