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

Checks 2, 4 and 5 are the ones this originally got wrong, and they failed in the
most comfortable way possible: a token with no `exp` and no `sub` was *accepted*
and mapped to the shared anonymous identity. It looked like a working request.
What it actually was is a credential that never expires, spending memory and
approvals that belong to everyone at once. PyJWT verifies a claim if it is there
and shrugs if it is not, so "required" has to be said out loud — `require=` and
`options={"verify_..."}` are not the same setting, and only one of them makes a
missing claim an error.

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
        """Pin the algorithms explicitly. An `algorithms` list derived from the
        token's own header is the classic JWT forgery: the attacker picks
        `none`, or downgrades RS256 to HS256 and signs with the public key."""
        if self.algorithms:
            return list(self.algorithms)
        return ["RS256"] if self.jwks_url else ["HS256"]


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
    """Verify and decode, or raise a PyJWT error. Split out because the two key
    sources differ only in where the key comes from — every check after that is
    identical, and duplicating them is how one source ends up laxer."""
    if policy.jwks_url:
        source = keys or JwksKeys(policy.jwks_url)
        key: Any = source.client().get_signing_key_from_jwt(token).key
    else:
        key = policy.secret
    return jwt.decode(
        token,
        key,
        algorithms=policy.signing_algorithms(),
        audience=policy.audience,
        issuer=policy.issuer,
        leeway=policy.leeway,
        # `require` makes a MISSING claim an error; the verify_* flags only say
        # what to do with one that is present. Both are needed.
        options={
            "require": list(REQUIRED_CLAIMS) + (["iss"] if policy.issuer else []),
            "verify_exp": True,
            "verify_aud": True,
            "verify_iss": bool(policy.issuer),
            "verify_signature": True,
        },
    )


def validate_token(
    token: str,
    *,
    policy: AuthPolicy,
    required_scope: str,
    keys: JwksKeys | None = None,
) -> tuple[bool, str, dict]:
    """Returns (ok, reason, claims). The reason is what the caller is told, so
    it names the failed check without echoing the token back."""
    try:
        claims = _decode(token, policy, keys)
    except jwt.ExpiredSignatureError:
        return False, "expired", {}
    except jwt.MissingRequiredClaimError as exc:
        return False, f"missing required claim '{exc.claim}'", {}
    except jwt.InvalidAudienceError:
        return False, "wrong audience — token not issued for this service", {}
    except jwt.InvalidIssuerError:
        return False, "wrong issuer — token not minted by the expected provider", {}
    except jwt.InvalidTokenError as exc:
        return False, f"invalid token: {exc}", {}
    except Exception as exc:  # noqa: BLE001 — a JWKS fetch can fail any number of ways
        return False, f"could not verify against the issuer's keys: {exc}", {}
    # A present-but-empty `sub` passes `require` and names nobody, which would
    # quietly land the caller in the shared anonymous namespace.
    if not str(claims.get("sub", "")).strip():
        return False, "token names no subject", {}
    if required_scope not in str(claims.get("scope", "")).split():
        # OAuth scopes are one space-separated string — a detail every
        # hand-rolled validator forgets, usually by testing `in` on the raw text
        return False, f"missing scope '{required_scope}'", {}
    return True, "ok", claims


def identity_from(
    authorization: str,
    *,
    policy: AuthPolicy,
    required_scope: str,
    keys: JwksKeys | None = None,
) -> tuple[bool, str, str]:
    """Resolve a request's identity. Returns (ok, reason, subject).

    With auth disabled everyone is ANONYMOUS and always allowed — the demo path.
    With auth enabled, a missing or failing token is rejected and the caller's
    identity is the token's verified `sub`."""
    if not policy.enabled:
        return True, "auth disabled", ANONYMOUS
    if not authorization.lower().startswith("bearer "):
        return False, "missing bearer token", ""
    token = authorization.split(" ", 1)[1]
    ok, reason, claims = validate_token(
        token, policy=policy, required_scope=required_scope, keys=keys
    )
    if not ok:
        return False, reason, ""
    return True, "ok", str(claims["sub"])
