"""OAuth 2.1 + PKCE — how a token gets minted, as opposed to how it gets checked.

`auth.py` is the resource server: it verifies whatever arrives. This is the other
half, and it exists because "we support OAuth" is easy to claim and easy to get
subtly wrong. It is what `docker-compose.oauth.yml` puts in front of the
assistant, and it is stdlib-only for the same reason `connectors.py` is: one
HTTP POST and a SHA-256 do not justify a dependency.

The flow, and what each step is actually defending against:

1. **Discovery.** Fetch `/.well-known/openid-configuration` for the authorize,
   token and JWKS endpoints. Hard-coded endpoints break at the first realm
   rename; worse, they are how a service ends up validating against one issuer
   and redirecting to another.

2. **PKCE.** Generate a random `code_verifier`, send only its SHA-256
   (`code_challenge`), and reveal the verifier at token exchange. An attacker
   who steals the authorization code — off a redirect URL, out of a log, from a
   malicious app claiming the same custom scheme — cannot exchange it without
   the verifier, which never left the client. OAuth 2.1 makes this mandatory for
   every client, not just public ones, and `plain` is not an option here: a
   challenge equal to its verifier defends nothing.

3. **State.** A random value echoed back and compared, which is what makes a
   forged callback fail. It is a CSRF token for the redirect and nothing else;
   it is not PKCE, and having one does not excuse skipping the other.

4. **Exact redirect matching.** The URI is registered with the issuer and sent
   again at exchange. Prefix or wildcard matching is how a code gets delivered
   to somebody else's server.

What is deliberately NOT here: the assistant never forwards a caller's token to
an upstream service. That is the confused-deputy trap from
`phase7-mcp/03-auth-modes`, and the defence is architectural — a token minted
for this audience is spent here and nowhere else.

Reference: ../../after/src/assistant/oauth.py. The tests run entirely offline;
`../../../phase8-deploy/01-compose/after/oauth/pkce_login.py` runs the same
functions against a real Keycloak once you want to watch a browser do it.
"""
from __future__ import annotations

import base64
import json
import urllib.parse
import urllib.request
from dataclasses import dataclass

# You will also want `hashlib` and `secrets` from the standard library. They are
# left out so the scaffold starts lint-clean; the TODOs below say where each one
# belongs.

#: RFC 7636 allows 43-128 characters; 32 random bytes base64url-encode to 43.
VERIFIER_BYTES = 32

#: `plain` is in the RFC and is not offered. A challenge that equals its
#: verifier gives an attacker holding the code everything they need.
CHALLENGE_METHOD = "S256"


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


@dataclass(frozen=True)
class Pkce:
    verifier: str
    challenge: str
    method: str = CHALLENGE_METHOD


def pkce_pair() -> Pkce:
    """TODO 1: a fresh verifier and its challenge.

    Draw the bytes with `secrets.token_bytes` and hash with `hashlib.sha256`.
    The verifier is high-entropy random, base64url with no padding (padding is
    not URL-safe and issuers differ on whether they strip it). The challenge is
    the base64url SHA-256 of the verifier's ASCII bytes — hash the *string*, not
    the bytes you drew, or the issuer's recomputation will never match.

    New per authorization request: a reused verifier is a long-lived secret with
    none of the handling."""
    raise NotImplementedError


@dataclass(frozen=True)
class Endpoints:
    authorization: str
    token: str
    jwks: str
    issuer: str

    @classmethod
    def from_document(cls, document: dict) -> Endpoints:
        return cls(
            authorization=document["authorization_endpoint"],
            token=document["token_endpoint"],
            jwks=document["jwks_uri"],
            issuer=document["issuer"],
        )


def discover(issuer_url: str, *, timeout: float = 10.0) -> Endpoints:
    """Read the issuer's OpenID configuration. One fetch at startup beats four
    URLs in four config files drifting apart."""
    url = issuer_url.rstrip("/") + "/.well-known/openid-configuration"
    with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
        return Endpoints.from_document(json.load(response))


def authorization_url(
    endpoints: Endpoints,
    *,
    client_id: str,
    redirect_uri: str,
    scope: str,
    audience: str,
    pkce: Pkce,
    state: str | None = None,
) -> tuple[str, str]:
    """TODO 2: the URL to send the user to, and the `state` to check on return.

    Generate a random `state` when the caller did not supply one, and return it
    alongside the URL — the caller cannot verify a callback against a value it
    was never told. Then build the query: `response_type=code` (OAuth 2.1
    deleted the implicit grant), the client id, the exact redirect URI, the
    scope, the state, the challenge and its method.

    Send `audience` too. The assistant validates `aud` (RFC 8707), so a token
    minted for some other service must not open this one — which means the
    client has to ask for one minted for this one."""
    raise NotImplementedError


def exchange_code(
    endpoints: Endpoints,
    *,
    code: str,
    client_id: str,
    redirect_uri: str,
    pkce: Pkce,
    timeout: float = 10.0,
) -> dict:
    """TODO 3: trade the authorization code for tokens.

    A form-encoded POST carrying `grant_type=authorization_code`, the code, the
    client id, the redirect URI *again*, and the `code_verifier` — that last one
    is the whole point of PKCE, and the redirect URI is re-checked by the issuer
    precisely because the first check happened before anyone proved who they
    were. Return the parsed JSON body."""
    raise NotImplementedError


def callback_params(redirect_url: str, *, expected_state: str) -> str:
    """TODO 4: pull the authorization code out of the callback, refusing a
    forged one.

    Three refusals, in order: an `error` parameter means the user declined or
    the issuer objected; a `state` that does not match means this callback did
    not come from our request; a callback with no `code` is not a success no
    matter what else it carries.

    Compare state with `secrets.compare_digest` rather than `==`. That is not
    theatre here — this runs on whatever the browser was told to visit, and a
    timing side channel on a CSRF token is a real (if narrow) way to guess it."""
    raise NotImplementedError
