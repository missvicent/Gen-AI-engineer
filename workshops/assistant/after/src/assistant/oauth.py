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
"""

from __future__ import annotations

import base64
import hashlib
import json
import secrets
import urllib.parse
import urllib.request
from dataclasses import dataclass

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
    """A fresh verifier and its challenge. New per authorization request — a
    reused verifier is a long-lived secret with none of the handling."""
    verifier = _b64url(secrets.token_bytes(VERIFIER_BYTES))
    challenge = _b64url(hashlib.sha256(verifier.encode("ascii")).digest())
    return Pkce(verifier=verifier, challenge=challenge)


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
    """The URL to send the user to, and the `state` to check when they return.

    `audience` is passed through because the assistant validates it (RFC 8707):
    a token minted for some other service must not open this one, so the client
    has to ask for one minted for this one.
    """
    state = state or _b64url(secrets.token_bytes(16))
    query = urllib.parse.urlencode(
        {
            "response_type": "code",  # OAuth 2.1: the implicit grant is gone
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "scope": scope,
            "state": state,
            "audience": audience,
            "code_challenge": pkce.challenge,
            "code_challenge_method": pkce.method,
        }
    )
    return f"{endpoints.authorization}?{query}", state


def exchange_code(
    endpoints: Endpoints,
    *,
    code: str,
    client_id: str,
    redirect_uri: str,
    pkce: Pkce,
    timeout: float = 10.0,
) -> dict:
    """Trade the authorization code for tokens, proving possession of the
    verifier. The redirect URI is sent again and must match exactly — the issuer
    checks it a second time precisely because the first check happened before
    anyone proved who they were."""
    body = urllib.parse.urlencode(
        {
            "grant_type": "authorization_code",
            "code": code,
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "code_verifier": pkce.verifier,
        }
    ).encode()
    request = urllib.request.Request(
        endpoints.token,
        data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return json.load(response)


def callback_params(redirect_url: str, *, expected_state: str) -> str:
    """Pull the authorization code out of the callback, refusing a forged one.

    Comparing state with `secrets.compare_digest` rather than `==` is not
    theatre here: this runs on whatever the browser was told to visit, and a
    timing side channel on a CSRF token is a real (if narrow) way to guess it.
    """
    query = urllib.parse.parse_qs(urllib.parse.urlparse(redirect_url).query)
    if "error" in query:
        raise ValueError(f"authorization failed: {query['error'][0]}")
    returned = (query.get("state") or [""])[0]
    if not secrets.compare_digest(returned, expected_state):
        raise ValueError("state mismatch — this callback did not come from our request")
    code = (query.get("code") or [""])[0]
    if not code:
        raise ValueError("callback carried no authorization code")
    return code
