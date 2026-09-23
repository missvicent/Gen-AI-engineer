"""The optional JWT gate, exercised entirely offline.

Tokens are minted here with the same key the service validates against, so every
accept and reject path runs in the fast tier with no identity provider anywhere —
including the RS256/JWKS lane, whose only network call is behind an injectable
seam.

The tests worth reading twice are the ones for claims that were missing rather
than wrong. A token with no `exp` and no `sub` used to be *accepted*: it looked
like a working request, and what it actually was is a credential that never
expires, spending memory and approvals that belong to everyone at once. Those
failures have no error message to grep for in production, which is why they need
a test here.
"""

import time

import jwt
import pytest
from fastapi.testclient import TestClient

from assistant import auth
from assistant.api import create_app
from assistant.settings import Settings

SECRET = "test-secret-with-at-least-32-bytes!!"  # HS256 wants >=32 bytes of key
AUDIENCE = "assistant"
ISSUER = "https://issuer.example/realms/assistant"


def secured_client(**overrides) -> TestClient:
    settings = Settings(jwt_secret=SECRET, jwt_audience=AUDIENCE, **overrides)
    return TestClient(create_app(settings), raise_server_exceptions=False)


def claims(scope: str, *, audience: str = AUDIENCE, expires_in: int | None = 300,
           sub: str | None = "alice", issuer: str | None = None) -> dict:
    body: dict = {"aud": audience, "scope": scope}
    if sub is not None:
        body["sub"] = sub
    if expires_in is not None:
        body["exp"] = int(time.time()) + expires_in
    if issuer is not None:
        body["iss"] = issuer
    return body


def mint(scope: str, *, key: str = SECRET, **kwargs) -> str:
    return jwt.encode(claims(scope, **kwargs), key, algorithm="HS256")


def bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# --- the demo path stays open ---------------------------------------------------
def test_without_a_key_source_the_zero_key_demo_path_stays_open():
    c = TestClient(create_app(Settings()))
    assert c.post("/ask", json={"question": "hello"}).status_code == 200


def test_health_reports_which_gate_is_live():
    """An operator outside the process needs to tell a shared secret from a real
    issuer: with HS256 every verifier can also mint, with JWKS none of them can."""
    assert TestClient(create_app(Settings())).get("/health").json()["tier"]["auth"] == "off"
    assert secured_client().get("/health").json()["tier"]["auth"] == "shared-secret"
    jwks = TestClient(create_app(Settings(jwks_url="https://issuer.example/jwks")))
    assert jwks.get("/health").json()["tier"]["auth"] == "jwks"


# --- claims that must be PRESENT, not merely valid if present -------------------
def test_a_token_that_never_expires_is_rejected():
    """The quiet one. PyJWT checks `exp` if it is there and shrugs if it is not,
    so an unexpiring token sailed through — a credential with no end date, which
    is not a session but a permanent key."""
    response = secured_client().post(
        "/ask", json={"question": "hello"},
        headers=bearer(mint("assistant:ask", expires_in=None)),
    )
    assert response.status_code == 401
    assert "exp" in response.json()["detail"]


def test_a_token_that_names_no_subject_is_rejected():
    """`sub` is not decoration: memory, approvals and the audit trail are all
    scoped by it. A token without one used to land in the shared anonymous
    namespace, which is every caller's namespace at once."""
    response = secured_client().post(
        "/ask", json={"question": "hello"}, headers=bearer(mint("assistant:ask", sub=None))
    )
    assert response.status_code == 401
    assert "sub" in response.json()["detail"]


def test_a_token_whose_subject_is_blank_is_rejected():
    """Present-but-empty passes a required-claims check and still names nobody."""
    response = secured_client().post(
        "/ask", json={"question": "hello"}, headers=bearer(mint("assistant:ask", sub="   "))
    )
    assert response.status_code == 401
    assert "subject" in response.json()["detail"]


def test_an_issuer_is_required_once_one_is_configured():
    c = secured_client(jwt_issuer=ISSUER)
    no_issuer = c.post(
        "/ask", json={"question": "hello"}, headers=bearer(mint("assistant:ask"))
    )
    assert no_issuer.status_code == 401
    assert "iss" in no_issuer.json()["detail"]

    wrong = c.post(
        "/ask", json={"question": "hello"},
        headers=bearer(mint("assistant:ask", issuer="https://evil.example")),
    )
    assert wrong.status_code == 401
    assert "issuer" in wrong.json()["detail"]

    right = c.post(
        "/ask", json={"question": "hello"},
        headers=bearer(mint("assistant:ask", issuer=ISSUER)),
    )
    assert right.status_code == 200


# --- expiry and clock skew ------------------------------------------------------
def test_an_expired_token_is_rejected():
    response = secured_client().post(
        "/ask", json={"question": "hello"},
        headers=bearer(mint("assistant:ask", expires_in=-600)),
    )
    assert response.status_code == 401
    assert "expired" in response.json()["detail"]


def test_clock_skew_is_a_policy_with_an_edge_on_both_sides():
    """Two machines disagreeing by a second should not 401 a valid token; a
    token expired ten minutes ago should never be accepted for convenience.
    Leeway is also the window in which a just-expired token still works, which
    is the argument for keeping it small."""
    tolerant = secured_client(jwt_leeway=120)
    just_expired = bearer(mint("assistant:ask", expires_in=-30))
    assert tolerant.post("/ask", json={"question": "x"}, headers=just_expired).status_code == 200
    long_expired = bearer(mint("assistant:ask", expires_in=-600))
    assert tolerant.post("/ask", json={"question": "x"}, headers=long_expired).status_code == 401

    strict = secured_client(jwt_leeway=0)
    assert strict.post("/ask", json={"question": "x"}, headers=just_expired).status_code == 401


# --- signature, audience, scope -------------------------------------------------
def test_a_valid_token_with_the_right_scope_is_accepted():
    c = secured_client()
    assert c.post(
        "/ask", json={"question": "hello"}, headers=bearer(mint("assistant:ask"))
    ).status_code == 200


def test_a_token_for_another_audience_is_rejected():
    # The RFC 8707 passthrough trap: a token minted for some OTHER service must
    # not open this one, even though the signature and expiry check out.
    response = secured_client().post(
        "/ask", json={"question": "hello"},
        headers=bearer(mint("assistant:ask", audience="other-api")),
    )
    assert response.status_code == 401
    assert "audience" in response.json()["detail"]


def test_a_token_missing_the_endpoint_scope_is_rejected():
    # can ask, but /approve demands assistant:approve
    response = secured_client().post(
        "/approve", json={"tool": "send_telegram"}, headers=bearer(mint("assistant:ask"))
    )
    assert response.status_code == 401
    assert "scope" in response.json()["detail"]


def test_a_forged_token_is_rejected():
    wrong = "a-different-32-byte-signing-key!!!!!"
    response = secured_client().post(
        "/ask", json={"question": "hello"}, headers=bearer(mint("assistant:ask", key=wrong))
    )
    assert response.status_code == 401


def test_an_unsigned_token_cannot_talk_its_way_in():
    """alg=none is the oldest JWT attack there is, and it only works against a
    validator that reads the algorithm off the token it is validating."""
    unsigned = jwt.encode(claims("assistant:ask"), key="", algorithm="none")
    response = secured_client().post(
        "/ask", json={"question": "hello"}, headers=bearer(unsigned)
    )
    assert response.status_code == 401


def test_every_mutating_endpoint_is_behind_the_gate():
    c = secured_client()
    assert c.post("/ingest", json={"docs": ["x"]}).status_code == 401
    assert c.post("/ask", json={"question": "x"}).status_code == 401
    assert c.post("/ask/stream", json={"question": "x"}).status_code == 401
    assert c.post("/approve", json={"tool": "x"}).status_code == 401
    # /health stays open: probes and dashboards should not need a token
    assert c.get("/health").status_code == 200


# --- the OAuth 2.1 lane: RS256 verified against an issuer's JWKS ----------------
def rsa_keypair():
    """A throwaway keypair. Generated per test run rather than checked in, so
    there is no private key in the repo for someone to reuse by accident."""
    pytest.importorskip("cryptography")
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = private.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()
    return pem, private.public_key()


class FakeJwks(auth.JwksKeys):
    """The issuer's public keys, without the issuer. Only the fetch is faked —
    every check after it is the code production runs, which is the point of
    having the fetch behind a seam at all."""

    def __init__(self, public_key) -> None:
        super().__init__(url="https://issuer.example/jwks")
        self._public = public_key

    def client(self):
        outer = self

        class _Client:
            def get_signing_key_from_jwt(self, token):
                return type("Key", (), {"key": outer._public})()

        return _Client()


def jwks_policy(**overrides) -> auth.AuthPolicy:
    return auth.AuthPolicy(
        jwks_url="https://issuer.example/jwks", audience=AUDIENCE, **overrides
    )


def test_an_issuer_signed_token_is_accepted_against_the_jwks():
    private_pem, public = rsa_keypair()
    token = jwt.encode(claims("assistant:ask", issuer=ISSUER), private_pem, algorithm="RS256")
    ok, reason, subject = auth.identity_from(
        f"Bearer {token}",
        policy=jwks_policy(issuer=ISSUER),
        required_scope="assistant:ask",
        keys=FakeJwks(public),
    )
    assert (ok, reason, subject) == (True, "ok", "alice")


def test_the_jwks_lane_enforces_the_same_required_claims():
    """The two key sources share one decode path precisely so they cannot drift
    into one being laxer than the other."""
    private_pem, public = rsa_keypair()
    token = jwt.encode(claims("assistant:ask", expires_in=None), private_pem, algorithm="RS256")
    ok, reason, _ = auth.identity_from(
        f"Bearer {token}",
        policy=jwks_policy(),
        required_scope="assistant:ask",
        keys=FakeJwks(public),
    )
    assert not ok and "exp" in reason


def _hs256_by_hand(body: dict, secret: bytes) -> str:
    import base64
    import hashlib
    import hmac
    import json

    def seg(raw: bytes) -> bytes:
        return base64.urlsafe_b64encode(raw).rstrip(b"=")

    signing_input = b".".join((
        seg(json.dumps({"alg": "HS256", "typ": "JWT"}).encode()),
        seg(json.dumps(body).encode()),
    ))
    mac = hmac.new(secret, signing_input, hashlib.sha256).digest()
    return (signing_input + b"." + seg(mac)).decode()


def test_an_hs256_token_cannot_be_smuggled_into_the_rs256_lane():
    """The algorithm-confusion attack: sign with the PUBLIC key as an HMAC
    secret and hope the validator accepts whatever the header claims. Pinning
    the algorithm list to the KEY SOURCE, not the token, is what stops it."""
    _, public = rsa_keypair()
    from cryptography.hazmat.primitives import serialization

    public_pem = public.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    # PyJWT refuses to MINT this ("asymmetric key ... should not be used as an
    # HMAC secret"), which is a fine guard for honest callers and no obstacle at
    # all to an attacker, who has hmac and base64. So forge it by hand — the
    # question under test is what the VERIFIER does with it.
    forged = _hs256_by_hand(claims("assistant:ask"), public_pem)
    ok, reason, _ = auth.identity_from(
        f"Bearer {forged}",
        policy=jwks_policy(),
        required_scope="assistant:ask",
        keys=FakeJwks(public),
    )
    assert not ok, reason


def test_an_unreachable_issuer_fails_closed():
    """If the keys cannot be fetched, no token can be verified — so no token is
    accepted. Failing open here would turn the identity provider's outage into
    an open door."""
    token = jwt.encode(claims("assistant:ask"), SECRET, algorithm="HS256")
    ok, reason, _ = auth.identity_from(
        f"Bearer {token}",
        policy=auth.AuthPolicy(jwks_url="http://127.0.0.1:1/jwks", audience=AUDIENCE),
        required_scope="assistant:ask",
    )
    assert not ok and "keys" in reason
