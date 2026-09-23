"""The OAuth 2.1 + PKCE client, offline.

Everything except the two HTTP calls is pure, so the parts that are easy to get
quietly wrong — the challenge derivation, the state comparison, sending the
verifier at exchange rather than at authorize — are all testable in the fast
tier. The network is exercised in `docker-compose.oauth.yml`, against Keycloak.
"""
import base64
import hashlib
import urllib.parse

import pytest

from assistant import oauth

ENDPOINTS = oauth.Endpoints(
    authorization="https://issuer.example/protocol/openid-connect/auth",
    token="https://issuer.example/protocol/openid-connect/token",
    jwks="https://issuer.example/protocol/openid-connect/certs",
    issuer="https://issuer.example",
)


def query_of(url: str) -> dict[str, str]:
    return {k: v[0] for k, v in urllib.parse.parse_qs(urllib.parse.urlparse(url).query).items()}


# --- PKCE ----------------------------------------------------------------------
def test_the_challenge_is_the_sha256_of_the_verifier():
    pair = oauth.pkce_pair()
    expected = base64.urlsafe_b64encode(
        hashlib.sha256(pair.verifier.encode("ascii")).digest()
    ).rstrip(b"=").decode()
    assert pair.challenge == expected
    assert pair.method == "S256"


def test_the_verifier_is_long_enough_to_be_unguessable():
    """RFC 7636 sets the floor at 43 characters, and the floor is the point:
    a short verifier is a brute-forceable one."""
    pair = oauth.pkce_pair()
    assert 43 <= len(pair.verifier) <= 128
    assert set(pair.verifier) <= set(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~"
    )


def test_every_authorization_gets_a_fresh_verifier():
    assert oauth.pkce_pair().verifier != oauth.pkce_pair().verifier


def test_plain_is_not_on_offer():
    """`plain` is in the RFC and defends nothing: a challenge equal to its
    verifier hands a code thief the exchange."""
    assert oauth.CHALLENGE_METHOD == "S256"
    assert oauth.pkce_pair().challenge != oauth.pkce_pair().verifier


# --- the authorization request --------------------------------------------------
def test_the_authorization_url_sends_the_challenge_and_never_the_verifier():
    pair = oauth.pkce_pair()
    url, _ = oauth.authorization_url(
        ENDPOINTS, client_id="assistant-cli", redirect_uri="http://127.0.0.1:8765/callback",
        scope="assistant:ask", audience="assistant", pkce=pair,
    )
    params = query_of(url)
    assert params["code_challenge"] == pair.challenge
    assert params["code_challenge_method"] == "S256"
    assert pair.verifier not in url, "the verifier must not leave until exchange"


def test_the_authorization_url_asks_for_a_code_not_a_token():
    """OAuth 2.1 removed the implicit grant. `response_type=token` puts an
    access token in a URL, which is to say in browser history and every proxy
    log between here and the issuer."""
    url, _ = oauth.authorization_url(
        ENDPOINTS, client_id="assistant-cli", redirect_uri="http://127.0.0.1:8765/callback",
        scope="assistant:ask", audience="assistant", pkce=oauth.pkce_pair(),
    )
    assert query_of(url)["response_type"] == "code"


def test_the_authorization_url_names_the_audience_it_wants_a_token_for():
    """The assistant rejects a token minted for another service, so the client
    has to ask for one minted for this one (RFC 8707)."""
    url, _ = oauth.authorization_url(
        ENDPOINTS, client_id="assistant-cli", redirect_uri="http://127.0.0.1:8765/callback",
        scope="assistant:ask", audience="assistant", pkce=oauth.pkce_pair(),
    )
    assert query_of(url)["audience"] == "assistant"


def test_state_is_generated_when_the_caller_does_not_supply_one():
    _, first = oauth.authorization_url(
        ENDPOINTS, client_id="c", redirect_uri="http://127.0.0.1:8765/callback",
        scope="s", audience="assistant", pkce=oauth.pkce_pair(),
    )
    _, second = oauth.authorization_url(
        ENDPOINTS, client_id="c", redirect_uri="http://127.0.0.1:8765/callback",
        scope="s", audience="assistant", pkce=oauth.pkce_pair(),
    )
    assert first and second and first != second


# --- the callback ---------------------------------------------------------------
def test_a_callback_with_the_right_state_yields_the_code():
    assert oauth.callback_params(
        "http://127.0.0.1:8765/callback?code=abc123&state=s-t-a-t-e",
        expected_state="s-t-a-t-e",
    ) == "abc123"


def test_a_forged_callback_is_refused():
    """Without this check, anyone who can make the browser visit the redirect
    URI can feed the client an authorization code of their choosing."""
    with pytest.raises(ValueError, match="state mismatch"):
        oauth.callback_params(
            "http://127.0.0.1:8765/callback?code=attacker-code&state=not-ours",
            expected_state="s-t-a-t-e",
        )


def test_a_callback_carrying_an_error_says_so_instead_of_returning_nothing():
    with pytest.raises(ValueError, match="access_denied"):
        oauth.callback_params(
            "http://127.0.0.1:8765/callback?error=access_denied&state=s",
            expected_state="s",
        )


def test_a_callback_with_no_code_is_an_error_not_an_empty_string():
    with pytest.raises(ValueError, match="no authorization code"):
        oauth.callback_params(
            "http://127.0.0.1:8765/callback?state=s", expected_state="s"
        )


# --- the token exchange ----------------------------------------------------------
def test_the_exchange_proves_possession_of_the_verifier(monkeypatch):
    """The whole point of PKCE lands in one field. A code stolen off a redirect
    is worthless without the verifier that never travelled with it."""
    sent: dict = {}

    class FakeResponse:
        def read(self) -> bytes:
            return b'{"access_token": "tok", "token_type": "Bearer"}'

        def __enter__(self):
            return self

        def __exit__(self, *_exc) -> None:
            return None

    def fake_urlopen(request, timeout=None):
        sent["url"] = request.full_url
        sent["body"] = dict(urllib.parse.parse_qsl(request.data.decode()))
        return FakeResponse()

    monkeypatch.setattr(oauth.urllib.request, "urlopen", fake_urlopen)

    pair = oauth.pkce_pair()
    tokens = oauth.exchange_code(
        ENDPOINTS, code="abc123", client_id="assistant-cli",
        redirect_uri="http://127.0.0.1:8765/callback", pkce=pair,
    )
    assert tokens["access_token"] == "tok"
    assert sent["url"] == ENDPOINTS.token
    assert sent["body"]["code_verifier"] == pair.verifier
    assert sent["body"]["grant_type"] == "authorization_code"
    # sent AGAIN at exchange: the issuer re-checks it now that someone has
    # proved who they are, which is the check that actually binds the code
    assert sent["body"]["redirect_uri"] == "http://127.0.0.1:8765/callback"


# --- discovery --------------------------------------------------------------------
def test_discovery_reads_the_endpoints_off_the_issuer():
    document = {
        "issuer": "https://issuer.example",
        "authorization_endpoint": "https://issuer.example/auth",
        "token_endpoint": "https://issuer.example/token",
        "jwks_uri": "https://issuer.example/certs",
    }
    endpoints = oauth.Endpoints.from_document(document)
    assert endpoints.jwks == "https://issuer.example/certs"
    assert endpoints.issuer == "https://issuer.example"
