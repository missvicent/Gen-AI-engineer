"""Real connectors — env-gated replacements for the stubs in tools.py.

The stubs stay the default so the whole capstone runs with zero keys. When the
env var is present, `build_assistant` swaps the tool BODY and nothing else: the
approval gate on send_telegram and the output re-screen on read_news apply to
the real connector exactly as they did to the stub, because policy lives in the
registry and the hardening wrapper, not in the connector.

Both connectors are stdlib-only (urllib + xml.etree) — no new dependencies for
one HTTP POST and one RSS parse.

## Raise inward, apologise outward

Each connector is two functions: an inner one that RAISES on failure and an
outer one that turns the failure into the friendly shape the tool contract
promises. The retry wrapper goes between them, and it has to, because a
connector that catches its own `OSError` and returns `{"error": ...}` is a
connector nothing can retry — from the outside a returned error dict is a
successful call. That is how a retry policy ends up decorative: present in the
code, never once triggered.

The split also puts the classification where the knowledge is. A 4xx from the
Bot API is `Permanent` (the same payload will be rejected again); a 5xx or a
dead socket is not. `resilience.py` cannot know that; this module can.

## Retried, and retried ONCE

`read_news` is a GET: retrying it costs a little latency and nothing else.
`send_telegram` is a send, and a timeout does not tell you whether the message
went out — retrying it is how one alert becomes three. So it gets `ONCE`: the
timeout still applies, the second attempt does not happen, and the outbox
(outbox.py) records the ambiguity for a human instead of guessing.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable
from xml.etree import ElementTree

from assistant.resilience import DEFAULT_POLICY, ONCE, Permanent, Policy, resilient

Sender = Callable[[str, str], dict]

#: Read-only, so retries are free of consequences; the timeout is what actually
#: protects the request budget.
NEWS_POLICY = Policy(attempts=DEFAULT_POLICY.attempts, timeout=10.0)


def _http_error_is_permanent(exc: BaseException) -> bool:
    """A 4xx is the server's final answer. Retrying it three times produces the
    same rejection three times, more slowly, while holding a connection."""
    return isinstance(exc, urllib.error.HTTPError) and 400 <= exc.code < 500


def telegram_sender(bot_token: str, timeout: float = 10.0, policy: Policy = ONCE) -> Sender:
    """A send_telegram body that talks to the real Bot API. Same signature and
    return shape as the stub, so the registry entry (and its approval gate) is
    unchanged."""

    def call(chat_id: str, message: str) -> dict:
        request = urllib.request.Request(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            data=json.dumps({"chat_id": chat_id, "text": message}).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            if _http_error_is_permanent(exc):
                raise Permanent(f"telegram refused: {exc.code}") from exc
            raise
        return {"sent": bool(body.get("ok")), "chat_id": chat_id}

    attempted = resilient(call, policy)

    def send(chat_id: str, message: str) -> dict:
        if not chat_id or not message:
            return {"error": "chat_id and message required"}
        try:
            return attempted(chat_id, message)
        except Exception as exc:
            # The tool contract is "return a dict"; a raised exception here would
            # crash the agent loop over an outage in one connector. The outbox
            # row already records that this was attempted, so the failure is
            # written down even though the caller only sees a polite string.
            return {"error": f"telegram unreachable: {exc}"}

    return send


def parse_feed(xml_text: str, limit: int = 5) -> str:
    """Turn RSS/Atom XML into the plain-text digest read_news returns. Pure and
    offline-testable; fetch_news is the thin network wrapper around it."""
    root = ElementTree.fromstring(xml_text)
    items = root.iter("item")  # RSS 2.0
    headlines = []
    for item in items:
        title = item.findtext("title") or ""
        if title.strip():
            headlines.append(title.strip())
        if len(headlines) >= limit:
            break
    if not headlines:  # Atom fallback: namespaced <entry><title>
        ns = {"a": "http://www.w3.org/2005/Atom"}
        for entry in root.findall("a:entry", ns):
            title = entry.findtext("a:title", default="", namespaces=ns)
            if title.strip():
                headlines.append(title.strip())
            if len(headlines) >= limit:
                break
    if not headlines:
        return "No headlines found in the feed."
    return "Headlines: " + " | ".join(headlines)


def news_fetcher(
    feed_url: str, timeout: float = 10.0, policy: Policy = NEWS_POLICY
) -> Callable[..., str]:
    """A read_news body that fetches a real, keyless RSS feed. Read-only, so the
    hardening wrapper re-screens whatever comes back — a poisoned headline is
    contained the same way a poisoned stub is."""

    def call(url: str) -> str:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return parse_feed(response.read().decode("utf-8", errors="replace"))

    attempted = resilient(call, policy)

    def fetch(url: str = "") -> str:
        try:
            return attempted(url or feed_url)
        except Exception as exc:
            # Degraded, not fatal: an answer without today's headlines is worth
            # more than a 500, and the span records which it was.
            return f"Feed unreachable: {exc}"

    return fetch
