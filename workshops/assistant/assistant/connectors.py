"""Real connectors — env-gated replacements for the stubs in tools.py.

The stubs stay the default so the whole capstone runs with zero keys. When the
env var is present, `build_assistant` swaps the tool BODY and nothing else: the
approval gate on send_telegram and the output re-screen on read_news apply to
the real connector exactly as they did to the stub, because policy lives in the
registry and the hardening wrapper, not in the connector.

Both connectors are stdlib-only (urllib + xml.etree) — no new dependencies for
one HTTP POST and one RSS parse.

## TODO: raise inward, apologise outward

Look at `send` below: it catches its own `OSError` and returns `{"error": ...}`.
That makes it a connector nothing can retry, because from the outside a returned
error dict is a *successful call*. This is how a retry policy ends up decorative
— present in the code, never once triggered.

Split each connector in two: an inner function that RAISES, and an outer one
that turns the failure into the friendly shape the tool contract promises. The
retry wrapper goes between them. The split also puts classification where the
knowledge is: a 4xx from the Bot API is `Permanent` (the same payload will be
rejected again), a 5xx or a dead socket is not, and `resilience.py` cannot know
that but this module can.

Then choose the policies. `read_news` is a GET — retrying costs a little latency
and nothing else. `send_telegram` is a send, and a timeout does not tell you
whether the message went out, so retrying is how one alert becomes three.

Reference: ../../after/src/assistant/connectors.py.
"""
from __future__ import annotations

import json
import urllib.request
from collections.abc import Callable
from xml.etree import ElementTree

from assistant.resilience import DEFAULT_POLICY, ONCE, Policy

Sender = Callable[[str, str], dict]

#: Read-only, so retries are free of consequences; the timeout is what actually
#: protects the request budget.
NEWS_POLICY = Policy(attempts=DEFAULT_POLICY.attempts, timeout=10.0)


def telegram_sender(
    bot_token: str, timeout: float = 10.0, policy: Policy = ONCE
) -> Sender:
    """A send_telegram body that talks to the real Bot API. Same signature and
    return shape as the stub, so the registry entry (and its approval gate) is
    unchanged.

    TODO 1: `policy` is accepted and ignored. Split the HTTP call out into an
    inner function that RAISES, wrap that in `resilience.resilient(call, policy)`,
    and raise `resilience.Permanent` for a 4xx (`urllib.error.HTTPError` with
    `400 <= code < 500`). Keep the outer `send` returning a dict on failure — the
    tool contract is "return a value", and a raised exception here would crash the
    agent loop over one connector's outage.
    """

    def send(chat_id: str, message: str) -> dict:
        if not chat_id or not message:
            return {"error": "chat_id and message required"}
        request = urllib.request.Request(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            data=json.dumps({"chat_id": chat_id, "text": message}).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read())
        except OSError as exc:
            return {"error": f"telegram unreachable: {exc}"}
        return {"sent": bool(body.get("ok")), "chat_id": chat_id}

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
    contained the same way a poisoned stub is.

    TODO 2: the same split, with a retrying `policy` this time. A GET is safe to
    repeat, and a feed that was unreachable a moment ago often is not now.
    """

    def fetch(url: str = "") -> str:
        target = url or feed_url
        try:
            with urllib.request.urlopen(target, timeout=timeout) as response:
                return parse_feed(response.read().decode("utf-8", errors="replace"))
        except OSError as exc:
            return f"Feed unreachable: {exc}"

    return fetch
