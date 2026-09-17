from contextual_chunks.contextual import contextualize, contextualize_all, split


def _fake_summarizer(prompt: str) -> str:
    return "This chunk is from the refunds policy section."


def test_split_uses_the_library_and_respects_size():
    text = ("Refunds are processed within five business days. " * 40).strip()
    chunks = split(text, chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)


def test_context_is_prepended_before_the_chunk():
    out = contextualize("full doc", "refunds take 5 days", _fake_summarizer)
    assert out.split("\n")[0] == "This chunk is from the refunds policy section."
    assert "refunds take 5 days" in out


def test_contextualize_all_preserves_count_and_order():
    chunks = ["a", "b", "c"]
    out = contextualize_all("doc", chunks, _fake_summarizer)
    assert len(out) == 3
    assert all(o.endswith(c) for o, c in zip(out, chunks, strict=True))
