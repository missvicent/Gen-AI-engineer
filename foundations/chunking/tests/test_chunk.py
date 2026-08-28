from src.chunk import fixed_size, heading_aware, worst_chunk

DOC = "# A\n" + ("alpha " * 600) + "\n# B\n" + ("beta " * 100)


def test_fixed_size_respects_window():
    chunks = fixed_size("word " * 2000, size=512, overlap=75)
    assert all(len(c.split()) <= 512 for c in chunks)
    assert len(chunks) > 1


def test_overlap_actually_overlaps():
    words = [f"w{i}" for i in range(1000)]
    chunks = fixed_size(" ".join(words), size=100, overlap=20)
    # last 20 words of chunk 0 should reappear at the start of chunk 1
    tail = chunks[0].split()[-20:]
    head = chunks[1].split()[:20]
    assert tail == head


def test_heading_aware_starts_sections_on_headings():
    chunks = heading_aware(DOC, max_words=512)
    assert any(c.startswith("# A") for c in chunks)
    assert any(c.startswith("# B") for c in chunks)


def test_worst_chunk_returns_shortest():
    assert worst_chunk(["a b c", "a", "a b"]) == "a"
