"""Parsing RBI pages into clauses, and chunking them."""

from grounded.chunking import Chunk, build_chunks, group_key


def _by_clause(blocks):
    out = {}
    for b in blocks:
        out.setdefault(b["clause"], []).append(b["text"])
    return out


def test_clause_ids_follow_document_numbering(blocks):
    clauses = [b["clause"] for b in blocks]
    assert "2(i)" in clauses and "2(ii)" in clauses        # "i." items under heading "2."
    assert "3.1" in clauses and "3.1.1" in clauses          # explicit dotted numbers
    assert "Annex I" in clauses


def test_noise_is_removed(blocks):
    text = " ".join(b["text"] for b in blocks)
    assert "Footnote text" not in text                      # footnotes dropped
    assert "loans1" not in text and "loans 1 " not in text  # <sup> markers dropped
    assert not any(b["text"].startswith("Index") for b in blocks)   # table of contents skipped
    assert "( 100 kb )" not in text                          # page header dropped


def test_tables_are_flattened_and_attached_to_their_clause(blocks):
    by = _by_clause(blocks)
    table = next(t for t in by["3"] if "|" in t)
    assert "Within 2 days | Zero" in table and "Rs 5,000" in table


def test_unnumbered_heading_belongs_to_the_clause_it_introduces(blocks):
    heading = next(b for b in blocks if b["text"] == "Customer liability")
    assert heading["clause"] == "3"


def test_group_key():
    assert group_key("10(ii)") == "10"
    assert group_key("3.3.4") == "3.3"
    assert group_key("2.1.1.2") == "2.1"
    assert group_key("Annex II") == "Annex II"


def test_clause_chunks_keep_clauses_together(clause_chunks):
    cooling = next(c for c in clause_chunks if c.id == "sample:2")
    assert cooling.clauses == ["2", "2(i)", "2(ii)"]
    assert "not less than three days" in cooling.text and "processing fee" in cooling.text
    assert cooling.text.startswith("sample | 2. Cooling-off period")      # header gives context
    assert not any(c.text.strip().endswith("Chapter I: Preliminary") for c in clause_chunks)


def test_covers_matches_clause_and_subclauses():
    c = Chunk("d:3.1", "d", ["3.1", "3.1.1"], "", "")
    assert c.covers("d:3.1") and c.covers("d:3.1.1") and c.covers("d:3")
    assert not c.covers("d:3.2") and not c.covers("other:3.1")


def test_fixed_chunks_cover_all_words_with_overlap(blocks):
    chunks = build_chunks("fixed", blocks)
    from grounded.chunking import fixed_chunks
    small = fixed_chunks(blocks, size=20, overlap=5)
    total_words = sum(len(b["text"].split()) for b in blocks)
    assert len(small) > 1
    assert small[0].text.split()[-5:] == small[1].text.split()[1:6]          # 5-word overlap (after the header word)
    assert sum(len(c.text.split()) - 1 for c in small) >= total_words        # nothing dropped
    assert chunks and all(c.id.startswith("sample#w") for c in chunks)
