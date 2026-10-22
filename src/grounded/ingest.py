"""Turn saved RBI web pages into clause-level sections.

Why parse structure instead of just extracting text?
    Regulations are organised in numbered clauses (e.g. 2.1.1.2, or 10 with
    items i, ii, iii). If we keep those numbers, every chunk of text knows which
    clause it came from, so the assistant can cite "fraud_risk:3.3.4" instead of
    "somewhere in the document", and the evaluation can check that the cited
    clause is the right one.

What the parser does
    1. Finds the document body (RBI puts it in <table class="tablebg">).
    2. Drops footnote markers (<sup>), footnotes, the page header and the
       table of contents.
    3. Walks the body in reading order and emits blocks: headings, paragraphs,
       list items and tables (tables are flattened to "cell | cell" rows, so the
       customer-liability tables stay readable).
    4. Gives every block a clause id:
         "2.1.1.2 ..."           -> 2.1.1.2      (explicit number at the start)
         heading "10. Cooling-off period", then "i. ..." -> 10, 10(i)
         unnumbered text         -> the clause it sits under
         "Annex II: ..."         -> Annex II
    5. Writes one JSON line per block to data/corpus/sections.jsonl.

Run:  python -m grounded.ingest
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass

from bs4 import BeautifulSoup, NavigableString, Tag

from grounded import config

NUMBERED = re.compile(r"^(\d{1,2}(?:\.\d{1,2})*)\.?\s+(?=\S)")
ROMAN_ITEM = re.compile(r"^\(?((?:x{0,2})(?:ix|iv|v?i{0,3}))[.)]\s+", re.IGNORECASE)
CHAPTER = re.compile(r"^chapter\s+[ivxl]+\b", re.IGNORECASE)
ANNEX = re.compile(r"^(annex(?:ure)?\s*[ivxl0-9]*)", re.IGNORECASE)


@dataclass
class Block:
    doc: str
    order: int
    clause: str          # e.g. "3.3.4", "10(ii)", "Annex II", "preamble"
    kind: str            # heading | para | table
    heading: str         # the nearest section heading, for context
    text: str


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _annex_id(text: str) -> str:
    """'Annex II: List of ...' -> 'Annex II';  'Appendix' -> 'Appendix'."""
    word, _, rest = text.partition(" ")
    numeral = rest.split(":")[0].strip().upper() if rest else ""
    numeral = numeral if re.fullmatch(r"[IVXL]+|\d+", numeral) else ""
    return f"{word.capitalize().replace('Annexure', 'Annex')} {numeral}".strip()


def _table_text(table: Tag) -> str:
    rows = []
    for tr in table.find_all("tr"):
        cells = [_clean(td.get_text(" ")) for td in tr.find_all(["td", "th"])]
        cells = [c for c in cells if c]
        if cells:
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def _raw_blocks(body: Tag):
    """Yield (kind, text) in reading order."""
    for child in body.children:
        if isinstance(child, NavigableString):
            text = _clean(str(child))
            if len(text) > 30:
                yield "para", text
            continue
        if not isinstance(child, Tag):
            continue
        if child.name == "table":
            first = _clean(child.get_text(" "))[:40].lower()
            if first.startswith(("index", "contents")):  # table of contents: skip
                continue
            text = _table_text(child)
            if text:
                yield "table", text
        elif child.name == "p":
            if "footnote" in (child.get("class") or []):
                continue
            text = _clean(child.get_text(" "))
            if text:
                yield ("heading" if "head" in (child.get("class") or []) else "para"), text
        elif child.name == "li" and not child.find("p"):
            text = _clean(child.get_text(" "))
            if text:
                yield "para", text
        elif child.name in {"div", "ol", "ul", "li", "span", "font", "center", "tbody", "tr", "td", "blockquote", "b"}:
            yield from _raw_blocks(child)


def parse_html(html: bytes | str, doc: str) -> list[Block]:
    soup = BeautifulSoup(html, "lxml")
    body = soup.select_one("table.tablebg")
    if body is None:
        raise ValueError(f"{doc}: couldn't find the RBI document body (table.tablebg)")
    for tag in body.select("sup, td.tableheader, script, style"):
        tag.decompose()

    # The outer table's cells hold the content; walk them in order.
    containers = body.find_all("td", recursive=True)
    container = max(containers, key=lambda td: len(td.find_all("p", recursive=False)), default=body)

    blocks: list[Block] = []
    clause, top, heading = "preamble", None, ""
    for kind, text in _raw_blocks(container):
        m = NUMBERED.match(text)
        if kind == "heading":
            if ANNEX.match(text) or text.lower().startswith("appendix"):
                clause = top = _annex_id(text)
            elif m:
                clause = top = m.group(1)
            elif text.lower() == "introduction":
                clause = top = "introduction"
            # chapter titles ("CHAPTER III") only change the context heading
            heading = text
        elif m and float(m.group(1).split(".")[0]) < 100:
            clause = top = m.group(1)
        elif ROMAN_ITEM.match(text) and top and ROMAN_ITEM.match(text).group(1):
            # "i. ...", "(ii) ..." are items of the clause they sit in: 10 -> 10(i), 3.2 -> 3.2(i)
            clause = f"{top}({ROMAN_ITEM.match(text).group(1).lower()})"
        blocks.append(Block(doc, len(blocks), clause, kind, heading, text))

    # An unnumbered heading ("Reversal Timeline for ...") introduces what FOLLOWS it,
    # so it takes the clause id of the next non-heading block.
    following = None
    for b in reversed(blocks):
        if b.kind == "heading" and not NUMBERED.match(b.text) and not ANNEX.match(b.text) and following is not None:
            b.clause = following
        else:
            following = b.clause
    return blocks


def main():
    config.SECTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with open(config.SECTIONS_PATH, "w", encoding="utf-8") as out:
        for doc in config.SOURCES:
            path = config.RAW_DIR / f"{doc}.html"
            if not path.exists():
                print(f"missing {path.name}: see data/README.md")
                continue
            blocks = parse_html(path.read_bytes(), doc)
            for b in blocks:
                out.write(json.dumps(asdict(b), ensure_ascii=False) + "\n")
            words = sum(len(b.text.split()) for b in blocks)
            clauses = len({b.clause for b in blocks})
            print(f"{doc:<20} {len(blocks):>4} blocks  {clauses:>4} clauses  {words:>6} words")
            total += len(blocks)
    print(f"wrote {total} blocks -> {config.SECTIONS_PATH.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
