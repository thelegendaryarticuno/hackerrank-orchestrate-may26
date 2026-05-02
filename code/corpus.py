"""Corpus loading and markdown chunking."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable


COMPANY_DIRS = {
    "hackerrank": "HackerRank",
    "claude": "Claude",
    "visa": "Visa",
}


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    company: str
    path: str
    title: str
    breadcrumbs: tuple[str, ...]
    heading: str
    body: str
    source_url: str
    product_area_hint: str

    @property
    def searchable_text(self) -> str:
        parts = [
            self.company,
            self.product_area_hint,
            self.title,
            " ".join(self.breadcrumbs),
            self.heading,
            self.body,
        ]
        return "\n".join(p for p in parts if p)


def load_corpus(root: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for dir_name, company in COMPANY_DIRS.items():
        company_root = root / dir_name
        if not company_root.exists():
            continue
        for path in sorted(company_root.rglob("*.md")):
            chunks.extend(_chunk_file(path, company_root, company))
    return chunks


def _chunk_file(path: Path, company_root: Path, company: str) -> list[Chunk]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    meta, body = _split_front_matter(raw)
    rel = path.relative_to(company_root)
    title = meta.get("title") or _first_heading(body) or path.stem.replace("-", " ")
    breadcrumbs = tuple(meta.get("breadcrumbs", []))
    source_url = str(meta.get("source_url", ""))
    product_area = _product_area(company, rel, breadcrumbs)
    sections = _split_sections(body)
    if not sections:
        sections = [("", body)]

    chunks: list[Chunk] = []
    for index, (heading, section_body) in enumerate(sections):
        for part_index, part in enumerate(_split_long_text(section_body)):
            cleaned = _clean_text(part)
            if len(cleaned.split()) < 15 and path.name != "index.md":
                continue
            doc_id = f"{company}:{rel}:{index}:{part_index}"
            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    company=company,
                    path=str(path),
                    title=title,
                    breadcrumbs=breadcrumbs,
                    heading=heading or title,
                    body=cleaned,
                    source_url=source_url,
                    product_area_hint=product_area,
                )
            )
    return chunks


def _split_front_matter(raw: str) -> tuple[dict[str, object], str]:
    if not raw.startswith("---\n"):
        return {}, raw
    end = raw.find("\n---", 4)
    if end == -1:
        return {}, raw
    front = raw[4:end].strip("\n")
    body = raw[end + 4 :].lstrip("\n")
    return _parse_simple_yaml(front), body


def _parse_simple_yaml(front: str) -> dict[str, object]:
    meta: dict[str, object] = {}
    lines = front.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip() or line.lstrip().startswith("#"):
            i += 1
            continue
        if ":" not in line:
            i += 1
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip().strip('"')
        if value:
            meta[key] = value
            i += 1
            continue
        items: list[str] = []
        i += 1
        while i < len(lines) and lines[i].startswith("  - "):
            items.append(lines[i][4:].strip().strip('"'))
            i += 1
        meta[key] = items
    return meta


def _first_heading(body: str) -> str:
    match = re.search(r"^#\s+(.+)$", body, flags=re.MULTILINE)
    return match.group(1).strip() if match else ""


def _split_sections(body: str) -> list[tuple[str, str]]:
    matches = list(re.finditer(r"^(#{1,4})\s+(.+)$", body, flags=re.MULTILINE))
    if not matches:
        return []
    sections: list[tuple[str, str]] = []
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        heading = match.group(2).strip()
        section_body = body[start:end].strip()
        if section_body:
            sections.append((heading, section_body))
    return sections


def _split_long_text(text: str, target_words: int = 650) -> Iterable[str]:
    words = text.split()
    if len(words) <= target_words:
        yield text
        return
    for start in range(0, len(words), target_words):
        yield " ".join(words[start : start + target_words])


def _clean_text(text: str) -> str:
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"^>+\s?", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _product_area(company: str, rel: Path, breadcrumbs: tuple[str, ...]) -> str:
    if len(rel.parts) > 1:
        area = rel.parts[0]
    elif breadcrumbs:
        area = breadcrumbs[0]
    else:
        area = rel.stem
    area = area.replace("_", "-").strip().lower()
    if company == "HackerRank":
        aliases = {
            "hackerrank-community": "community",
            "general-help": "general-help",
        }
        return aliases.get(area, area)
    if company == "Claude":
        return area
    return breadcrumbs[0].lower() if breadcrumbs else "visa-support"
