from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

import yaml

SplitStrategy = Literal["paragraph", "heading", "fixed"]
TimestampStrategy = Literal["lexical_ordered", "mtime_ordered", "frontmatter_date"]


@dataclass(frozen=True)
class WikiTimelineBuildSettings:
    chunk_size_chars: int = 1600
    chunk_overlap_chars: int = 200
    min_chunk_chars: int = 200
    split_strategy: SplitStrategy = "heading"
    max_chunks_per_doc: int = 200
    include_headings_in_chunk: bool = True
    timestamp_strategy: TimestampStrategy = "lexical_ordered"
    base_timestamp: datetime = field(
        default_factory=lambda: datetime(2020, 1, 1, tzinfo=timezone.utc)
    )
    delta_seconds_per_event: int = 10
    allow_pdf: bool = True
    blank_titles: bool = False


@dataclass
class WikiTimelineBuildResult:
    timeline: dict[str, Any]
    files_scanned: int
    docs_parsed: int
    events_emitted: int
    skipped_files: list[dict[str, str]] = field(default_factory=list)


def build_wiki_timeline(
    input_dir: str | Path,
    settings: WikiTimelineBuildSettings | None = None,
    specific_files: list[str] | None = None,
) -> WikiTimelineBuildResult:
    settings = settings or WikiTimelineBuildSettings()
    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(f"Input directory not found: {root}")

    if specific_files:
        files = [root / f for f in specific_files]
        files_scanned = len(files)
    else:
        files = _iter_files(root)
        files_scanned = len(files)
        files = _sort_files(files, settings.timestamp_strategy)

    base = settings.base_timestamp
    events: list[dict[str, Any]] = []
    docs_parsed = 0
    skipped: list[dict[str, str]] = []

    for file_index, path in enumerate(files):
        try:
            doc = _parse_document(path, allow_pdf=settings.allow_pdf)
        except Exception as e:
            skipped.append({"path": str(path), "reason": str(e)})
            continue

        docs_parsed += 1
        doc_id = _doc_id(root, path)
        doc_title = doc.get("title") or path.stem
        doc_text = doc.get("text") or ""
        doc_meta = doc.get("metadata") or {}

        chunks = _chunk_document(
            doc_text,
            strategy=settings.split_strategy,
            chunk_size=settings.chunk_size_chars,
            overlap=settings.chunk_overlap_chars,
            min_size=settings.min_chunk_chars,
            max_chunks=settings.max_chunks_per_doc,
            include_headings=settings.include_headings_in_chunk,
        )

        for chunk_index, chunk in enumerate(chunks):
            step = len(events)
            ts = _timestamp_for_event(
                settings=settings,
                base=base,
                global_index=step,
                file_index=file_index,
                document=doc,
                path=path,
            )
            event_id = f"w-{doc_id}-{chunk_index:04d}"

            meta = {
                "source_path": str(path.relative_to(root)),
                "doc_id": doc_id,
                "doc_title": doc_title,
                "doc_type": doc.get("type"),
                "chunk_index": chunk_index,
                "chunk_start": chunk.get("start"),
                "chunk_end": chunk.get("end"),
                "headings": chunk.get("headings") or [],
                "settings": {
                    "chunk_size_chars": settings.chunk_size_chars,
                    "chunk_overlap_chars": settings.chunk_overlap_chars,
                    "min_chunk_chars": settings.min_chunk_chars,
                    "split_strategy": settings.split_strategy,
                    "include_headings_in_chunk": settings.include_headings_in_chunk,
                },
            }
            if doc_meta:
                meta["doc_metadata"] = doc_meta

            events.append(
                {
                    "event_id": event_id,
                    "timestamp": ts.isoformat(),
                    "event_type": "wiki_chunk",
                    "title": ""
                    if settings.blank_titles
                    else f"{doc_title} · chunk {chunk_index + 1}/{len(chunks)}",
                    "description": chunk.get("text") or "",
                    "metadata": meta,
                }
            )

    timeline = {
        "timeline_id": f"wiki-{_dataset_hash(root, files)}",
        "description": json.dumps(
            {
                "source": str(root),
                "settings": {
                    "chunk_size_chars": settings.chunk_size_chars,
                    "chunk_overlap_chars": settings.chunk_overlap_chars,
                    "min_chunk_chars": settings.min_chunk_chars,
                    "split_strategy": settings.split_strategy,
                    "max_chunks_per_doc": settings.max_chunks_per_doc,
                    "include_headings_in_chunk": settings.include_headings_in_chunk,
                    "timestamp_strategy": settings.timestamp_strategy,
                    "delta_seconds_per_event": settings.delta_seconds_per_event,
                    "allow_pdf": settings.allow_pdf,
                },
            },
            ensure_ascii=False,
        ),
        "events": events,
    }

    return WikiTimelineBuildResult(
        timeline=timeline,
        files_scanned=files_scanned,
        docs_parsed=docs_parsed,
        events_emitted=len(events),
        skipped_files=skipped,
    )


def _iter_files(root: Path) -> list[Path]:
    candidates: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in {".md", ".markdown", ".txt", ".pdf"}:
            candidates.append(path)
    return candidates


def _sort_files(files: list[Path], strategy: TimestampStrategy) -> list[Path]:
    if strategy == "mtime_ordered":
        return sorted(files, key=lambda p: (p.stat().st_mtime, str(p)))
    return sorted(files, key=lambda p: str(p))


def _dataset_hash(root: Path, files: list[Path]) -> str:
    h = hashlib.sha1()
    for p in files:
        try:
            rel = str(p.relative_to(root)).encode("utf-8")
        except Exception:
            rel = str(p).encode("utf-8")
        h.update(rel)
        try:
            h.update(str(int(p.stat().st_mtime)).encode("utf-8"))
            h.update(str(p.stat().st_size).encode("utf-8"))
        except Exception:
            pass
    return h.hexdigest()[:12]


def _doc_id(root: Path, path: Path) -> str:
    rel = str(path.relative_to(root)).encode("utf-8")
    return hashlib.sha1(rel).hexdigest()[:12]


def _parse_document(path: Path, allow_pdf: bool) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix in {".md", ".markdown"}:
        return _parse_markdown(path)
    if suffix == ".txt":
        return _parse_text(path)
    if suffix == ".pdf":
        if not allow_pdf:
            raise ValueError("PDF parsing disabled by settings")
        return _parse_pdf(path)
    raise ValueError(f"Unsupported file type: {suffix}")


def _parse_text(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return {"type": "txt", "title": path.stem, "text": text, "metadata": {}}


def _parse_markdown(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    metadata: dict[str, Any] = {}
    text = raw

    if raw.startswith("---"):
        parts = raw.split("\n", 1)
        if len(parts) == 2:
            rest = parts[1]
            fm_end = rest.find("\n---")
            if fm_end != -1:
                fm_text = rest[:fm_end].strip()
                metadata = yaml.safe_load(fm_text) or {}
                text = rest[fm_end + len("\n---") :].lstrip("\n")

    title = _first_markdown_h1(text) or metadata.get("title") or path.stem
    return {"type": "md", "title": title, "text": text, "metadata": metadata}


def _first_markdown_h1(text: str) -> str | None:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return None


def _parse_pdf(path: Path) -> dict[str, Any]:
    try:
        from pypdf import PdfReader  # type: ignore[import-not-found]
    except Exception as e:
        raise ImportError("Missing optional dependency: pypdf") from e

    reader = PdfReader(str(path))
    parts: list[str] = []
    for page in reader.pages:
        txt = page.extract_text() or ""
        if txt:
            parts.append(txt)
    text = "\n\n".join(parts)
    title = None
    try:
        if reader.metadata and getattr(reader.metadata, "title", None):
            title = str(reader.metadata.title)
    except Exception:
        title = None
    meta: dict[str, Any] = {}
    try:
        if reader.metadata:
            meta = {k.lstrip("/"): str(v) for k, v in reader.metadata.items() if v is not None}
    except Exception:
        meta = {}
    return {"type": "pdf", "title": title or path.stem, "text": text, "metadata": meta}


def _chunk_document(
    text: str,
    strategy: SplitStrategy,
    chunk_size: int,
    overlap: int,
    min_size: int,
    max_chunks: int,
    include_headings: bool,
) -> list[dict[str, Any]]:
    text = text.strip()
    if not text:
        return []

    if strategy == "fixed":
        return _window_chunks(text, chunk_size, overlap, min_size, max_chunks)

    if strategy == "paragraph":
        units = _split_paragraphs(text)
        return _pack_units(units, chunk_size, overlap, min_size, max_chunks)

    units = _split_markdown_headings(text, include_headings=include_headings)
    return _pack_units(units, chunk_size, overlap, min_size, max_chunks)


def _split_paragraphs(text: str) -> list[dict[str, Any]]:
    raw_parts = [p.strip() for p in text.split("\n\n") if p.strip()]
    units: list[dict[str, Any]] = []
    cursor = 0
    for part in raw_parts:
        start = text.find(part, cursor)
        if start == -1:
            start = cursor
        end = start + len(part)
        cursor = end
        units.append({"text": part, "start": start, "end": end, "headings": []})
    return units


def _split_markdown_headings(text: str, include_headings: bool) -> list[dict[str, Any]]:
    lines = text.splitlines(keepends=True)
    units: list[dict[str, Any]] = []
    headings: list[str] = []
    buf: list[str] = []
    buf_start = 0
    offset = 0

    def flush(end_offset: int) -> None:
        nonlocal buf, buf_start
        content = "".join(buf).strip()
        if not content:
            buf = []
            buf_start = end_offset
            return
        prefix = ""
        if include_headings and headings:
            prefix = "\n".join(f"## {h}" for h in headings[-2:]).strip() + "\n\n"
        merged = (prefix + content).strip()
        units.append(
            {
                "text": merged,
                "start": buf_start,
                "end": end_offset,
                "headings": list(headings),
            }
        )
        buf = []
        buf_start = end_offset

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            level = len(stripped) - len(stripped.lstrip("#"))
            if 1 <= level <= 6 and stripped[level : level + 1] == " ":
                flush(offset)
                title = stripped[level + 1 :].strip()
                headings[:] = headings[: max(0, level - 1)]
                headings.append(title)
                buf.append(line)
                offset += len(line)
                continue

        buf.append(line)
        offset += len(line)

    flush(offset)
    return units


def _pack_units(
    units: list[dict[str, Any]],
    chunk_size: int,
    overlap: int,
    min_size: int,
    max_chunks: int,
) -> list[dict[str, Any]]:
    combined = "\n\n".join(u["text"] for u in units).strip()
    if not combined:
        return []
    return _window_chunks(combined, chunk_size, overlap, min_size, max_chunks)


def _window_chunks(
    text: str,
    chunk_size: int,
    overlap: int,
    min_size: int,
    max_chunks: int,
) -> list[dict[str, Any]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size_chars must be > 0")
    if overlap < 0:
        raise ValueError("chunk_overlap_chars must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("chunk_overlap_chars must be < chunk_size_chars")

    chunks: list[dict[str, Any]] = []
    start = 0
    n = len(text)
    while start < n and len(chunks) < max_chunks:
        end = min(n, start + chunk_size)
        chunk_text = text[start:end].strip()
        if len(chunk_text) >= min_size or (end == n and chunk_text):
            chunks.append({"text": chunk_text, "start": start, "end": end, "headings": []})
        if end == n:
            break
        start = end - overlap
    return chunks


def _timestamp_for_event(
    settings: WikiTimelineBuildSettings,
    base: datetime,
    global_index: int,
    file_index: int,
    document: dict[str, Any],
    path: Path,
) -> datetime:
    if settings.timestamp_strategy == "mtime_ordered":
        try:
            return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        except Exception:
            return base + timedelta(seconds=global_index * settings.delta_seconds_per_event)

    if settings.timestamp_strategy == "frontmatter_date":
        meta = document.get("metadata") or {}
        for key in ("date", "created", "created_at", "updated", "updated_at"):
            if meta.get(key):
                try:
                    dt = datetime.fromisoformat(str(meta[key]).replace("Z", "+00:00"))
                    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
                except Exception:
                    break

    return base + timedelta(seconds=global_index * settings.delta_seconds_per_event)
