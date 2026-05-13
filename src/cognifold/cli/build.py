"""Build command for Cognifold CLI."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


def add_build_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore
    """Add the build-timeline subcommand parser."""
    build_parser = subparsers.add_parser(
        "build-timeline", help="Build a timeline JSON from external sources"
    )
    build_parser.add_argument(
        "--source",
        type=str,
        choices=["wiki"],
        default="wiki",
        help="Source type (default: wiki)",
    )
    build_parser.add_argument(
        "--input",
        type=str,
        default="data/wiki",
        help="Input directory (default: data/wiki)",
    )
    build_parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output timeline JSON path (default: data/generated/wiki_timeline_<timestamp>.json)",
    )
    build_parser.add_argument(
        "--files",
        type=str,
        action="append",
        help="Specific files to include (ordered)",
    )
    build_parser.add_argument(
        "--blank-titles",
        action="store_true",
        help="Leave event titles blank (use content only)",
    )
    build_parser.add_argument("--chunk-size", type=int, default=1600)
    build_parser.add_argument("--chunk-overlap", type=int, default=200)
    build_parser.add_argument("--min-chunk-size", type=int, default=200)
    build_parser.add_argument(
        "--split-strategy",
        type=str,
        choices=["paragraph", "heading", "fixed"],
        default="heading",
    )
    build_parser.add_argument("--max-chunks-per-doc", type=int, default=200)
    build_parser.add_argument(
        "--include-headings",
        action="store_true",
        help="Include recent headings in each chunk",
    )
    build_parser.add_argument(
        "--timestamp-strategy",
        type=str,
        choices=["lexical_ordered", "mtime_ordered", "frontmatter_date"],
        default="lexical_ordered",
    )
    build_parser.add_argument("--delta-seconds", type=int, default=10)
    build_parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Disable PDF parsing (skip *.pdf files)",
    )


def build_command(args: argparse.Namespace) -> int:
    """Handle build-timeline subcommand."""
    from cognifold.importers.wiki import WikiTimelineBuildSettings, build_wiki_timeline

    if args.source != "wiki":
        print(f"Error: Unsupported source: {args.source}")
        return 1

    settings = WikiTimelineBuildSettings(
        chunk_size_chars=args.chunk_size,
        chunk_overlap_chars=args.chunk_overlap,
        min_chunk_chars=args.min_chunk_size,
        split_strategy=args.split_strategy,
        max_chunks_per_doc=args.max_chunks_per_doc,
        include_headings_in_chunk=bool(args.include_headings),
        timestamp_strategy=args.timestamp_strategy,
        delta_seconds_per_event=args.delta_seconds,
        allow_pdf=not bool(args.no_pdf),
        blank_titles=bool(args.blank_titles),
    )

    result = build_wiki_timeline(args.input, settings=settings, specific_files=args.files)

    output = args.output
    if not output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = f"data/generated/wiki_timeline_{timestamp}.json"

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result.timeline, ensure_ascii=False, indent=2))

    print(f"Wrote timeline: {out_path}")
    print(f"  Files scanned: {result.files_scanned}")
    print(f"  Docs parsed: {result.docs_parsed}")
    print(f"  Events emitted: {result.events_emitted}")
    if result.skipped_files:
        print(f"  Skipped files: {len(result.skipped_files)}")

    return 0
