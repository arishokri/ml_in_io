#!/usr/bin/env python3
"""
Collect feedback HTML files from:
  feedback/<student_name>/<hw_name>/<hw_name>.html

and package them into a zip where each entry is renamed to:
  <hw_name>_<student_name>.html

Original files are NOT modified, moved, or renamed.

Usage examples:
  python bundle_feedback.py --hw hw1
  python bundle_feedback.py --hw hw1 --feedback-root feedback --out zips
  python bundle_feedback.py --hw hw1 --zip-name hw1_feedback.zip
  python bundle_feedback.py --hw hw1 --dry-run
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def next_available_path(path: Path) -> Path:
    """If path exists, return path with _2, _3, ... before extension."""
    if not path.exists():
        return path
    i = 2
    while True:
        candidate = path.with_name(f"{path.stem}_{i}{path.suffix}")
        if not candidate.exists():
            return candidate
        i += 1


def safe_student_name(p: Path, feedback_root: Path, hw: str) -> str | None:
    """
    Given a file path like feedback/<student>/<hw>/<hw>.html,
    infer <student> safely based on relative parts.
    """
    try:
        rel = p.relative_to(feedback_root)
    except ValueError:
        return None

    # Expected: <student>/<hw>/<hw>.html
    if len(rel.parts) != 3:
        return None
    student, hw_dir, filename = rel.parts
    if hw_dir != hw:
        return None
    if filename != f"{hw}.html":
        return None
    return student


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hw", required=True, help="Homework/assignment name (e.g., hw1)")
    ap.add_argument(
        "--feedback-root",
        type=Path,
        default=Path("feedback"),
        help="Root directory containing feedback/ (default: feedback)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("."),
        help="Output directory for the zip (default: current directory)",
    )
    ap.add_argument(
        "--zip-name",
        default=None,
        help="Zip filename (default: <hw>_feedback.zip)",
    )
    ap.add_argument(
        "--flat",
        action="store_true",
        help="(default behavior) Store files at zip root. Included for readability.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be added without writing a zip",
    )
    args = ap.parse_args()

    hw: str = args.hw.strip()
    if not hw:
        raise ValueError("--hw cannot be empty")

    feedback_root: Path = args.feedback_root
    if not feedback_root.exists():
        raise FileNotFoundError(f"Feedback root not found: {feedback_root}")

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    zip_name = args.zip_name or f"{hw}_feedback.zip"
    zip_path = next_available_path(out_dir / zip_name)

    # Find files matching feedback/*/<hw>/<hw>.html
    pattern = f"*/{hw}/{hw}.html"
    candidates = sorted(feedback_root.glob(pattern))

    if not candidates:
        print(f"No files found matching: {feedback_root}/{pattern}")
        return 0

    # Build a plan (and detect collisions)
    planned = []
    used_arc_names: set[str] = set()
    skipped = 0

    for src in candidates:
        student = safe_student_name(src, feedback_root, hw)
        if not student:
            skipped += 1
            continue

        # Zip entry name (flat at root):
        arc_name = f"{hw}_{student}.html"

        # Avoid collisions if two students map to same name (unlikely but possible)
        if arc_name in used_arc_names:
            base = Path(arc_name).stem
            suffix = Path(arc_name).suffix
            i = 2
            while f"{base}_{i}{suffix}" in used_arc_names:
                i += 1
            arc_name = f"{base}_{i}{suffix}"

        used_arc_names.add(arc_name)
        planned.append((src, arc_name))

    if args.dry_run:
        print(f"[DRY] Would create: {zip_path}")
        for src, arc in planned:
            print(f"[DRY] add {src} as {arc}")
        if skipped:
            print(f"[DRY] Skipped {skipped} unexpected path(s).")
        print(f"[DRY] Total files: {len(planned)}")
        return 0

    added = 0
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for src, arc in planned:
            # zf.write reads the file and stores it under arcname; source remains unchanged.
            zf.write(src, arcname=arc)
            added += 1

    print(f"Done. Wrote {added} file(s) to '{zip_path}'. Skipped {skipped} file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
