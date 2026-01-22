#!/usr/bin/env python3
"""
Unzip an archive of notebooks named like:
  <student_id>_anything_anything_anything.ipynb

and place them into:
  submitted/<student_id>/<assignment>/<assignment>.ipynb

By default:
  assignment = zip stem (e.g., hw1.zip -> hw1)

Usage:
  python organize_submissions.py hw1.zip
  python organize_submissions.py hw1.zip --assignment hw1 --out submitted
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def safe_extract_member(
    zf: zipfile.ZipFile, member: zipfile.ZipInfo, dest_path: Path
) -> None:
    """
    Extract a single zip member to dest_path WITHOUT trusting member.filename,
    preventing Zip Slip by never using the member's path for output.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with zf.open(member, "r") as src, open(dest_path, "wb") as dst:
        dst.write(src.read())


def next_available_path(path: Path) -> Path:
    """
    If path exists, return a new path with _2, _3, ... inserted before extension.
    """
    if not path.exists():
        return path
    i = 2
    while True:
        candidate = path.with_name(f"{path.stem}_{i}{path.suffix}")
        if not candidate.exists():
            return candidate
        i += 1


def infer_student_id(filename: str) -> str | None:
    """
    Student id is the first underscore-separated token of the *base filename*.
    Returns None if it can't be inferred.
    """
    base = Path(filename).name  # ignore any zip internal folders
    if "_" not in base:
        return None
    student_id = base.split("_", 1)[0].strip()
    return student_id or None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "zipfile", type=Path, help="Path to the .zip archive (e.g., hw1.zip)"
    )
    p.add_argument(
        "--assignment",
        default=None,
        help="Assignment name (default: zip stem, e.g., hw1)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("submitted"),
        help="Output root directory (default: submitted)",
    )
    p.add_argument(
        "--dry-run", action="store_true", help="Show actions without writing files"
    )
    p.add_argument(
        "--allow-non-ipynb",
        action="store_true",
        help="Also process non-.ipynb files (default: no)",
    )
    args = p.parse_args()

    zip_path: Path = args.zipfile
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip file not found: {zip_path}")

    assignment = args.assignment or zip_path.stem
    out_root: Path = args.out

    extracted = 0
    skipped = 0

    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue

            base_name = Path(member.filename).name

            if (not args.allow_non_ipynb) and (
                not base_name.lower().endswith(".ipynb")
            ):
                skipped += 1
                continue

            student_id = infer_student_id(base_name)
            if not student_id:
                skipped += 1
                continue

            target_dir = out_root / student_id / assignment
            target_path = target_dir / f"{assignment}.ipynb"
            target_path = next_available_path(target_path)

            if args.dry_run:
                print(f"[DRY] {member.filename} -> {target_path}")
            else:
                safe_extract_member(zf, member, target_path)

            extracted += 1

    print(
        f"Done. Extracted {extracted} file(s) into '{out_root}/'. Skipped {skipped} file(s)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
