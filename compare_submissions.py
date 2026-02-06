import argparse
import hashlib
from pathlib import Path


def sha256_digest(filepath, chunk_size=8192):
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def main(dir1, dir2):
    dir1 = Path(dir1)
    dir2 = Path(dir2)

    if not dir1.is_dir() or not dir2.is_dir():
        raise ValueError("Both inputs must be valid directories.")

    files1 = {p.name: p for p in dir1.iterdir() if p.is_file()}
    files2 = {p.name: p for p in dir2.iterdir() if p.is_file()}

    common_files = sorted(set(files1) & set(files2))
    only_in_1 = sorted(set(files1) - set(files2))
    only_in_2 = sorted(set(files2) - set(files1))

    print("=== Comparing common files ===")
    for name in common_files:
        hash1 = sha256_digest(files1[name])
        hash2 = sha256_digest(files2[name])

        if hash1 == hash2:
            print(f"[MATCH]    {name}")
        else:
            print(f"[DIFF]     {name}")
            print(f"  dir1: {hash1}")
            print(f"  dir2: {hash2}")

    if only_in_1:
        print("\n=== Only in dir1 ===")
        for name in only_in_1:
            print(f"[MISSING]  {name}")

    if only_in_2:
        print("\n=== Only in dir2 ===")
        for name in only_in_2:
            print(f"[MISSING]  {name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare SHA-256 hashes of files with the same name in two directories."
    )
    parser.add_argument("dir1", help="First directory")
    parser.add_argument("dir2", help="Second directory")

    args = parser.parse_args()
    main(args.dir1, args.dir2)
