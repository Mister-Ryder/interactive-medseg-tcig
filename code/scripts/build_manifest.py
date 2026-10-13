"""Scan a dataset directory and write a lightweight TCIG manifest.

The output is JSON Lines by default so it can be streamed on servers with large
datasets. Pairing is heuristic and conservative: files with label-like tokens
are treated as masks, and matching uses normalized stems after removing common
image/label suffixes.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Iterable, Optional


IMAGE_EXTENSIONS = (
    ".nii.gz",
    ".nii",
    ".mha",
    ".mhd",
    ".nrrd",
    ".npz",
    ".npy",
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
)
LABEL_TOKENS = ("label", "labels", "mask", "masks", "seg", "segs", "segmentation", "gt", "truth")
IMAGE_TOKENS = ("image", "images", "img", "imgs", "volume", "volumes", "scan", "scans")


@dataclass(frozen=True)
class ManifestRecord:
    case_id: str
    image: str
    label: Optional[str]
    dataset: str
    image_bytes: int
    label_bytes: Optional[int]
    multiclass_label: Optional[str] = None
    multiclass_label_bytes: Optional[int] = None
    image_sha1: Optional[str] = None
    label_sha1: Optional[str] = None
    multiclass_label_sha1: Optional[str] = None


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a TCIG dataset manifest from a directory tree.")
    parser.add_argument("--root", type=Path, required=True, help="Dataset root to scan.")
    parser.add_argument("--output", type=Path, required=True, help="Manifest path (.jsonl or .json).")
    parser.add_argument("--dataset", default=None, help="Dataset name stored in each record.")
    parser.add_argument("--relative-to", type=Path, default=None, help="Base path for stored file paths.")
    parser.add_argument("--image-dir", default=None, help="Explicit image directory relative to root.")
    parser.add_argument("--label-dir", default=None, help="Explicit binary/primary label directory relative to root.")
    parser.add_argument(
        "--multiclass-label-dir",
        default=None,
        help="Optional explicit multiclass label directory relative to root.",
    )
    parser.add_argument("--hash", action="store_true", help="Add SHA1 hashes. Slower on large datasets.")
    parser.add_argument("--allow-unlabeled", action="store_true", help="Keep image records without paired labels.")
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"dataset root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"dataset root is not a directory: {root}")

    base = (args.relative_to or root).resolve()
    dataset = args.dataset or root.name
    if args.image_dir or args.label_dir or args.multiclass_label_dir:
        if not args.image_dir:
            raise ValueError("--image-dir is required when explicit directories are used")
        records = build_manifest_from_dirs(
            root=root,
            dataset=dataset,
            relative_to=base,
            image_dir=args.image_dir,
            label_dir=args.label_dir,
            multiclass_label_dir=args.multiclass_label_dir,
            include_hash=args.hash,
            allow_unlabeled=args.allow_unlabeled,
        )
    else:
        records = build_manifest(
            root=root,
            dataset=dataset,
            relative_to=base,
            include_hash=args.hash,
            allow_unlabeled=args.allow_unlabeled,
        )
    write_manifest(records, args.output)
    print(f"wrote {len(records)} records to {args.output}")
    if records:
        paired = sum(record.label is not None for record in records)
        print(f"paired labels: {paired}/{len(records)}")
        multiclass = sum(record.multiclass_label is not None for record in records)
        if multiclass:
            print(f"paired multiclass labels: {multiclass}/{len(records)}")
    return 0


def build_manifest(
    *,
    root: Path,
    dataset: str,
    relative_to: Path,
    include_hash: bool = False,
    allow_unlabeled: bool = False,
) -> list[ManifestRecord]:
    files = sorted(path for path in root.rglob("*") if path.is_file() and _has_supported_extension(path))
    images = [path for path in files if not _is_label_file(path.relative_to(root))]
    labels = [path for path in files if _is_label_file(path.relative_to(root))]
    labels_by_key = {_case_key(path): path for path in labels}

    records: list[ManifestRecord] = []
    for image in images:
        key = _case_key(image)
        label = labels_by_key.get(key)
        if label is None and not allow_unlabeled:
            continue
        records.append(
            ManifestRecord(
                case_id=key,
                image=_display_path(image, relative_to),
                label=None if label is None else _display_path(label, relative_to),
                dataset=dataset,
                image_bytes=image.stat().st_size,
                label_bytes=None if label is None else label.stat().st_size,
                image_sha1=_sha1(image) if include_hash else None,
                label_sha1=None if label is None or not include_hash else _sha1(label),
            )
        )
    return records


def build_manifest_from_dirs(
    *,
    root: Path,
    dataset: str,
    relative_to: Path,
    image_dir: str,
    label_dir: Optional[str] = None,
    multiclass_label_dir: Optional[str] = None,
    include_hash: bool = False,
    allow_unlabeled: bool = False,
) -> list[ManifestRecord]:
    image_root = root / image_dir
    if not image_root.is_dir():
        raise NotADirectoryError(f"image directory does not exist: {image_root}")

    label_root = None if label_dir is None else root / label_dir
    if label_root is not None and not label_root.is_dir():
        raise NotADirectoryError(f"label directory does not exist: {label_root}")

    multiclass_root = None if multiclass_label_dir is None else root / multiclass_label_dir
    if multiclass_root is not None and not multiclass_root.is_dir():
        raise NotADirectoryError(f"multiclass label directory does not exist: {multiclass_root}")

    images = sorted(path for path in image_root.rglob("*") if path.is_file() and _has_supported_extension(path))
    labels_by_stem = _files_by_stem(label_root) if label_root is not None else {}
    multiclass_by_stem = _files_by_stem(multiclass_root) if multiclass_root is not None else {}

    records: list[ManifestRecord] = []
    for image in images:
        case_id = _stem_without_extension(image)
        label = labels_by_stem.get(case_id)
        multiclass_label = multiclass_by_stem.get(case_id)
        if label is None and not allow_unlabeled:
            continue
        records.append(
            ManifestRecord(
                case_id=case_id,
                image=_display_path(image, relative_to),
                label=None if label is None else _display_path(label, relative_to),
                dataset=dataset,
                image_bytes=image.stat().st_size,
                label_bytes=None if label is None else label.stat().st_size,
                multiclass_label=None
                if multiclass_label is None
                else _display_path(multiclass_label, relative_to),
                multiclass_label_bytes=None if multiclass_label is None else multiclass_label.stat().st_size,
                image_sha1=_sha1(image) if include_hash else None,
                label_sha1=None if label is None or not include_hash else _sha1(label),
                multiclass_label_sha1=None
                if multiclass_label is None or not include_hash
                else _sha1(multiclass_label),
            )
        )
    return records


def write_manifest(records: Iterable[ManifestRecord], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(record) for record in records]
    if output.suffix.lower() == ".json":
        output.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
        return
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _has_supported_extension(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(extension) for extension in IMAGE_EXTENSIONS)


def _is_label_file(path: Path) -> bool:
    tokens: list[str] = []
    for part in path.parts:
        tokens.extend(token for token in re.split(r"[^a-z0-9]+", part.lower()) if token)
    return any(token in LABEL_TOKENS for token in tokens)


def _case_key(path: Path) -> str:
    name = _stem_without_extension(path).lower()
    tokens = [token for token in re.split(r"[^a-z0-9]+", name) if token]
    tokens = [token for token in tokens if token not in LABEL_TOKENS and token not in IMAGE_TOKENS]
    return "_".join(tokens) or path.stem.lower()


def _stem_without_extension(path: Path) -> str:
    name = path.name
    lower = name.lower()
    for extension in IMAGE_EXTENSIONS:
        if lower.endswith(extension):
            name = name[: -len(extension)]
            break
    return name


def _files_by_stem(root: Optional[Path]) -> dict[str, Path]:
    if root is None:
        return {}
    return {
        _stem_without_extension(path): path
        for path in sorted(root.rglob("*"))
        if path.is_file() and _has_supported_extension(path)
    }


def _display_path(path: Path, relative_to: Path) -> str:
    try:
        return path.resolve().relative_to(relative_to).as_posix()
    except ValueError:
        return str(path.resolve())


def _sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    sys.exit(main())
