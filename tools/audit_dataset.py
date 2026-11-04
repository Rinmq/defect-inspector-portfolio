
"""KolektorSDD2 read-only audit. Ignore only byte-identical copy files."""

import hashlib
import json
from collections import Counter
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/raw/KolektorSDD2"
REPORT = ROOT / "artifacts/dataset_audit.json"


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scan_split(split):
    folder = DATASET / split
    if not folder.is_dir():
        raise FileNotFoundError(folder)

    paths = sorted(p for p in folder.rglob("*")
                   if p.is_file() and p.suffix.lower() == ".png")
    valid_paths, ignored_copies, issues = [], [], []

    for path in paths:
        if not path.stem.lower().endswith(" (copy)"):
            valid_paths.append(path)
            continue

        original = path.with_name(path.stem[:-7] + path.suffix)

        if original.is_file() and file_hash(path) == file_hash(original):
            ignored_copies.append({
                "copy": path.relative_to(DATASET).as_posix(),
                "original": original.relative_to(DATASET).as_posix(),
            })
        else:
            issues.append(f"Unverified copy: {path.relative_to(DATASET)}")
            valid_paths.append(path)

    images, masks = {}, {}

    for path in valid_paths:
        stem = path.stem.lower()
        is_mask = stem.endswith(("_gt", "_mask"))

        if is_mask:
            stem = stem[:-3] if stem.endswith("_gt") else stem[:-5]

        parent = path.parent.relative_to(folder).as_posix().lower()
        key = (parent, stem)
        target = masks if is_mask else images

        if key in target:
            issues.append(f"Duplicate key: {path.relative_to(DATASET)}")
        else:
            target[key] = path

    counts, sizes, records = Counter(), Counter(), []

    for key, image in images.items():
        mask = masks.get(key)
        image_name = image.relative_to(DATASET).as_posix()
        mask_name = mask.relative_to(DATASET).as_posix() if mask else None

        label = None
        width = height = None

        if mask is None:
            issues.append(f"Missing mask: {image_name}")
            counts["missing_masks"] += 1

        try:
            with Image.open(image) as photo:
                width, height = photo.size
                sizes[f"{width}x{height}"] += 1

            if mask is not None:
                with Image.open(mask) as gt:
                    if gt.size != (width, height):
                        issues.append(f"Size mismatch: {image_name}")
                        counts["size_mismatches"] += 1
                    else:
                        label = "defect" if gt.convert("L").getbbox() else "good"
                        counts[label] += 1

        except (OSError, ValueError) as exc:
            issues.append(f"Unreadable: {image_name}: {exc}")
            counts["unreadable"] += 1

        records.append({
            "image": image_name,
            "mask": mask_name,
            "label": label,
            "width": width,
            "height": height,
        })

    for key, mask in masks.items():
        if key not in images:
            issues.append(f"Orphan mask: {mask.relative_to(DATASET)}")
            counts["orphan_masks"] += 1

    summary = {
        "images": len(images),
        "masks": len(masks),
        "good": counts["good"],
        "defect": counts["defect"],
        "ignored_verified_copies": len(ignored_copies),
        "missing_masks": counts["missing_masks"],
        "orphan_masks": counts["orphan_masks"],
        "size_mismatches": counts["size_mismatches"],
        "unreadable": counts["unreadable"],
        "common_image_sizes": sizes.most_common(10),
    }
    return summary, records, ignored_copies, issues


def main():
    if not DATASET.is_dir():
        raise FileNotFoundError(DATASET)

    splits, records, ignored_copies, issues = {}, [], [], []

    for name in ("train", "test"):
        summary, split_records, copies, split_issues = scan_split(name)

        splits[name] = summary
        records.extend(split_records)
        ignored_copies.extend(copies)
        issues.extend(split_issues)

        print(
            f"[{name}] images={summary['images']} "
            f"masks={summary['masks']} "
            f"good={summary['good']} defect={summary['defect']}"
        )
        print(f"  common sizes: {summary['common_image_sizes'][:3]}")
        print(
            f"  ignored copies={summary['ignored_verified_copies']}, "
            f"missing masks={summary['missing_masks']}, "
            f"orphan masks={summary['orphan_masks']}, "
            f"size mismatches={summary['size_mismatches']}, "
            f"unreadable={summary['unreadable']}"
        )

    report = {
        "dataset": "KolektorSDD2",
        "label_rule": "GT mask has nonzero pixel => defect; otherwise good",
        "official_test_preserved": True,
        "weak_split_files": sorted(
            p.name for p in DATASET.glob("split_weakly_*.pyb")
        ),
        "splits": splits,
        "ignored_verified_copies": ignored_copies,
        "issues": issues,
        "records": records,
    }

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Total image records: {len(records)}")
    print(f"Verified copies ignored: {len(ignored_copies)}")
    print(f"Issues: {len(issues)}")
    print(f"Report: {REPORT}")

    for issue in issues[:10]:
        print(f"  - {issue}")


if __name__ == "__main__":
    main()
