
"""Create reproducible classification splits from an audited KSDD2 dataset."""

import hashlib
import json
from collections import Counter
from pathlib import Path, PurePosixPath

from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "raw" / "KolektorSDD2"
AUDIT = ROOT / "artifacts" / "dataset_audit.json"
OUTPUT = ROOT / "data" / "splits" / "ksdd2_v1.json"

SEED = 42
VALIDATION_RATIO = 0.20
CLASS_TO_ID = {"good": 0, "defect": 1}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def validate_record(record: dict, source: str) -> dict:
    image_name = record["image"]
    mask_name = record["mask"]
    label = record["label"]

    if label not in CLASS_TO_ID:
        raise ValueError(f"Invalid label: {image_name}: {label}")

    if not mask_name:
        raise ValueError(f"Missing mask: {image_name}")

    for relative_name in (image_name, mask_name):
        relative = PurePosixPath(relative_name)

        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.parts[0] != source
        ):
            raise ValueError(f"Invalid path: {relative_name}")

        if not (DATASET / relative_name).is_file():
            raise FileNotFoundError(DATASET / relative_name)

    return {
        "image": image_name,
        "mask": mask_name,
        "label": label,
        "label_id": CLASS_TO_ID[label],
        "width": record["width"],
        "height": record["height"],
    }


def count_classes(records: list[dict]) -> dict:
    counts = Counter(item["label"] for item in records)

    return {
        "total": len(records),
        "good": counts["good"],
        "defect": counts["defect"],
    }


def check_exact_duplicates(splits: dict) -> None:
    hashes = {}

    for split_name, records in splits.items():
        for record in records:
            image_path = DATASET / record["image"]
            digest = sha256_file(image_path)

            if digest in hashes:
                previous_split, previous_image = hashes[digest]

                if previous_split != split_name:
                    raise ValueError(
                        "Identical images cross split boundaries: "
                        f"{previous_split}/{previous_image} and "
                        f"{split_name}/{record['image']}"
                    )
            else:
                hashes[digest] = (split_name, record["image"])


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(
            f"Split manifest already exists: {OUTPUT}. "
            "Do not overwrite an existing experiment split."
        )

    audit = json.loads(AUDIT.read_text(encoding="utf-8"))

    if audit["issues"]:
        raise ValueError("Dataset audit contains unresolved issues.")

    if audit["dataset"] != "KolektorSDD2":
        raise ValueError("Unexpected dataset.")

    original_train = []
    original_test = []

    for record in audit["records"]:
        source = PurePosixPath(record["image"]).parts[0]

        if source == "train":
            original_train.append(validate_record(record, "train"))
        elif source == "test":
            original_test.append(validate_record(record, "test"))
        else:
            raise ValueError(f"Unexpected source split: {source}")

    if len(original_train) != 2331 or len(original_test) != 1004:
        raise ValueError("Dataset counts differ from the audited baseline.")

    all_images = [item["image"] for item in original_train + original_test]

    if len(all_images) != len(set(all_images)):
        raise ValueError("Duplicate image paths detected.")

    train_records, val_records = train_test_split(
        original_train,
        test_size=VALIDATION_RATIO,
        random_state=SEED,
        stratify=[item["label"] for item in original_train],
    )

    splits = {
        "train": sorted(train_records, key=lambda x: x["image"]),
        "val": sorted(val_records, key=lambda x: x["image"]),
        "test": sorted(original_test, key=lambda x: x["image"]),
    }

    train_names = {item["image"] for item in splits["train"]}
    val_names = {item["image"] for item in splits["val"]}
    test_names = {item["image"] for item in splits["test"]}

    if train_names & val_names or train_names & test_names or val_names & test_names:
        raise ValueError("Split overlap detected.")

    check_exact_duplicates(splits)

    manifest = {
        "schema_version": 1,
        "dataset": "KolektorSDD2",
        "source_audit_sha256": sha256_file(AUDIT),
        "seed": SEED,
        "validation_ratio": VALIDATION_RATIO,
        "split_method": "stratified_train_test_split",
        "class_to_id": CLASS_TO_ID,
        "defect_class_id": 1,
        "grouping": {
            "verified": False,
            "reason": (
                "Product and capture-sequence identifiers "
                "have not been established."
            ),
            "limitation": (
                "Exact duplicate images are checked, but "
                "related or near-duplicate images may still "
                "occur across splits."
            ),
        },
        "official_test_preserved": True,
        "counts": {
            name: count_classes(records)
            for name, records in splits.items()
        },
        "splits": splits,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    for name, counts in manifest["counts"].items():
        print(
            f"{name}: total={counts['total']} "
            f"good={counts['good']} "
            f"defect={counts['defect']}"
        )

    print("Exact cross-split duplicates: 0")
    print("Grouping verified: False")
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
