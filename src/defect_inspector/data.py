
"""KolektorSDD2 classification dataset using a saved split manifest."""

import json
from pathlib import Path, PurePosixPath
from typing import Callable, Any

from PIL import Image
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = PROJECT_ROOT / "data/splits/ksdd2_v1.json"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data/raw/KolektorSDD2"

CLASS_TO_ID = {"good": 0, "defect": 1}


class KSDD2Dataset(Dataset):
    def __init__(
        self,
        split: str,
        transform: Callable | None = None,
        manifest_path: Path | None = None,
        data_root: Path | None = None,
    ):
        if split not in ("train", "val", "test"):
            raise ValueError(f"Invalid split: {split}")

        self.split = split
        self.transform = transform
        self.data_root = Path(data_root or DEFAULT_DATA_ROOT)

        manifest_file = Path(manifest_path or DEFAULT_MANIFEST)

        with manifest_file.open("r", encoding="utf-8") as file:
            manifest = json.load(file)

        if manifest.get("dataset") != "KolektorSDD2":
            raise ValueError("Unexpected dataset")

        if manifest.get("class_to_id") != CLASS_TO_ID:
            raise ValueError("Class mapping does not match")

        self.samples = []
        source_split = "test" if split == "test" else "train"
        seen_paths = set()

        for record in manifest["splits"][split]:
            relative = PurePosixPath(record["image"])
            parts = relative.parts

            if (
                not parts
                or relative.is_absolute()
                or ".." in parts
                or parts[0] != source_split
            ):
                raise ValueError(f"Invalid image path: {relative}")

            label = record["label"]
            label_id = record["label_id"]

            if label not in CLASS_TO_ID:
                raise ValueError(f"Unknown label: {label}")

            if label_id != CLASS_TO_ID[label]:
                raise ValueError(f"Label ID mismatch: {relative}")

            if relative in seen_paths:
                raise ValueError(f"Duplicate image path: {relative}")

            seen_paths.add(relative)

            image_path = self.data_root.joinpath(*parts)

            if not image_path.is_file():
                raise FileNotFoundError(image_path)

            self.samples.append((image_path, label_id))

        expected_count = manifest["counts"][split]["total"]

        if len(self.samples) != expected_count:
            raise ValueError("Sample count mismatch")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        image_path, label_id = self.samples[index]

        with Image.open(image_path) as file:
            image = file.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label_id
