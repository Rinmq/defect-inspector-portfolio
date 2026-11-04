
import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np


PANEL_W = 180
PANEL_H = 350
CARD_W = 390
CARD_H = 425
COLUMNS = 4


def read_image(path: Path, flags: int) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)

    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, flags)

    if image is None:
        raise ValueError(f"Cannot decode: {path}")

    return image


def fit_to_panel(
    image: np.ndarray,
    width: int,
    height: int,
    interpolation: int,
) -> np.ndarray:
    original_h, original_w = image.shape[:2]

    scale = min(
        width / original_w,
        height / original_h,
    )

    new_w = max(1, round(original_w * scale))
    new_h = max(1, round(original_h * scale))

    resized = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=interpolation,
    )

    if image.ndim == 3:
        canvas = np.full(
            (height, width, 3),
            35,
            dtype=np.uint8,
        )
    else:
        canvas = np.zeros(
            (height, width),
            dtype=np.uint8,
        )

    x = (width - new_w) // 2
    y = (height - new_h) // 2

    canvas[y:y + new_h, x:x + new_w] = resized

    return canvas


def build_card(
    image_path: Path,
    mask_path: Path,
    score: float,
    category: str,
) -> np.ndarray:
    image = read_image(
        image_path,
        cv2.IMREAD_COLOR,
    )

    mask = read_image(
        mask_path,
        cv2.IMREAD_GRAYSCALE,
    )

    if image.shape[:2] != mask.shape:
        raise ValueError(
            f"Image/mask size mismatch: {image_path}"
        )

    original = fit_to_panel(
        image,
        PANEL_W,
        PANEL_H,
        cv2.INTER_AREA,
    )

    display_mask = fit_to_panel(
        mask,
        PANEL_W,
        PANEL_H,
        cv2.INTER_NEAREST,
    )

    overlay = original.copy()
    selected = display_mask > 0

    if np.any(selected):
        red = np.array([0, 0, 255], dtype=np.float32)

        overlay[selected] = (
            overlay[selected].astype(np.float32) * 0.4
            + red * 0.6
        ).astype(np.uint8)

        count, labels, stats, centers = (
            cv2.connectedComponentsWithStats(
                selected.astype(np.uint8),
                connectivity=8,
            )
        )

        components = sorted(
            range(1, count),
            key=lambda i: int(
                stats[i, cv2.CC_STAT_AREA]
            ),
            reverse=True,
        )

        for component in components[:5]:
            cx, cy = centers[component]

            cv2.circle(
                overlay,
                (round(cx), round(cy)),
                12,
                (0, 255, 255),
                2,
            )

    card = np.full(
        (CARD_H, CARD_W, 3),
        22,
        dtype=np.uint8,
    )

    cv2.putText(
        card,
        f"{category}  {image_path.name}",
        (10, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (240, 240, 240),
        2,
    )

    cv2.putText(
        card,
        f"Defect score: {score:.5f}",
        (10, 47),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.53,
        (180, 210, 255),
        1,
    )

    card[65:65 + PANEL_H, 10:10 + PANEL_W] = (
        original
    )

    card[65:65 + PANEL_H, 200:200 + PANEL_W] = (
        overlay
    )

    return card


def create_sheet(
    rows: list[dict],
    test_dir: Path,
    output_path: Path,
    category: str,
) -> None:
    if not rows:
        print(f"No {category} images")
        return

    rows_count = math.ceil(len(rows) / COLUMNS)

    sheet = np.full(
        (
            rows_count * CARD_H + 48,
            COLUMNS * CARD_W,
            3,
        ),
        18,
        dtype=np.uint8,
    )

    cv2.putText(
        sheet,
        f"{category}: {len(rows)} images | "
        "Red/yellow = ground truth, NOT model localization",
        (12, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (235, 235, 235),
        1,
    )

    for index, row in enumerate(rows):
        name = row["image"]
        image_path = test_dir / name
        mask_path = test_dir / (
            f"{Path(name).stem}_GT.png"
        )

        card = build_card(
            image_path,
            mask_path,
            float(row["defect_score"]),
            category,
        )

        grid_y = index // COLUMNS
        grid_x = index % COLUMNS

        y = 48 + grid_y * CARD_H
        x = grid_x * CARD_W

        sheet[
            y:y + CARD_H,
            x:x + CARD_W,
        ] = card

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    success, encoded = cv2.imencode(
        ".png",
        sheet,
    )

    if not success:
        raise RuntimeError("PNG encoding failed")

    encoded.tofile(str(output_path))

    print(
        f"Saved {category} ({len(rows)}): {output_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--test-dir",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    run_dir = args.run.resolve()
    test_dir = args.test_dir.resolve()

    with (run_dir / "test_predictions.csv").open(
        newline="",
        encoding="utf-8",
    ) as file:
        rows = list(csv.DictReader(file))

    false_negatives = [
        row
        for row in rows
        if row["true_label"] == "1"
        and row["predicted_label"] == "0"
    ]

    false_positives = [
        row
        for row in rows
        if row["true_label"] == "0"
        and row["predicted_label"] == "1"
    ]

    false_negatives.sort(
        key=lambda row: float(row["defect_score"])
    )

    false_positives.sort(
        key=lambda row: float(row["defect_score"]),
        reverse=True,
    )

    output_dir = run_dir / "error_analysis"

    create_sheet(
        false_negatives,
        test_dir,
        output_dir / "false_negatives.png",
        "FN",
    )

    create_sheet(
        false_positives,
        test_dir,
        output_dir / "false_positives.png",
        "FP",
    )


if __name__ == "__main__":
    main()
