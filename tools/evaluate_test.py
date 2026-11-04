
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from defect_inspector.gui import InferenceEngine


def read_mask(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Missing mask: {path}")

    data = np.fromfile(str(path), dtype=np.uint8)
    mask = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise ValueError(f"Cannot decode mask: {path}")

    return mask


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    run_dir = args.run.resolve()
    test_dir = args.test_dir.resolve()

    if not test_dir.is_dir():
        raise FileNotFoundError(test_dir)

    images = sorted(
        path
        for path in test_dir.glob("*.png")
        if not path.stem.endswith("_GT")
    )

    if len(images) != 1004:
        raise ValueError(
            f"Expected 1004 test images, found {len(images)}"
        )

    engine = InferenceEngine(run_dir)

    y_true = []
    y_pred = []
    y_scores = []
    rows = []

    print(f"Test images: {len(images)}")
    print(f"Threshold: {engine.threshold:.4f}")

    for index, image_path in enumerate(images, start=1):
        mask_path = image_path.with_name(
            f"{image_path.stem}_GT.png"
        )

        mask = read_mask(mask_path)
        true_label = int(np.any(mask != 0))

        result = engine.predict(image_path)

        score = float(result["defect_score"])
        predicted_label = int(result["is_defect"])

        if not np.isfinite(score):
            raise ValueError(
                f"Invalid score: {image_path}"
            )

        y_true.append(true_label)
        y_pred.append(predicted_label)
        y_scores.append(score)

        rows.append(
            {
                "image": image_path.name,
                "true_label": true_label,
                "predicted_label": predicted_label,
                "defect_score": score,
                "correct": int(true_label == predicted_label),
                "good_logit": result["logits"][0],
                "defect_logit": result["logits"][1],
                "elapsed_ms": result["elapsed_ms"],
            }
        )

        if index % 100 == 0 or index == len(images):
            print(f"Processed: {index}/{len(images)}")

    good_count = y_true.count(0)
    defect_count = y_true.count(1)

    if (good_count, defect_count) != (894, 110):
        raise ValueError(
            "Unexpected test label distribution: "
            f"good={good_count}, defect={defect_count}"
        )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    metrics = {
        "dataset": "KolektorSDD2",
        "split": "official_test",
        "total_images": len(images),
        "good_images": good_count,
        "defect_images": defect_count,
        "threshold": engine.threshold,
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "average_precision": float(
            average_precision_score(y_true, y_scores)
        ),
        "roc_auc": float(
            roc_auc_score(y_true, y_scores)
        ),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, y_pred, zero_division=0)
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "false_negative_images": [
            row["image"]
            for row in rows
            if row["true_label"] == 1
            and row["predicted_label"] == 0
        ],
        "false_positive_images": [
            row["image"]
            for row in rows
            if row["true_label"] == 0
            and row["predicted_label"] == 1
        ],
    }

    csv_path = run_dir / "test_predictions.csv"
    json_path = run_dir / "test_evaluation.json"

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\n=== OFFICIAL TEST RESULTS ===")
    print(f"Images: {len(images)}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"AP:        {metrics['average_precision']:.4f}")
    print(f"ROC AUC:   {metrics['roc_auc']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1:        {metrics['f1']:.4f}")

    print("\nConfusion matrix:")
    print(f"TN={tn} FP={fp}")
    print(f"FN={fn} TP={tp}")

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
