
"""Train ResNet18 and select the best checkpoint using validation AP."""

import argparse
import hashlib
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torchvision
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn
from torch.utils.data import DataLoader

from defect_inspector.data import KSDD2Dataset, DEFAULT_MANIFEST
from defect_inspector.model import build_model
from defect_inspector.preprocessing import (
    EvalPreprocessor,
    PreprocessingConfig,
    TrainPreprocessor,
)

ROOT = Path(__file__).resolve().parents[2]
SEED = 42
THRESHOLD = 0.5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path: Path, value: dict | list) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def train_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0.0
    total_examples = 0

    for batch_index, (images, labels) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, labels)

        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError(
                f"Non-finite training loss at epoch {epoch}"
            )

        loss.backward()
        optimizer.step()

        count = labels.size(0)
        total_examples += count
        total_loss += loss.item() * count

        if batch_index % 50 == 0 or batch_index == len(loader):
            print(
                f"  epoch={epoch} "
                f"batch={batch_index}/{len(loader)} "
                f"train_loss={total_loss / total_examples:.4f}",
                flush=True,
            )

    return total_loss / total_examples


@torch.inference_mode()
def validate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_examples = 0
    true_labels = []
    defect_scores = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError("Non-finite validation loss")

        count = labels.size(0)
        total_examples += count
        total_loss += loss.item() * count

        true_labels.extend(labels.cpu().tolist())
        defect_scores.extend(
            torch.softmax(logits, dim=1)[:, 1].cpu().tolist()
        )

    predicted = [
        int(score >= THRESHOLD)
        for score in defect_scores
    ]

    tn, fp, fn, tp = (
        int(value)
        for value in confusion_matrix(
            true_labels, predicted, labels=[0, 1]
        ).ravel()
    )

    metrics = {
        "loss": total_loss / total_examples,
        "average_precision": float(
            average_precision_score(true_labels, defect_scores)
        ),
        "accuracy": float(
            accuracy_score(true_labels, predicted)
        ),
        "defect_precision": float(
            precision_score(
                true_labels, predicted, zero_division=0
            )
        ),
        "defect_recall": float(
            recall_score(
                true_labels, predicted, zero_division=0
            )
        ),
        "defect_f1": float(
            f1_score(true_labels, predicted, zero_division=0)
        ),
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        },
        "threshold": THRESHOLD,
    }

    return metrics, true_labels, defect_scores


def main():
    parser = argparse.ArgumentParser(
        description="Train ResNet18 on KolektorSDD2"
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    if args.epochs <= 0 or args.batch_size <= 0 or args.lr <= 0:
        parser.error(
            "epochs, batch-size and lr must be positive"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required for this training run"
        )

    device = torch.device("cuda")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    preprocessing = PreprocessingConfig()

    train_dataset = KSDD2Dataset(
        "train",
        transform=TrainPreprocessor(
            preprocessing,
            horizontal_flip_probability=0.0,
        ),
    )

    val_dataset = KSDD2Dataset(
        "val",
        transform=EvalPreprocessor(preprocessing),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        generator=torch.Generator().manual_seed(SEED),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    run_id = datetime.now().strftime(
        "resnet18_ksdd2_%Y%m%d_%H%M%S_%f"
    )
    run_dir = ROOT / "artifacts" / "experiments" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    configuration = {
        "seed": SEED,
        "dataset": "KolektorSDD2",
        "split_manifest": "data/splits/ksdd2_v1.json",
        "split_manifest_sha256": sha256_file(DEFAULT_MANIFEST),
        "official_test_used_during_training": False,
        "related_image_groups_verified": False,
        "architecture": "resnet18",
        "initial_weights": "IMAGENET1K_V1",
        "fine_tuning": "all_layers",
        "loss": "CrossEntropyLoss (unweighted)",
        "optimizer": "AdamW",
        "learning_rate": args.lr,
        "weight_decay": 0.01,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "train_horizontal_flip_probability": 0.0,
        "model_selection": (
            "highest validation average_precision "
            "(ties: earlier epoch)"
        ),
        "defect_threshold": THRESHOLD,
        "input": preprocessing.to_metadata(),
        "pytorch": torch.__version__,
        "torchvision": torchvision.__version__,
        "gpu": torch.cuda.get_device_name(0),
    }

    save_json(
        run_dir / "experiment_config.json",
        configuration,
    )

    print(f"Run directory: {run_dir}", flush=True)
    print(
        f"Train: {len(train_dataset)} / "
        f"Validation: {len(val_dataset)}",
        flush=True,
    )

    model = build_model(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01,
    )

    best_ap = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            epoch,
        )

        val_metrics, true_labels, defect_scores = validate(
            model,
            val_loader,
            criterion,
            device,
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val": val_metrics,
        })

        save_json(run_dir / "history.json", history)

        print(
            f"Epoch {epoch}/{args.epochs} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_AP={val_metrics['average_precision']:.4f} "
            f"val_F1={val_metrics['defect_f1']:.4f} "
            f"FN={val_metrics['confusion_matrix']['fn']} "
            f"FP={val_metrics['confusion_matrix']['fp']}",
            flush=True,
        )

        if val_metrics["average_precision"] > best_ap:
            best_ap = val_metrics["average_precision"]

            state_on_cpu = {
                name: tensor.detach().cpu()
                for name, tensor in model.state_dict().items()
            }

            torch.save(
                {
                    "model_state_dict": state_on_cpu,
                    "epoch": epoch,
                    "val_ap": best_ap,
                },
                run_dir / "best_model.pth",
            )

            metadata = {
                "schema_version": 1,
                "architecture": "resnet18",
                "classes": {
                    "0": "good",
                    "1": "defect",
                },
                "defect_class_id": 1,
                "input": preprocessing.to_metadata(),
                "output_type": "logits",
                "defect_threshold": THRESHOLD,
                "selected_epoch": epoch,
                "selection_metric": (
                    "validation_average_precision"
                ),
                "validation_metrics": val_metrics,
            }

            save_json(
                run_dir / "model_metadata.json",
                metadata,
            )

            predictions = []

            for (image_path, _), true_label, score in zip(
                val_dataset.samples,
                true_labels,
                defect_scores,
                strict=True,
            ):
                prediction = int(score >= THRESHOLD)

                predictions.append({
                    "image": image_path.relative_to(
                        val_dataset.data_root
                    ).as_posix(),
                    "true_label": true_label,
                    "predicted_label": prediction,
                    "defect_score": score,
                    "correct": prediction == true_label,
                })

            save_json(
                run_dir / "best_validation_predictions.json",
                predictions,
            )

            print(
                f"  Saved new best checkpoint: "
                f"epoch={epoch}, AP={best_ap:.4f}",
                flush=True,
            )

    print(
        f"Training complete. Best validation AP: {best_ap:.4f}",
        flush=True,
    )
    print(f"Artifacts: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
