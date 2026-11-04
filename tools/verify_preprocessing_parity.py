
import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from defect_inspector.gui import InferenceEngine
from defect_inspector.preprocessing import (
    PreprocessingConfig,
    EvalPreprocessor,
    TrainPreprocessor,
)


def max_abs_error(a, b):
    return float(np.max(np.abs(a - b)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--images", type=Path, nargs="+", required=True)
    args = parser.parse_args()

    run_dir = args.run.resolve()
    root = Path(__file__).resolve().parent.parent

    cpp_exe = (
        root
        / "cpp"
        / "build"
        / "Release"
        / "defect_inspector_cpp.exe"
    )

    model_path = run_dir / "model.onnx"
    metadata_path = run_dir / "model_metadata.json"

    for path in (cpp_exe, model_path, metadata_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    with metadata_path.open(encoding="utf-8") as file:
        metadata = json.load(file)

    config = PreprocessingConfig.from_metadata(metadata)

    eval_preprocessor = EvalPreprocessor(config)

    train_preprocessor = TrainPreprocessor(
        config,
        horizontal_flip_probability=0.0,
    )

    gui_engine = InferenceEngine(run_dir)

    session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name

    expected_shape = (
        1,
        3,
        config.height,
        config.width,
    )

    failures = []

    for image_arg in args.images:
        image_path = image_arg.resolve()

        print(f"\n=== {image_path.name} ===")

        if not image_path.is_file():
            raise FileNotFoundError(image_path)

        with Image.open(image_path) as original:
            image = original.convert("RGB")

            eval_tensor = (
                eval_preprocessor(image)
                .detach()
                .cpu()
                .numpy()
            )

            train_tensor = (
                train_preprocessor(image)
                .detach()
                .cpu()
                .numpy()
            )

        eval_tensor = np.ascontiguousarray(
            eval_tensor[None, ...],
            dtype=np.float32,
        )

        train_tensor = np.ascontiguousarray(
            train_tensor[None, ...],
            dtype=np.float32,
        )

        if eval_tensor.shape != expected_shape:
            raise ValueError(
                f"Eval shape mismatch: {eval_tensor.shape}"
            )

        if train_tensor.shape != expected_shape:
            raise ValueError(
                f"Train shape mismatch: {train_tensor.shape}"
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            cpp_tensor_path = temp / "input.f32"
            cpp_json_path = temp / "result.json"

            command = [
                str(cpp_exe),
                "--model",
                str(model_path),
                "--metadata",
                str(metadata_path),
                "--image",
                str(image_path),
                "--output-tensor",
                str(cpp_tensor_path),
                "--output-json",
                str(cpp_json_path),
            ]

            process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
            )

            if process.returncode != 0:
                raise RuntimeError(
                    "C++ inference failed:\n"
                    + process.stdout
                    + "\n"
                    + process.stderr
                )

            cpp_tensor = np.fromfile(
                cpp_tensor_path,
                dtype=np.float32,
            )

            if cpp_tensor.size != int(np.prod(expected_shape)):
                raise ValueError("C++ tensor size mismatch")

            cpp_tensor = cpp_tensor.reshape(expected_shape)

            with cpp_json_path.open(encoding="utf-8") as file:
                cpp_result = json.load(file)

        # 既存の学習・評価前処理によるTensorを使い、
        # Python ONNX Runtimeで推論する
        eval_logits = np.asarray(
            session.run(
                None,
                {input_name: eval_tensor},
            )[0],
            dtype=np.float32,
        ).reshape(-1)

        cpp_logits = np.asarray(
            cpp_result["logits"],
            dtype=np.float32,
        )

        # 実際のGUI推論エンジンを呼び出す
        gui_result = gui_engine.predict(image_path)

        gui_logits = np.asarray(
            gui_result["logits"],
            dtype=np.float32,
        )

        train_error = max_abs_error(
            train_tensor,
            eval_tensor,
        )

        tensor_error = max_abs_error(
            eval_tensor,
            cpp_tensor,
        )

        cpp_logit_error = max_abs_error(
            eval_logits,
            cpp_logits,
        )

        gui_logit_error = max_abs_error(
            eval_logits,
            gui_logits,
        )

        score_error = abs(
            gui_result["defect_score"]
            - float(cpp_result["defect_score"])
        )

        checks = {
            "Train vs Eval preprocessing": np.allclose(
                train_tensor,
                eval_tensor,
                atol=1e-5,
                rtol=0,
            ),
            "Eval vs C++ tensor": np.allclose(
                eval_tensor,
                cpp_tensor,
                atol=1e-5,
                rtol=0,
            ),
            "Eval ORT vs C++ logits": np.allclose(
                eval_logits,
                cpp_logits,
                atol=1e-4,
                rtol=1e-3,
            ),
            "Eval ORT vs GUI logits": np.allclose(
                eval_logits,
                gui_logits,
                atol=1e-4,
                rtol=1e-3,
            ),
            "GUI vs C++ defect score": (
                score_error <= 1e-5
            ),
            "GUI vs C++ decision": (
                int(gui_result["is_defect"])
                == int(cpp_result["predicted_class_id"])
            ),
        }

        print(f"Train/Eval tensor error: {train_error:.10f}")
        print(f"Eval/C++ tensor error:   {tensor_error:.10f}")
        print(f"Eval/C++ logit error:    {cpp_logit_error:.10f}")
        print(f"Eval/GUI logit error:    {gui_logit_error:.10f}")
        print(f"GUI/C++ score error:     {score_error:.10f}")

        for label, passed in checks.items():
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] {label}")

        if not all(checks.values()):
            failures.append(image_path.name)

    print("\n=== FINAL RESULT ===")
    print(f"Images tested: {len(args.images)}")
    print(f"Images passed: {len(args.images) - len(failures)}")

    if failures:
        print(f"OVERALL: FAIL ({', '.join(failures)})")
        raise SystemExit(1)

    print("OVERALL: PASS")


if __name__ == "__main__":
    main()
