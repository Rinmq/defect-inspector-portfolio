
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


def load_python_tensor(image_path: Path, metadata: dict) -> np.ndarray:
    config = metadata["input"]

    if (
        config["color_space"] != "RGB"
        or config["layout"] != "NCHW"
        or config["dtype"] != "float32"
        or config["interpolation"] != "bilinear"
        or config["resize_method"] != "stretch"
    ):
        raise ValueError("Unsupported preprocessing configuration")

    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    resized = cv2.resize(
        rgb,
        (int(config["width"]), int(config["height"])),
        interpolation=cv2.INTER_LINEAR,
    )

    image = resized.astype(np.float32)
    image *= np.float32(config["pixel_scale"])

    mean = np.asarray(config["mean"], dtype=np.float32)
    std = np.asarray(config["std"], dtype=np.float32)

    image = (image - mean) / std

    tensor = np.transpose(image, (2, 0, 1))[None, ...]

    return np.ascontiguousarray(tensor, dtype=np.float32)


def load_cpp_tensor(path: Path, shape: tuple) -> np.ndarray:
    tensor = np.fromfile(path, dtype=np.float32)

    expected_count = math.prod(shape)

    if tensor.size != expected_count:
        raise ValueError(
            f"Tensor size mismatch: "
            f"expected={expected_count}, actual={tensor.size}"
        )

    return tensor.reshape(shape)


def run_onnx(session: ort.InferenceSession, tensor: np.ndarray):
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: tensor})[0]

    logits = np.asarray(output, dtype=np.float32).reshape(-1)

    if logits.shape != (2,):
        raise ValueError(f"Unexpected logits shape: {logits.shape}")

    if not np.all(np.isfinite(logits)):
        raise ValueError("Non-finite logits")

    return logits


def defect_score(logits: np.ndarray) -> float:
    z = logits.astype(np.float64)
    z -= np.max(z)
    probabilities = np.exp(z)
    probabilities /= probabilities.sum()
    return float(probabilities[1])


def max_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    args = parser.parse_args()

    run_dir = args.run

    with (run_dir / "model_metadata.json").open(
        encoding="utf-8"
    ) as file:
        metadata = json.load(file)

    with (run_dir / "cpp_result.json").open(
        encoding="utf-8"
    ) as file:
        cpp_result = json.load(file)

    config = metadata["input"]

    shape = (
        1,
        3,
        int(config["height"]),
        int(config["width"]),
    )

    python_tensor = load_python_tensor(args.image, metadata)

    cpp_tensor = load_cpp_tensor(
        run_dir / "cpp_input.f32",
        shape,
    )

    session = ort.InferenceSession(
        str(run_dir / "model.onnx"),
        providers=["CPUExecutionProvider"],
    )

    cpp_logits = np.asarray(
        cpp_result["logits"],
        dtype=np.float32,
    )

    python_logits_same_tensor = run_onnx(
        session,
        cpp_tensor,
    )

    python_logits_python_tensor = run_onnx(
        session,
        python_tensor,
    )

    tensor_difference = max_error(python_tensor, cpp_tensor)

    same_tensor_logit_difference = max_error(
        python_logits_same_tensor,
        cpp_logits,
    )

    full_pipeline_logit_difference = max_error(
        python_logits_python_tensor,
        cpp_logits,
    )

    python_score = defect_score(python_logits_python_tensor)
    cpp_score = float(cpp_result["defect_score"])

    threshold = float(cpp_result["threshold"])

    python_decision = int(python_score >= threshold)
    cpp_decision = int(cpp_result["predicted_class_id"])

    tensor_ok = bool(
        np.allclose(
            python_tensor,
            cpp_tensor,
            atol=1e-5,
            rtol=0,
        )
    )

    same_tensor_logits_ok = bool(
        np.allclose(
            python_logits_same_tensor,
            cpp_logits,
            atol=1e-4,
            rtol=1e-3,
        )
    )

    full_pipeline_logits_ok = bool(
        np.allclose(
            python_logits_python_tensor,
            cpp_logits,
            atol=1e-4,
            rtol=1e-3,
        )
    )

    decision_ok = python_decision == cpp_decision

    print("=== Python vs C++ verification ===")
    print(f"Image: {args.image}")
    print(f"Tensor shape: {shape}")

    print("\n[1] Preprocessing")
    print(f"Max absolute error: {tensor_difference:.10f}")
    print(f"PASS: {tensor_ok}")

    print("\n[2] Same tensor / different runtimes")
    print(f"C++ logits: {cpp_logits.tolist()}")
    print(
        "Python ORT logits:",
        python_logits_same_tensor.tolist(),
    )
    print(
        f"Max absolute error: "
        f"{same_tensor_logit_difference:.10f}"
    )
    print(f"PASS: {same_tensor_logits_ok}")

    print("\n[3] Complete inference path")
    print(
        "Python logits:",
        python_logits_python_tensor.tolist(),
    )
    print(
        f"Max absolute error: "
        f"{full_pipeline_logit_difference:.10f}"
    )
    print(f"Python defect score: {python_score:.10f}")
    print(f"C++ defect score:    {cpp_score:.10f}")
    print(f"Python predicted ID: {python_decision}")
    print(f"C++ predicted ID:    {cpp_decision}")
    print(f"Logits PASS: {full_pipeline_logits_ok}")
    print(f"Decision PASS: {decision_ok}")

    all_passed = all(
        [
            tensor_ok,
            same_tensor_logits_ok,
            full_pipeline_logits_ok,
            decision_ok,
        ]
    )

    print("\nOVERALL:", "PASS" if all_passed else "FAIL")

    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
