
# Defect Inspector

PyTorchで学習したResNet18をONNXへ変換し、
Python GUIおよびC++ CLIから正常・欠陥を分類するアプリケーション。

## 1. 概要

工業製品の表面画像を入力し、次の2クラスを判定する。

- `0`: good（正常）
- `1`: defect（欠陥）

使用データセット：KolektorSDD2

本プロジェクトは画像単位の2クラス分類を目的とする。
欠陥位置の検出やセグメンテーションは実装していない。

## 2. 技術構成

| 用途 | 技術 |
|---|---|
| 学習 | Python 3.12 / PyTorch |
| モデル | ResNet18 |
| データ処理 | OpenCV / NumPy |
| モデル交換 | ONNX Opset 18 |
| Python推論 | ONNX Runtime 1.31.1 |
| C++推論 | C++17 / ONNX Runtime 1.31.1 |
| C++画像処理 | OpenCV 4.14.0 |
| GUI | PySide6 |
| C++ビルド | CMake / MSVC / vcpkg |

## 3. データセット

KolektorSDD2の公式Trainを層化分割し、
公式Testは分割を変更せずに保持した。

| Split | Good | Defect | Total |
|---|---:|---:|---:|
| Train | 1667 | 197 | 1864 |
| Validation | 418 | 49 | 467 |
| Test | 894 | 110 | 1004 |

総画像数：3335

データ監査では、同一内容のコピー2ファイルを除外。
最終的な監査エラーは0件。

分割間の完全重複は検出されていない。
ただし、関連画像のグループ独立性は未検証。

### データセットの入手と配置

データセットは本リポジトリに同梱していない。

公式配布先：
https://www.vicos.si/resources/kolektorsdd2/

ダウンロードした画像と対応する`_GT.png`マスクを
`data/raw/KolektorSDD2/train/`および
`data/raw/KolektorSDD2/test/`に配置する。

`data/raw/`はGit管理対象外。

### データセットの利用条件

KolektorSDD2のライセンスはCC BY-NC-SA 4.0。
データセットの出典を明記し、非商用利用などの条件に従う。

引用論文：
Božič, Tabernik, Skočaj,
"Mixed supervision for surface-defect detection:
from weakly to fully supervised learning,"
Computers in Industry, 2021.

学習済みモデルの再配布条件については別途確認が必要。

## 4. モデルと前処理

モデル：ImageNet事前学習済みResNet18

最終層を512次元から2クラスへ変更し、
全層をFine-tuningした。

学習条件：

- Epoch: 12
- Batch size: 8
- Optimizer: AdamW
- Learning rate: 0.0001
- Loss: CrossEntropyLoss
- モデル選択基準: Validation AP

最良チェックポイント：Epoch 11

最良Validation AP：0.9733

前処理：

- RGB
- 高さ608、幅224
- OpenCV INTER_LINEARでリサイズ
- float32
- NCHW
- 画素値を255で除算
- ImageNet mean/stdで正規化

前処理設定はmodel_metadata.jsonで管理する。

## 5. 主要ファイル

### Python

- `src/defect_inspector/data.py`
- `src/defect_inspector/preprocessing.py`
- `src/defect_inspector/model.py`
- `src/defect_inspector/train.py`
- `src/defect_inspector/export_onnx.py`
- `src/defect_inspector/gui.py`

### C++

- `cpp/vcpkg.json`
- `cpp/CMakeLists.txt`
- `cpp/src/main.cpp`

### 検証ツール

- `tools/audit_dataset.py`
- `tools/prepare_splits.py`
- `tools/compare_cpp_python.py`
- `tools/evaluate_test.py`
- `tools/visualize_test_errors.py`
- `tools/verify_preprocessing_parity.py`

## 6. Python環境

Windows PowerShellを使用する。

プロジェクトルートから実行：

    py -3.12 -m venv .venv

    .\.venv\Scripts\python.exe --version

必要な依存関係はpyproject.tomlに定義している。

環境構築時は仮想環境を作成し、
プロジェクトをeditable installする。

    .\.venv\Scripts\python.exe -m pip install -e ".[dev]"

学習にはCUDA対応PyTorchとGPUを使用した。
推論はCPU版ONNX Runtimeで動作する。

### 学習済みモデルの配布について

学習済みの`model.onnx`は、再配布条件の確認が完了するまで
本リポジトリに含めていない。

GUIとC++ CLIの実行には、データセットを別途取得し、
ローカルでモデルを学習・ONNX形式へエクスポートする必要がある。

以下の起動例は、対応する`model.onnx`を
実験フォルダーに配置した環境を想定している。
## 7. Python GUIの起動

PowerShellで実行：

    # リポジトリのルートで実行

    $run = Join-Path (Get-Location).Path "artifacts\experiments\resnet18_ksdd2_20261102_070156_312941"

    .\.venv\Scripts\python.exe -m defect_inspector.gui --run "$run"

GUIから画像を選択して推論できる。

確認済みの機能：

- 画像プレビュー
- 正常・欠陥判定
- 欠陥スコア表示
- logit表示
- 判定閾値表示
- 処理時間表示

正常画像・欠陥画像の両方で動作確認済み。

## 8. C++ CLI

必要なもの：

- Visual Studio C++開発環境
- CMake
- vcpkg
- OpenCV
- nlohmann/json
- ONNX Runtime C++ SDK 1.31.1

C++用ONNX Runtime SDKは本リポジトリに同梱していない。

公式リリース：
https://github.com/microsoft/onnxruntime/releases/tag/v1.31.1

Windows x64 CPU用の
`onnxruntime-win-x64-1.31.1.zip`を取得して展開し、
次のファイルが存在するよう配置する。

- `external/onnxruntime-win-x64-1.31.1/include/onnxruntime_cxx_api.h`
- `external/onnxruntime-win-x64-1.31.1/lib/onnxruntime.lib`
- `external/onnxruntime-win-x64-1.31.1/lib/onnxruntime.dll`

CMakeの`ONNXRUNTIME_ROOT`で別の配置先も指定できる。
また、ビルド例の`$HOME/vcpkg`は各環境のパスに変更する。

C++のビルド：

    # リポジトリのルートで実行

    cmake -S .\cpp -B .\cpp\build -G "Visual Studio 18 2026" -A x64 "-DCMAKE_TOOLCHAIN_FILE=$HOME/vcpkg/scripts/buildsystems/vcpkg.cmake"

    cmake --build .\cpp\build --config Release --parallel 4

生成された実行ファイル：

`cpp/build/Release/defect_inspector_cpp.exe`

画像1枚の推論例：

    # リポジトリのルートで実行

    $run = Join-Path (Get-Location).Path "artifacts\experiments\resnet18_ksdd2_20261102_070156_312941"

    .\cpp\build\Release\defect_inspector_cpp.exe --model "$run\model.onnx" --metadata "$run\model_metadata.json" --image ".\data\raw\KolektorSDD2\train\10004.png"

JSON形式でlogit、欠陥スコア、判定を出力する。

## 9. 公式Test評価

評価対象：KolektorSDD2公式Test 1004枚

判定閾値：0.5

| Metric | Result |
|---|---:|
| Accuracy | 0.9741 |
| Average Precision | 0.9074 |
| ROC AUC | 0.9574 |
| Precision | 0.9565 |
| Recall | 0.8000 |
| F1 | 0.8713 |

混同行列：

| | Predicted Good | Predicted Defect |
|---|---:|---:|
| Actual Good | 890 | 4 |
| Actual Defect | 22 | 88 |

欠陥110枚のうち22枚を見逃した。
したがって、見逃し率は20%。

Accuracyだけで品質を判断せず、
Recall・FN・FPも評価する必要がある。

評価の再実行：

    .\.venv\Scripts\python.exe .\tools\evaluate_test.py --run "$run" --test-dir ".\data\raw\KolektorSDD2\test"

評価結果：

- `test_evaluation.json`：リポジトリに収録済み
- `test_predictions.csv`：評価時に生成。リポジトリには未収録

評価の再実行にはKolektorSDD2公式Test画像が必要。

## 10. Python/C++整合性検証

両実装で同じONNXモデルを使用する。

検証内容：

1. 前処理Tensorの比較
2. 同一Tensorでの推論logit比較
3. 画像から推論までのlogit比較
4. 欠陥スコアと最終判定の比較

確認済み：

- TN：3画像PASS
- TP：3画像PASS
- FN：3画像PASS
- FP：3画像PASS

合計：12 / 12 PASS

さらに4画像で既存EvalPreprocessorと
C++・GUIの推論経路を比較し、4 / 4 PASS。

これらは検証対象画像における一致を示すもので、
全画像での完全一致を保証するものではない。

## 11. 誤分類分析

公式Testの誤分類26枚を可視化した。

- False Negative：22枚
- False Positive：4枚

可視化結果（リポジトリには未収録）：

- `error_analysis/false_negatives.png`
- `error_analysis/false_positives.png`

データセット配置後に公式Test評価を再実行し、
次のコマンドで誤分類画像を生成できる。

    .\.venv\Scripts\python.exe .\tools\visualize_test_errors.py --run "$run" --test-dir ".\data\raw\KolektorSDD2\test"

画像中の赤い領域は正解マスクであり、
モデルが予測した欠陥位置ではない。

見逃し画像には、小さな点状欠陥、
細い傷、端付近の欠陥が複数見られた。

ただし、比較的大きな欠陥の見逃しもある。

欠陥の面積、リサイズ、背景テクスチャ、
クラス不均衡などの影響が考えられるが、
原因は確定していない。

赤いマスクはデータセットの正解領域であり、
ResNet18が予測した欠陥位置ではない。

## 12. 制約と今後の改善

現在の実装は研究・学習目的のプロトタイプ。

主な制約：

- 欠陥の見逃しが22件ある
- 画像内の欠陥位置は出力できない
- グループ単位のデータ独立性は未検証
- Python/C++の一致検証は一部画像のみ
- CPU推論速度の体系的な性能評価は未実施
- 本番環境での品質・安全性は未検証

改善候補：

- Validationデータを使った閾値検討
- 入力解像度・前処理の検証
- クラス不均衡への対処
- 小欠陥に対応するモデル構造の検討
- 欠陥検出・セグメンテーションへの拡張
- 独立した評価データでの再検証

Test画像の分析を基にモデルを改良した場合、
同じTestでの再評価を独立した最終評価とは扱わない。

## 13. 現在のステータス

Defect Inspectorの画像分類プロトタイプとして、
主要機能の実装・動作確認・基本評価を完了。

工業現場への実運用を保証するものではない。

## 14. 学習からGUI起動までの再現手順

### 前提条件

- Windows
- Python 3.12
- NVIDIA CUDA対応GPU
- KolektorSDD2データセット

データセットの公式配布先：
https://www.vicos.si/resources/kolektorsdd2/

公式Train/Testを次の場所に配置する。

- `data/raw/KolektorSDD2/train/`
- `data/raw/KolektorSDD2/test/`

リポジトリには分割定義
`data/splits/ksdd2_v1.json`が含まれている。
画像ファイルのパスは、この定義と一致させる必要がある。

### 1. Python環境の構築

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

### CUDA対応PyTorchの導入

通常の`pip install -e ".[dev]"`ではCPU版PyTorchが
インストールされる場合がある。

NVIDIA GPUで学習する場合は、次のコマンドを追加で実行する。

```powershell
.\.venv\Scripts\python.exe -m pip install --force-reinstall --no-deps "torch==2.11.0+cu128" "torchvision==0.26.0+cu128" --index-url https://download.pytorch.org/whl/cu128

.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

CUDAが正しく認識されれば`2.11.0+cu128 True`と表示される。

インストールには数GBの空き容量が必要。

### 2. モデルの学習

```powershell
.\.venv\Scripts\python.exe -m defect_inspector.train --epochs 12 --batch-size 8 --lr 0.0001
```

学習完了後、`artifacts/experiments/`に
新しい実験フォルダーが作成される。

### 3. ONNX形式に変換

```powershell
$run = Get-ChildItem .\artifacts\experiments -Directory |
    Where-Object { Test-Path (Join-Path $_.FullName "best_model.pth") } |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1 -ExpandProperty FullName

.\.venv\Scripts\python.exe -m defect_inspector.export_onnx --run-dir "$run"
```

### 4. GUIを起動

```powershell
.\.venv\Scripts\python.exe -m defect_inspector.gui --run "$run"
```

GUIから画像を選択して、正常・欠陥を分類する。

### 注意

- 学習済みONNXモデルは配布していない。
- 学習にはCUDA対応GPUが必要。
- GPUやライブラリの環境によって結果が変わる可能性がある。
- この再現手順は第三者のクリーン環境では未検証。
