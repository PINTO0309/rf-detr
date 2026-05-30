# RFDETRSegXLarge ローカル学習パイプライン

このメモは、`RFDETRSegXLarge` をローカルの単一 NVIDIA GPU でインスタンスセグメンテーション用に学習するための実行手順と、RF-DETR 内部でどのコード経路が使われるかを整理したものです。

## モデル概要

`RFDETRSegXLarge` は RF-DETR の高容量インスタンスセグメンテーションモデルです。公開 API では `rfdetr.variants.RFDETRSegXLarge` として定義され、`RFDETRSeg` を継承するため、学習時は通常の `TrainConfig` ではなく `SegmentationTrainConfig` が使われます。

主なデフォルト設定は以下です。

| 項目                         | 値                              |
| ---------------------------- | ------------------------------- |
| モデルクラス                 | `RFDETRSegXLarge`               |
| 設定クラス                   | `RFDETRSegXLargeConfig`         |
| 学習設定                     | `SegmentationTrainConfig`       |
| 入力解像度                   | `624`                           |
| `patch_size`                 | `12`                            |
| `num_windows`                | `2`                             |
| 必須ブロックサイズ           | `patch_size * num_windows = 24` |
| `num_queries` / `num_select` | `300` / `300`                   |
| pretrained weights           | `rf-detr-seg-xlarge.pt`         |

学習パイプラインの大枠は次の流れです。

```text
RFDETRSegXLarge
  -> RFDETRSegXLargeConfig
  -> SegmentationTrainConfig
  -> RFDETR.train()
  -> RFDETRModelModule + RFDETRDataModule + build_trainer()
  -> PyTorch Lightning Trainer.fit()
```

## 環境準備

このリポジトリの Python 要件は `pyproject.toml` の `requires-python` を確認してください。現在の設定では Python 3.12 系が前提です。

ローカルリポジトリから学習用 extra を含めてインストールします。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[train,loggers,kornia]"
```

CUDA が見えていることを確認します。

```bash
python - <<'PY'
import torch

print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY
```

`batch_size="auto"` は CUDA 上で実際に forward/backward のプローブを走らせて安全な micro batch を探すため、CPU では使えません。単一 NVIDIA GPU での本学習を前提にしてください。

## データセット準備

推奨形式は COCO segmentation です。`dataset_dir` には次のようなディレクトリを渡します。

```text
dataset/
  train/
    _annotations.coco.json
    image_001.jpg
    image_002.jpg
  valid/
    _annotations.coco.json
    image_101.jpg
  test/
    _annotations.coco.json
    image_201.jpg
```

`RFDETR.train(dataset_dir=...)` は `train/_annotations.coco.json` がある場合に COCO 形式として自動検出します。Roboflow 形式のデフォルトでは `train`, `valid`, `test` split を想定し、テスト時は `test` split が使われます。

インスタンスセグメンテーションでは、各 annotation に少なくとも次の情報が必要です。

```json
{
  "id": 1,
  "image_id": 1,
  "category_id": 1,
  "bbox": [
    100,
    150,
    200,
    180
  ],
  "area": 36000,
  "iscrowd": 0,
  "segmentation": [
    [
      100,
      150,
      150,
      150,
      200,
      200,
      150,
      250,
      100,
      200
    ]
  ]
}
```

- `bbox` は COCO 標準の `[x, y, width, height]` です。
- `category_id` はカテゴリ定義と一致している必要があります。カスタム COCO / Roboflow データセットでは内部で contiguous label に remap されます。
- `iscrowd=1` や退化した bbox は学習対象から除外されます。
- `segmentation` は polygon list または COCO RLE を使えます。欠落すると segmentation loss / mAP の計算に必要な mask が作れません。

YOLO-Seg も補足的に使えます。`data.yaml` または `data.yml` と `train/images`, `train/labels`, `valid/images`, `valid/labels` がある場合、YOLO 形式として検出されます。segmentation label は次の形です。

```text
<class_id> <x1> <y1> <x2> <y2> <x3> <y3> ...
```

座標は画像サイズで正規化された polygon 座標です。COCO に比べると既存ツールや検証の流れが分かれやすいため、まずは COCO segmentation でパイプラインを安定させるのが無難です。

## 推奨学習スクリプト

`train_seg_xlarge.py` の例です。

```python
from rfdetr import RFDETRSegXLarge


DATASET_DIR = "dataset"
OUTPUT_DIR = "output/seg-xlarge"


model = RFDETRSegXLarge(
    gradient_checkpointing=True,
)

model.train(
    dataset_dir=DATASET_DIR,
    output_dir=OUTPUT_DIR,
    epochs=100,
    batch_size="auto",
    auto_batch_target_effective=16,
    lr=1e-4,
    lr_encoder=1.5e-4,
    early_stopping=True,
    early_stopping_patience=10,
    early_stopping_min_delta=0.001,
    skip_best_epochs=3,
    tensorboard=True,
    progress_bar="tqdm",
    device="cuda",
)
```

実行します。

```bash
python train_seg_xlarge.py
```

`RFDETRSegXLarge()` はデフォルトで `rf-detr-seg-xlarge.pt` をモデルキャッシュにダウンロードして fine-tuning を始めます。重みを使わず scratch で始めたい場合は `RFDETRSegXLarge(pretrain_weights=None, gradient_checkpointing=True)` としますが、通常は精度・収束速度の面で pretrained weights から始める方が適切です。

### 固定 batch の代替設定

auto batch が使えない場合や、手元の GPU で OOM が出る場合は、まず micro batch を 1 に固定します。

```python
model.train(
    dataset_dir=DATASET_DIR,
    output_dir=OUTPUT_DIR,
    epochs=100,
    batch_size=1,
    grad_accum_steps=16,
    lr=1e-4,
    lr_encoder=1.5e-4,
    multi_scale=False,
    tensorboard=True,
    progress_bar="tqdm",
    device="cuda",
)
```

`batch_size * grad_accum_steps` が単一 GPU での effective batch size です。上の例では `1 * 16 = 16` になります。メモリに余裕があれば `batch_size=2, grad_accum_steps=8` や `batch_size=4, grad_accum_steps=4` に上げます。

## 実行監視と成果物

TensorBoard は `output_dir` を見ます。

```bash
tensorboard --logdir output/seg-xlarge
```

主な出力は以下です。

| ファイル                               | 用途                                              |
| -------------------------------------- | ------------------------------------------------- |
| `checkpoint.pth` または `last.ckpt` 系 | 最新状態から学習を再開するための checkpoint       |
| `checkpoint_<N>.pth`                   | `checkpoint_interval` ごとの保存                  |
| `checkpoint_best_ema.pth`              | EMA weight で validation が最良だった checkpoint  |
| `checkpoint_best_regular.pth`          | 通常 weight で validation が最良だった checkpoint |
| `checkpoint_best_total.pth`            | 推論・export に使う最終ベスト checkpoint          |
| `training_config.json`                 | 実行時の train/model config と class names        |

監視で特に見る metric は以下です。

- `train/loss`: 総合 training loss
- `val/loss`: validation loss
- `val/mAP_50_95`, `val/mAP_50`: bbox mAP
- `val/segm_mAP_50_95`, `val/segm_mAP_50`: segmentation mAP
- `val/ema_mAP_50_95`: EMA が有効な場合の EMA bbox mAP
- `val/F1`, `val/precision`, `val/recall`: confidence sweep 由来の分類・検出指標

## 内部パイプライン調査

関連実装は主に以下です。

- `src/rfdetr/variants.py`: `RFDETRSegXLarge` が `RFDETRSeg` を継承し、`_model_config_class = RFDETRSegXLargeConfig` を持つ。
- `src/rfdetr/config.py`: `RFDETRSegXLargeConfig` と `SegmentationTrainConfig` のデフォルト値を定義する。
- `src/rfdetr/detr.py`: 公開 `RFDETR.train()` API を実装する。
- `src/rfdetr/training/*`: Lightning の `RFDETRModelModule`, `RFDETRDataModule`, `build_trainer()` を実装する。

処理の流れは次の通りです。

```text
1. RFDETRSegXLarge(...)
   - RFDETRSegXLargeConfig を構築
   - pretrain_weights があればモデルキャッシュへ解決して download
   - 推論用 ModelContext を初期化

2. model.train(...)
   - device="cuda" を PyTorch Lightning の accelerator/devices に変換
   - resolution が指定されていれば patch_size * num_windows で割り切れるか検証
   - SegmentationTrainConfig を構築
   - batch_size="auto" なら CUDA 上で auto-batch probe を実行
   - dataset_dir から num_classes を検出して model_config に反映

3. RFDETRModelModule
   - build_model_from_config() で segmentation head 付きモデルを構築
   - load_pretrain_weights() で pretrained / fine-tuned checkpoint をロード
   - build_criterion_from_config() で bbox loss と mask loss を構築
   - training_step() で outputs, targets から loss を計算

4. RFDETRDataModule
   - dataset_file="roboflow" の場合は dataset_dir から COCO / YOLO を自動検出
   - segmentation_head=True なので masks を target に含める
   - collate block size は patch_size * num_windows = 24
   - train/valid/test DataLoader を作成

5. build_trainer()
   - precision を CUDA 能力と amp 設定から決定
   - segmentation + DDP の場合は find_unused_parameters=True を設定
   - EMA, COCOEvalCallback, BestModelCallback, early stopping, loggers を接続

6. Trainer.fit()
   - epoch ごとに train / validation を実行
   - COCOEvalCallback が bbox mAP と segm mAP を計算
   - BestModelCallback が best checkpoint と checkpoint_best_total.pth を保存
```

`RFDETRSegXLarge` の `resolution=624` は `24` で割り切れます。独自に `resolution` を変える場合も、必ず `patch_size * num_windows = 24` の倍数にしてください。

## 再開と推論

中断した学習を optimizer / scheduler state も含めて続ける場合は `resume` を使います。

```python
from rfdetr import RFDETRSegXLarge

model = RFDETRSegXLarge(gradient_checkpointing=True)
model.train(
    dataset_dir="dataset",
    output_dir="output/seg-xlarge",
    resume="output/seg-xlarge/checkpoint.pth",
    batch_size="auto",
    auto_batch_target_effective=16,
    device="cuda",
)
```

学習済み weight から新しい run を始める、または推論する場合は `pretrain_weights` に `checkpoint_best_total.pth` を渡します。

```python
from rfdetr import RFDETRSegXLarge

model = RFDETRSegXLarge(
    pretrain_weights="output/seg-xlarge/checkpoint_best_total.pth",
)

detections = model.predict("sample.jpg")
print(detections)
```

使い分けは以下です。

- `resume="output/checkpoint.pth"`: 同じ学習 run を途中から再開する。optimizer state と epoch も復元する。
- `pretrain_weights="output/checkpoint_best_total.pth"`: weight だけを読み、推論または新しい fine-tuning run の初期値にする。

## トラブルシュート

### CUDA OOM

まず `gradient_checkpointing=True` を有効にし、`batch_size="auto"` を使います。それでも落ちる場合は以下へ切り替えます。

```python
model = RFDETRSegXLarge(gradient_checkpointing=True)
model.train(
    dataset_dir="dataset",
    output_dir="output/seg-xlarge",
    batch_size=1,
    grad_accum_steps=16,
    multi_scale=False,
    use_ema=False,
    device="cuda",
)
```

`multi_scale=False` は最大解像度側のメモリ増加を避けます。`use_ema=False` は EMA copy 分のメモリを減らしますが、最終精度が変わる可能性があります。

### `segmentation` 欠落

COCO annotation に `segmentation` がない、または空の場合、mask target が正しく作れません。まず `train/_annotations.coco.json` の annotation に `segmentation` が入っているか確認してください。

```bash
python - <<'PY'
import json
from pathlib import Path

ann = json.loads(Path("dataset/train/_annotations.coco.json").read_text())
missing = [a["id"] for a in ann["annotations"] if not a.get("segmentation")]
print("missing segmentation:", len(missing))
print(missing[:20])
PY
```

### 解像度エラー

`RFDETRSegXLarge` の resolution は `24` の倍数である必要があります。例えば `600`, `624`, `648`, `672` は有効ですが、`640` は `24` で割り切れないため無効です。

### pretrained checkpoint mismatch

segmentation model には segmentation head 付き checkpoint を読み込んでください。検出モデルの checkpoint を `RFDETRSegXLarge` に読む、または segmentation checkpoint を検出モデルに読むと、`segmentation_head` の不一致で失敗します。

また、`patch_size` が異なる checkpoint も基本的に互換性がありません。`RFDETRSegXLarge` は `patch_size=12` です。

### TensorBoard が出ない

`tensorboard=True` はデフォルトですが、パッケージがない場合は警告を出して無効化されます。次を入れてから再実行してください。

```bash
pip install -e ".[loggers]"
```

### DDP / 複数 GPU

この資料の主対象は単一 NVIDIA GPU です。複数 GPU で実行する場合は PyTorch Lightning の DDP 設定、`devices`, effective batch size、segmentation head での `find_unused_parameters=True` の扱いを別途確認してください。

## 最小チェックリスト

- [ ] Python バージョンが `pyproject.toml` の `requires-python` と合っている。
- [ ] `pip install -e ".[train,loggers,kornia]"` が成功している。
- [ ] `torch.cuda.is_available()` が `True`。
- [ ] `dataset/train/_annotations.coco.json` と `dataset/valid/_annotations.coco.json` がある。
- [ ] annotation に `bbox`, `category_id`, `iscrowd`, `segmentation` がある。
- [ ] `RFDETRSegXLarge` の resolution は `24` の倍数。
- [ ] OOM 時の代替設定として `batch_size=1`, `grad_accum_steps=16`, `multi_scale=False` を用意している。
- [ ] 推論には `checkpoint_best_total.pth`、学習再開には `resume` 用 checkpoint を使う。
