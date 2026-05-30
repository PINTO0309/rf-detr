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

`dataset_file="deimv2_coco"` かつ `augmentation_profile="deimv2"` の WholeBody49 学習では、ユーザーが `num_queries` / `num_select` を明示していない場合に `1240` / `1240` へ自動調整します。通常の COCO/Roboflow/YOLO 学習では上表の `300` / `300` のままです。

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
pip install --force-reinstall "numpy==1.26.4" "pyarrow==14.0.1"
pip install -e ".[train,loggers,kornia]"
```

`pyarrow==14.0.1` は NumPy 2.x と ABI 互換でない wheel を読むことがあるため、学習環境では `numpy==1.26.4` に固定します。`AttributeError: _ARRAY_API not found` や `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x` が出た場合も、上記の強制再インストールで NumPy 1.x へ戻してください。

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

## デフォルト augmentation の処理

`model.train()` で `aug_config` を指定しない場合、学習 split にはデフォルト augmentation が有効です。validation / test split にはランダム augmentation は入りません。

デフォルト設定の要点は以下です。

| 設定                           | デフォルト値                     | 意味                                                       |
| ------------------------------ | -------------------------------- | ---------------------------------------------------------- |
| `aug_config`                   | `None`                           | `src/rfdetr/datasets/aug_config.py` の `AUG_CONFIG` を使う |
| `AUG_CONFIG`                   | `{"HorizontalFlip": {"p": 0.5}}` | 50% の左右反転だけが有効                                   |
| `augmentation_backend`         | `"cpu"`                          | Albumentations 系 transform を dataset 側で実行する        |
| `multi_scale`                  | `True`                           | 学習 batch 開始時にもランダム resize を行う                |
| `expanded_scales`              | `True`                           | 解像度候補を広めに取る                                     |
| `multi_scale_min_offset`       | `None`                           | multi-scale 候補の下限 offset を任意に制限する             |
| `multi_scale_max_offset`       | `None`                           | multi-scale 候補の上限 offset を任意に制限する             |
| `square_resize_div_64`         | `True`                           | train/valid/test を正方形 resize 系 pipeline にする        |
| `do_random_resize_via_padding` | `False`                          | dataset 側は最大 scale 固定、batch 側で multi-scale する   |

### 学習 dataset 側の transform

`RFDETRSegXLarge` のデフォルトでは `resolution=624`, `patch_size=12`, `num_windows=2` なので、必須ブロックサイズは `24` です。`multi_scale=True` かつ `expanded_scales=True` の場合、学習用の scale 候補は次の 11 個になります。

```text
504, 528, 552, 576, 600, 624, 648, 672, 696, 720, 744
```

`resolution=624` のまま拡大方向だけを `648` までに抑える場合は `multi_scale_max_offset=1` を指定します。`expanded_scales=False` と組み合わせると候補は次になります。

```text
552, 576, 600, 624, 648
```

COCO / Roboflow segmentation の train split では、dataset の `__getitem__` 時におおむね次の順で処理されます。

```text
画像 + COCO annotation 読み込み
  -> segmentation polygon/RLE から mask target を作成
  -> ランダム resize/crop
     - 直接 Resize(height=s, width=s)
     - または SmallestMaxSize(400/500/600)
       -> RandomSizedCrop(min_max_height=[384, 600])
       -> Resize(height=s, width=s)
  -> AlbumentationsWrapper(AUG_CONFIG)
     - HorizontalFlip(p=0.5)
  -> ToImage()
  -> ToDtype(torch.float32, scale=True)
  -> Normalize(ImageNet mean/std)
     - bbox は絶対 xyxy から正規化 cxcywh に変換
```

デフォルトの `do_random_resize_via_padding=False` では、dataset 側の `s` は scale 候補の最大値 `744` に固定されます。その後の Lightning batch 側で、実際に使う scale が batch ごとに `504` から `744` の範囲で選ばれます。`HorizontalFlip` や resize/crop は幾何変換なので、画像、bbox、segmentation mask が同じ変換で更新されます。mask だけが置き去りになる処理ではありません。

validation / test split では固定の `Resize(height=624, width=624)`、`ToImage()`、`ToDtype()`、`Normalize()` が使われ、`HorizontalFlip` や random crop は入りません。

### `square_resize_div_64` の分岐

`square_resize_div_64=True` は `make_coco_transforms_square_div_64()` を使う設定です。名前に `div_64` とありますが、`RFDETRSegXLarge` で実際に重要なのは `patch_size * num_windows = 24` の倍数です。`resolution=624` と multi-scale 候補はこの `24` の倍数になるように作られ、DataLoader の `collate_fn` も batch の `H` / `W` を `24` の倍数へ padding します。

挙動は以下です。

```text
square_resize_div_64=True
  train:
    - 正方形 Resize(height=s, width=s)
    - または SmallestMaxSize(400/500/600)
      -> RandomSizedCrop(...)
      -> 正方形 Resize(height=s, width=s)
  val/test/val_speed:
    - 固定 Resize(height=resolution, width=resolution)
```

この設定ではアスペクト比は保持されません。最終的に正方形へ resize されます。RF-DETR の既定学習経路ではこの設定が有効で、`RFDETRSegXLarge` のローカル単一 GPU 学習でも基本はこれを前提にします。

`square_resize_div_64=False` にすると `make_coco_transforms()` を使います。train split は短辺側を `s` に合わせ、長辺を最大 `1333` に抑える非正方形 pipeline になります。

```text
square_resize_div_64=False
  train:
    - SmallestMaxSize(max_size=s)
      -> LongestMaxSize(max_size=1333)
    - または SmallestMaxSize(400/500/600)
      -> RandomCrop(height=384, width=384)
      -> SmallestMaxSize(max_size=s)
      -> LongestMaxSize(max_size=1333)
  val/test:
    - SmallestMaxSize(max_size=resolution)
      -> LongestMaxSize(max_size=1333)
  val_speed:
    - 固定 Resize(height=resolution, width=resolution)
```

非正方形 pipeline でも、最後に DataLoader の `make_collate_fn(block_size=patch_size * num_windows)` が batch 内の最大 `H` / `W` を `24` の倍数に切り上げて padding します。つまり、`square_resize_div_64=False` は「padding しない」という意味ではなく、「dataset transform で正方形に潰さず、collate で必要な分だけ padding する」という意味です。

### `do_random_resize_via_padding` の分岐

`do_random_resize_via_padding` は、multi-scale のランダム性を dataset 側で入れるか、Lightning batch 側で入れるかを切り替えます。

デフォルトの `False` では、dataset builder が transform に `skip_random_resize=True` を渡します。

```text
do_random_resize_via_padding=False
  dataset transform:
    - multi-scale 候補を計算する
    - ただし resize target は最大 scale のみを使う
    - RFDETRSegXLarge では s=744
  collate:
    - batch を block_size=24 の倍数へ padding
  on_train_batch_start:
    - global_step を seed にして scale を選ぶ
    - samples.tensors と samples.mask を選択 scale へ interpolate
```

この場合、auto batch probe も最大 scale を使うため、`RFDETRSegXLarge` では概ね `744x744` 相当を基準に micro batch を探します。OOM を避けるには保守的ですが、batch 開始時の resize により training step の入力解像度は batch ごとに変わります。

`True` にすると、dataset builder が transform に `skip_random_resize=False` を渡し、`RFDETRModelModule.on_train_batch_start()` の追加 resize は止まります。

```text
do_random_resize_via_padding=True
  dataset transform:
    - 各 sample の resize/crop target s を multi-scale 候補から選ぶ
    - RFDETRSegXLarge では 504..744 のいずれか
  collate:
    - batch 内の最大 H/W に合わせる
    - さらに block_size=24 の倍数へ切り上げて padding
    - padding 領域は NestedTensor.mask で model に渡す
  on_train_batch_start:
    - multi-scale resize は実行しない
```

この場合は sample ごとのサイズ差を padding mask で扱います。batch 内に大きい sample が混じると、その batch 全体の padded tensor は大きくなるため、メモリ使用量は batch 構成に依存します。一方で、batch 開始時の追加 interpolate を避けられます。

整理すると次の通りです。

| 設定組み合わせ                                                     | train resize の形                    | multi-scale が入る場所   | batch の揃え方                          |
| ------------------------------------------------------------------ | ------------------------------------ | ------------------------ | --------------------------------------- |
| `square_resize_div_64=True`, `do_random_resize_via_padding=False`  | 正方形、dataset 側は最大 scale       | `on_train_batch_start()` | resize 後に `24` の倍数へ padding       |
| `square_resize_div_64=True`, `do_random_resize_via_padding=True`   | 正方形、sample ごとに random scale   | dataset transform        | batch 最大 scale に合わせて padding     |
| `square_resize_div_64=False`, `do_random_resize_via_padding=False` | 非正方形、dataset 側は最大 scale     | `on_train_batch_start()` | batch 最大 H/W を `24` の倍数へ padding |
| `square_resize_div_64=False`, `do_random_resize_via_padding=True`  | 非正方形、sample ごとに random scale | dataset transform        | batch 最大 H/W を `24` の倍数へ padding |

通常はデフォルトの `square_resize_div_64=True`, `do_random_resize_via_padding=False` で始めます。元画像のアスペクト比維持を重視する場合は `square_resize_div_64=False` を検討します。batch 側 interpolate を避けたい、または sample ごとの multi-scale + padding の挙動を試したい場合だけ `do_random_resize_via_padding=True` を検討します。

### 画像スケーリングで Pillow が使われるか

ローカル学習パイプラインでは、画像の読み込みには Pillow が使われますが、resize / crop 後 resize などのスケーリング本体は Pillow ではありません。

処理ごとの実体は以下です。

| 処理                                                           | 実装箇所                                   | スケーリング実体                                       |
| -------------------------------------------------------------- | ------------------------------------------ | ------------------------------------------------------ |
| dataset 画像読み込み                                           | `torchvision.datasets.CocoDetection`       | `PIL.Image.open(...).convert("RGB")`                   |
| RF-DETR の COCO annotation 変換                                | `src/rfdetr/datasets/coco.py`              | Pillow 画像サイズを参照するだけ                        |
| train/val/test の dataset resize                               | `AlbumentationsWrapper` + Albumentations   | NumPy 配列に変換後、Albumentations 経由の `cv2.resize` |
| `RandomSizedCrop` の crop 後 resize                            | Albumentations                             | crop は NumPy、resize は `cv2.resize`                  |
| segmentation mask の resize                                    | Albumentations                             | `cv2.resize`。デフォルトは mask 用 nearest             |
| `do_random_resize_via_padding=False` 時の batch 側 multi-scale | `RFDETRModelModule.on_train_batch_start()` | `torch.nn.functional.interpolate`                      |
| `augmentation_backend="gpu"` 時の augmentation                 | Kornia                                     | GPU tensor 上の Kornia transform                       |

`src/rfdetr/datasets/transforms.py` の `AlbumentationsWrapper` は、入力の `PIL.Image` を `np.array(image)` に変換してから Albumentations に渡し、出力を `Image.fromarray(...)` で PIL に戻します。この `Image.fromarray(...)` は形式変換であり、resize の実体ではありません。

`pyproject.toml` では training extra に `albumentations==2.0.8` が固定されています。このバージョンの `Resize`, `SmallestMaxSize`, `LongestMaxSize`, `RandomSizedCrop` は内部で Albumentations / albucore の `resize()` を呼び、最終的に `cv2.resize` を使います。デフォルト interpolation は画像が `cv2.INTER_LINEAR`、mask が `cv2.INTER_NEAREST` です。

推論 API の `RFDETR.predict()` も、文字列 path の読み込みには `PIL.Image.open()` を使います。ただし、resize 前に `torchvision.transforms.functional.to_tensor()` で tensor 化し、その後 `torchvision.transforms.functional.resize()` を tensor に対して呼ぶため、通常の `predict()` の入力 resize も Pillow resize ではありません。

例外として、export / deploy 用の補助コードやドキュメントには `PIL.Image.resize()` を使うサンプルがあります。これはローカル学習の `RFDETRDataModule` / `CocoDetection` pipeline とは別経路です。

### Lightning batch 側の multi-scale

デフォルトでは dataset 側の transform に加えて、`RFDETRModelModule.on_train_batch_start()` でも学習 batch ごとに multi-scale resize が走ります。

```text
train DataLoader batch
  -> collate_fn で patch_size * num_windows = 24 の倍数に揃える
  -> on_train_batch_start()
     - global_step で seed を固定
     - scale 候補から 1 つ選ぶ
     - samples.tensors と samples.mask を interpolate
  -> training_step()
```

この段階では target bbox は正規化済みなので、画像 tensor と padding mask を同じ scale に resize します。`multi_scale=False` にすると、この batch 側の resize は止まります。

### augmentation を変える / 止める

デフォルトの左右反転だけを止めたい場合は `aug_config={}` を渡します。ただし、これは `AUG_CONFIG` を空にするだけです。dataset 側の resize/crop、batch 側の `multi_scale`、正規化までは止まりません。

```python
model.train(
    dataset_dir=DATASET_DIR,
    output_dir=OUTPUT_DIR,
    aug_config={},
    device="cuda",
)
```

左右反転と batch 側 multi-scale の両方を止め、より固定的な設定に寄せる例です。

```python
model.train(
    dataset_dir=DATASET_DIR,
    output_dir=OUTPUT_DIR,
    aug_config={},
    multi_scale=False,
    device="cuda",
)
```

用意済み preset を使う場合は `src/rfdetr/datasets/aug_config.py` から import します。

```python
from rfdetr.datasets.aug_config import AUG_CONSERVATIVE

model.train(
    dataset_dir=DATASET_DIR,
    output_dir=OUTPUT_DIR,
    aug_config=AUG_CONSERVATIVE,
    device="cuda",
)
```

代表的な preset は `AUG_CONSERVATIVE`, `AUG_AGGRESSIVE`, `AUG_AERIAL`, `AUG_INDUSTRIAL` です。インスタンスセグメンテーションでは mask と bbox が同時に変換される必要があるため、まずはデフォルトか `AUG_CONSERVATIVE` で validation mAP を確認し、過度な回転・shear・blur を一度に入れない方が調査しやすくなります。

`augmentation_backend="auto"` または `"gpu"` を指定すると、条件が合う場合は Kornia による GPU augmentation / normalize に切り替わります。今回の単一 GPU ローカル学習手順では、挙動確認が容易なデフォルトの `augmentation_backend="cpu"` を基準にします。

## 実行監視と成果物

TensorBoard は `output_dir` を見ます。

```bash
tensorboard --logdir output/seg-xlarge
```

主な出力は以下です。

| ファイル                      | 用途                                                    |
| ----------------------------- | ------------------------------------------------------- |
| `last.ckpt`                   | optimizer / scheduler / epoch を含む最新再開 checkpoint |
| `checkpoint_<N>.pth`          | `checkpoint_interval` ごとの保存                        |
| `checkpoint_best_ema.pth`     | EMA weight で validation が最良だった checkpoint        |
| `checkpoint_best_regular.pth` | 通常 weight で validation が最良だった checkpoint       |
| `checkpoint_best_total.pth`   | 推論・export に使う最終ベスト checkpoint                |
| `training_config.json`        | 実行時の train/model config と class names              |

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
    resume="output/seg-xlarge/last.ckpt",
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

- `resume="output/last.ckpt"`: 同じ学習 run を途中から再開する。optimizer state、scheduler state、epoch、Lightning callback/DataModule state も復元する。
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
    multi_scale=True,
    expanded_scales=False,
    multi_scale_max_offset=1,
    use_ema=False,
    device="cuda",
)
```

`multi_scale_max_offset=1` は `resolution=624` のまま multi-scale の最大候補を `648x648` に抑えます。さらに厳しい場合は `multi_scale=False` で batch 側 multi-scale resize 自体を止めます。`use_ema=False` は EMA copy 分のメモリを減らしますが、最終精度が変わる可能性があります。

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
