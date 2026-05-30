# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""DEIMv2-compatible COCO/parquet dataset support."""

from __future__ import annotations

import json
from collections.abc import Iterable
from io import BytesIO
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from rfdetr.datasets.coco import convert_coco_poly_to_mask
from rfdetr.datasets.deimv2_transforms import (
    Deimv2Compose,
    build_deimv2_transform_ops,
)
from rfdetr.utilities.logger import get_logger

logger = get_logger()

_WHOLEBODY49_FLIP_PAIRS = (
    (9, 15),
    (10, 14),
    (11, 13),
    (23, 24),
    (27, 28),
    (30, 31),
    (33, 34),
    (37, 38),
    (40, 41),
    (43, 44),
    (46, 47),
)


class _SimpleCocoApi:
    """Small COCO-like metadata object used by RF-DETR callbacks."""

    def __init__(self, categories: list[dict[str, Any]], label2cat: dict[int, int]) -> None:
        self.cats = {int(cat["id"]): cat for cat in categories}
        self.label2cat = label2cat


class CocoParquetStore:
    """Row-per-image COCO parquet reader with optional RAM preload."""

    def __init__(self, ann_file: str | Path, *, preload: bool = True, show_progress: bool = False) -> None:
        try:
            import pyarrow.parquet as pq
        except ImportError as err:
            raise ImportError(
                "DEIMv2 parquet datasets require pyarrow. Install with: pip install 'rfdetr[train]'"
            ) from err

        self.ann_file = Path(ann_file)
        self._pq = pq
        self._parquet = pq.ParquetFile(self.ann_file)
        self._preloaded_rows: list[dict[str, Any]] | None = None
        self.categories = self._load_categories()
        if preload:
            self.preload(show_progress=show_progress)

    def __len__(self) -> int:
        """Return the number of image rows."""
        return int(self._parquet.metadata.num_rows)

    def _load_categories(self) -> list[dict[str, Any]]:
        metadata = self._parquet.metadata.metadata or {}
        raw = metadata.get(b"coco_categories_json")
        if raw is None:
            return []
        categories: list[dict[str, Any]] = json.loads(raw.decode("utf-8"))
        return categories

    def preload(self, *, show_progress: bool = False) -> None:
        """Read all rows into memory to avoid repeated disk I/O."""
        iterator: Iterable[int] = range(len(self))
        if show_progress:
            from tqdm.auto import tqdm

            iterator = tqdm(iterator, desc=f"preload {self.ann_file.name}")
        self._preloaded_rows = [self._read_row(i) for i in iterator]

    def _read_row(self, idx: int) -> dict[str, Any]:
        table = self._parquet.read_row_group(idx)
        row: dict[str, Any] = table.to_pylist()[0]
        row["annotations"] = json.loads(row.pop("annotations_json"))
        return row

    def read_item(self, idx: int) -> tuple[Image.Image, dict[str, Any]]:
        """Return ``(image, row_metadata)`` for one row."""
        row = self._preloaded_rows[idx] if self._preloaded_rows is not None else self._read_row(idx)
        image = Image.open(BytesIO(row["image_bytes"])).convert("RGB")
        return image, row


def _resolve_wholebody49_root(dataset_dir: str | Path) -> Path:
    root = Path(dataset_dir)
    if (root / "annotations").exists():
        return root
    candidate = root / "tools" / "dataset" / "wholebody49"
    if (candidate / "annotations").exists():
        return candidate
    raise FileNotFoundError(
        f"Could not find DEIMv2 wholebody49 annotations under {root}. "
        "Expected annotations/*.parquet or tools/dataset/wholebody49/annotations/*.parquet."
    )


def _classes_from_file(root: Path) -> list[dict[str, Any]]:
    classes_file = root / "classes.txt"
    if not classes_file.exists():
        return []
    names = [line.strip() for line in classes_file.read_text().splitlines() if line.strip()]
    return [{"id": idx, "name": name} for idx, name in enumerate(names)]


def _decode_masks_and_valid(segmentations: list[Any], height: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    masks: list[torch.Tensor] = []
    valid: list[bool] = []
    for segmentation in segmentations:
        has_segmentation = bool(segmentation)
        if has_segmentation:
            try:
                mask = convert_coco_poly_to_mask([segmentation], height, width)[0].to(torch.bool)
                has_segmentation = bool(mask.any())
            except Exception:
                logger.warning("Skipping malformed segmentation during DEIMv2 dataset conversion.", exc_info=True)
                mask = torch.zeros((height, width), dtype=torch.bool)
                has_segmentation = False
        else:
            mask = torch.zeros((height, width), dtype=torch.bool)
        masks.append(mask)
        valid.append(has_segmentation)
    if not masks:
        return torch.zeros((0, height, width), dtype=torch.bool), torch.zeros((0,), dtype=torch.bool)
    return torch.stack(masks, dim=0), torch.tensor(valid, dtype=torch.bool)


class Deimv2CocoDetection(torch.utils.data.Dataset[tuple[Any, dict[str, Any]]]):  # type: ignore[misc]
    """DEIMv2 WholeBody49 COCO/parquet dataset."""

    def __init__(
        self,
        root: str | Path,
        ann_file: str | Path,
        transforms: Any | None,
        *,
        include_masks: bool,
        preload_parquet: bool = True,
        mask_target_class_ids: list[int] | None = None,
        segm_eval_category_ids: list[int] | None = None,
        segm_ignore_missing_masks: bool = True,
    ) -> None:
        self.root = Path(root)
        self.store = CocoParquetStore(ann_file, preload=preload_parquet)
        categories = self.store.categories or _classes_from_file(self.root)
        sorted_categories = sorted(categories, key=lambda item: int(item["id"]))
        self.cat2label = {int(cat["id"]): i for i, cat in enumerate(sorted_categories)}
        self.label2cat = {label: cat_id for cat_id, label in self.cat2label.items()}
        self.coco = _SimpleCocoApi(categories, self.label2cat)
        self._transforms = transforms
        self.include_masks = include_masks
        self.mask_target_class_ids = set(mask_target_class_ids or [0])
        eval_category_ids = segm_eval_category_ids if segm_eval_category_ids is not None else self.mask_target_class_ids
        self.segm_eval_category_ids = set(eval_category_ids)
        self.segm_ignore_missing_masks = segm_ignore_missing_masks
        self._epoch = -1

    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.store)

    @property
    def epoch(self) -> int:
        """Current training epoch used by DEIMv2 transform policies."""
        return self._epoch

    def set_epoch(self, epoch: int) -> None:
        """Set current training epoch."""
        self._epoch = int(epoch)
        if self._transforms is not None and hasattr(self._transforms, "set_epoch"):
            self._transforms.set_epoch(epoch)

    def load_item(self, idx: int) -> tuple[Image.Image, dict[str, Any]]:
        """Load and convert a sample before online transforms."""
        image, row = self.store.read_item(idx)
        target = self._prepare(image, row)
        return image, target

    def __getitem__(self, idx: int) -> tuple[Any, dict[str, Any]]:
        """Return one transformed sample."""
        image, target = self.load_item(idx)
        if self._transforms is not None:
            result = self._transforms(image, target, self)
            image, target = result[0], result[1]
        return image, target

    def _prepare(self, image: Image.Image, row: dict[str, Any]) -> dict[str, Any]:
        width, height = image.size
        annotations = [obj for obj in row["annotations"] if obj.get("iscrowd", 0) == 0]
        boxes = torch.as_tensor(
            [obj.get("bbox", [0, 0, 0, 0]) for obj in annotations],
            dtype=torch.float32,
        ).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=width)
        boxes[:, 1::2].clamp_(min=0, max=height)

        labels = []
        original_category_ids = []
        for obj in annotations:
            category_id = int(obj["category_id"])
            original_category_ids.append(category_id)
            labels.append(self.cat2label.get(category_id, category_id))
        labels_tensor = torch.tensor(labels, dtype=torch.int64)
        category_tensor = torch.tensor(original_category_ids, dtype=torch.int64)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        labels_tensor = labels_tensor[keep]
        category_tensor = category_tensor[keep]

        target: dict[str, Any] = {
            "boxes": boxes,
            "labels": labels_tensor,
            "image_id": torch.tensor([int(row["image_id"])]),
            "area": torch.tensor([obj.get("area", 0.0) for obj in annotations], dtype=torch.float32)[keep],
            "iscrowd": torch.tensor([obj.get("iscrowd", 0) for obj in annotations], dtype=torch.int64)[keep],
            "orig_size": torch.as_tensor([height, width], dtype=torch.int64),
            "size": torch.as_tensor([height, width], dtype=torch.int64),
        }

        if self.include_masks:
            segmentations = [obj.get("segmentation", []) for obj in annotations]
            masks, has_mask = _decode_masks_and_valid(segmentations, height, width)
            masks = masks[keep]
            has_mask = has_mask[keep]
            mask_class = torch.tensor(
                [int(label.item()) in self.mask_target_class_ids for label in labels_tensor],
                dtype=torch.bool,
            )
            segm_eval_class = torch.tensor(
                [int(label.item()) in self.segm_eval_category_ids for label in labels_tensor],
                dtype=torch.bool,
            )
            target["masks"] = masks
            target["mask_valid"] = has_mask & mask_class
            target["segm_eval_valid"] = (has_mask | (not self.segm_ignore_missing_masks)) & segm_eval_class
        return target


def _default_deimv2_entries(args: Any, resolution: int, image_set: str) -> list[dict[str, Any]]:
    if image_set != "train":
        return [
            {"type": "Resize", "size": [resolution, resolution]},
            {"type": "ConvertPILImage", "dtype": "float32", "scale": True},
            {"type": "Normalize", "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
            {"type": "ConvertBoxes", "fmt": "cxcywh", "normalize": True},
        ]

    entries: list[dict[str, Any]] = []
    if getattr(args, "mosaic_prob", -0.1) > 0:
        entries.append(
            {
                "type": "Mosaic",
                "output_size": resolution,
                "probability": 1.0,
                "use_cache": True,
            }
        )
    entries.extend(
        [
            {"type": "RandomPhotometricDistort", "p": 0.5},
            {"type": "RandomZoomOut", "p": 0.5, "fill": 0, "side_range": [1.0, 1.5]},
            {"type": "RandomIoUCrop", "p": 0.8},
            {"type": "SanitizeBoundingBoxes", "min_size": 1},
            {
                "type": "RandomHorizontalFlipWithClass",
                "p": 0.5,
                "class_pairs": getattr(args, "class_flip_pairs", None) or list(_WHOLEBODY49_FLIP_PAIRS),
            },
            {"type": "Resize", "size": [resolution, resolution]},
            {"type": "SanitizeBoundingBoxes", "min_size": 1},
            {"type": "ConvertPILImage", "dtype": "float32", "scale": True},
            {"type": "Normalize", "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
            {"type": "ConvertBoxes", "fmt": "cxcywh", "normalize": True},
        ]
    )
    return entries


def make_deimv2_transforms(image_set: str, args: Any, resolution: int) -> Deimv2Compose:
    """Build DEIMv2-compatible transforms."""
    raw_entries = getattr(args, "deim_aug_config", None) or _default_deimv2_entries(args, resolution, image_set)
    ops = build_deimv2_transform_ops(raw_entries, mask_resize_origin=getattr(args, "mask_resize_origin", "center"))
    policy = None
    if image_set == "train":
        policy = {
            "name": "stop_epoch",
            "epoch": getattr(args, "deim_transform_policy_epochs", [4, 29, 90]),
            "ops": getattr(
                args,
                "deim_transform_policy_ops",
                ["RandomPhotometricDistort", "RandomZoomOut", "RandomIoUCrop"],
            ),
        }
    return Deimv2Compose(ops, policy=policy, mosaic_prob=getattr(args, "mosaic_prob", -0.1))


def build_deimv2_coco(image_set: str, args: Any, resolution: int) -> Deimv2CocoDetection:
    """Build the DEIMv2 WholeBody49 parquet dataset."""
    root = _resolve_wholebody49_root(args.dataset_dir)
    annotations = root / "annotations"
    split = image_set.split("_")[0]
    paths = {
        "train": annotations / "train_ins.parquet",
        "val": annotations / "val_ins.parquet",
        "test": annotations / "test_ins.parquet",
    }
    ann_file = paths.get(split, paths["val"])
    if split == "val" and not ann_file.exists():
        ann_file = annotations / "val.parquet"
    if split == "test" and not ann_file.exists():
        ann_file = annotations / "val_ins.parquet"
    if not ann_file.exists():
        raise FileNotFoundError(f"DEIMv2 annotation file not found: {ann_file}")

    transforms = make_deimv2_transforms(image_set, args, resolution)
    return Deimv2CocoDetection(
        root,
        ann_file,
        transforms,
        include_masks=getattr(args, "segmentation_head", False),
        preload_parquet=getattr(args, "preload_parquet", True),
        mask_target_class_ids=getattr(args, "mask_target_class_ids", [0]),
        segm_eval_category_ids=getattr(args, "segm_eval_category_ids", [0]),
        segm_ignore_missing_masks=getattr(args, "segm_ignore_missing_masks", True),
    )
