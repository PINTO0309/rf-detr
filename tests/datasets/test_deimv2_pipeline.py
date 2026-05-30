# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for DEIMv2-compatible dataset and augmentation helpers."""

from __future__ import annotations

import json
from io import BytesIO

import pytest
import torch
from PIL import Image

from rfdetr.datasets.deimv2 import CocoParquetStore, Deimv2CocoDetection
from rfdetr.datasets.deimv2_transforms import Deimv2CollateFunction, RandomHorizontalFlipWithClass


def _png_bytes() -> bytes:
    image = Image.new("RGB", (8, 8), color=(255, 0, 0))
    buf = BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture()
def deimv2_parquet(tmp_path):
    """Create a tiny DEIMv2 row-per-image parquet file."""
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    ann_file = tmp_path / "annotations" / "train_ins.parquet"
    ann_file.parent.mkdir()
    categories = [{"id": 0, "name": "body"}, {"id": 1, "name": "joint"}]
    annotations = [
        {
            "id": 1,
            "image_id": 7,
            "bbox": [1, 1, 4, 4],
            "category_id": 0,
            "area": 16,
            "iscrowd": 0,
            "segmentation": [[1, 1, 5, 1, 5, 5, 1, 5]],
        },
        {
            "id": 2,
            "image_id": 7,
            "bbox": [2, 2, 2, 2],
            "category_id": 1,
            "area": 4,
            "iscrowd": 0,
            "segmentation": [],
        },
    ]
    table = pa.table(
        {
            "image_id": [7],
            "file_name": ["sample.png"],
            "width": [8],
            "height": [8],
            "image_bytes": [_png_bytes()],
            "annotations_json": [json.dumps(annotations)],
        }
    )
    metadata = dict(table.schema.metadata or {})
    metadata[b"coco_categories_json"] = json.dumps(categories).encode()
    metadata[b"coco_parquet_format"] = b"deimv2.coco-image-row.v1"
    pq.write_table(table.replace_schema_metadata(metadata), ann_file)
    return tmp_path, ann_file


def test_coco_parquet_store_preloads_image_bytes_and_annotations(deimv2_parquet) -> None:
    """Parquet rows are decoded from RAM into PIL images and annotation dicts."""
    _, ann_file = deimv2_parquet
    store = CocoParquetStore(ann_file, preload=True)

    image, row = store.read_item(0)

    assert image.size == (8, 8)
    assert row["image_id"] == 7
    assert row["annotations"][0]["category_id"] == 0
    assert store.categories[0]["name"] == "body"


def test_deimv2_dataset_marks_missing_masks_invalid(deimv2_parquet) -> None:
    """Missing keypoint-like segmentations stay as bbox targets but are excluded from mask supervision."""
    root, ann_file = deimv2_parquet
    dataset = Deimv2CocoDetection(
        root,
        ann_file,
        transforms=None,
        include_masks=True,
        preload_parquet=True,
        mask_target_class_ids=[0],
        segm_eval_category_ids=[0],
    )

    _, target = dataset[0]

    assert target["boxes"].shape == (2, 4)
    assert target["masks"].shape == (2, 8, 8)
    assert target["mask_valid"].tolist() == [True, False]
    assert target["segm_eval_valid"].tolist() == [True, False]


def test_random_horizontal_flip_with_class_swaps_pairs() -> None:
    """Class-aware flip mirrors DEIMv2 left/right class pair handling."""
    transform = RandomHorizontalFlipWithClass(p=1.0, class_pairs=[[1, 2]])
    image = Image.new("RGB", (10, 10))
    target = {
        "boxes": torch.tensor([[1.0, 1.0, 4.0, 4.0], [5.0, 1.0, 8.0, 4.0]]),
        "labels": torch.tensor([1, 2]),
    }

    _, out, _ = transform((image, target, object()))

    assert out["labels"].tolist() == [2, 1]


def test_deimv2_collate_mixup_keeps_masks_aligned() -> None:
    """Mask-aware MixUp concatenates object-level fields without collate multi-scale resizing."""
    collate = Deimv2CollateFunction(block_size=4, mixup_prob=1.0, mixup_epochs=[0, 2])
    collate.set_epoch(1)
    image = torch.zeros((3, 8, 8))
    target = {
        "boxes": torch.tensor([[0.5, 0.5, 0.25, 0.25]]),
        "labels": torch.tensor([0]),
        "area": torch.tensor([4.0]),
        "iscrowd": torch.tensor([0]),
        "masks": torch.ones((1, 8, 8), dtype=torch.bool),
        "mask_valid": torch.tensor([True]),
        "segm_eval_valid": torch.tensor([True]),
        "orig_size": torch.tensor([8, 8]),
        "size": torch.tensor([8, 8]),
    }

    _, targets = collate([(image, target), (image + 1, target)])

    assert targets[0]["boxes"].shape[0] == 2
    assert targets[0]["masks"].shape[0] == 2
    assert targets[0]["mask_valid"].tolist() == [True, True]


def test_deimv2_collate_state_restores_epoch() -> None:
    """Collate augmentation epoch survives checkpoint resume."""
    collate = Deimv2CollateFunction(block_size=4, mixup_prob=1.0, mixup_epochs=[0, 2])
    collate.set_epoch(7)
    restored = Deimv2CollateFunction(block_size=4, mixup_prob=1.0, mixup_epochs=[0, 2])

    restored.load_state_dict(collate.state_dict())

    assert restored.epoch == 7
