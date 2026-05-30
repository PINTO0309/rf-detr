# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for WholeBody49 DEIMv2 model defaults."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from rfdetr.config import RFDETRSegXLargeConfig, TrainConfig
from rfdetr.detr import RFDETR


def test_deimv2_profile_sets_wholebody49_query_defaults() -> None:
    """DEIMv2 WholeBody49 training uses 1240 query slots when the user did not override them."""
    model = object.__new__(RFDETR)
    model.model_config = RFDETRSegXLargeConfig(pretrain_weights=None)
    model.model = SimpleNamespace()
    model.get_model = MagicMock(return_value=SimpleNamespace())
    train_config = TrainConfig(
        dataset_dir="wholebody49",
        dataset_file="deimv2_coco",
        augmentation_profile="deimv2",
    )

    model._apply_deimv2_model_defaults(train_config)

    assert model.model_config.num_queries == 1240
    assert model.model_config.num_select == 1240
    model.get_model.assert_called_once_with(model.model_config)


def test_deimv2_profile_preserves_explicit_query_override() -> None:
    """Explicit query settings remain authoritative for DEIMv2 WholeBody49 training."""
    model = object.__new__(RFDETR)
    model.model_config = RFDETRSegXLargeConfig(
        pretrain_weights=None,
        num_queries=512,
        num_select=512,
    )
    model.model = SimpleNamespace()
    model.get_model = MagicMock(return_value=SimpleNamespace())
    train_config = TrainConfig(
        dataset_dir="wholebody49",
        dataset_file="deimv2_coco",
        augmentation_profile="deimv2",
    )

    model._apply_deimv2_model_defaults(train_config)

    assert model.model_config.num_queries == 512
    assert model.model_config.num_select == 512
    model.get_model.assert_not_called()
