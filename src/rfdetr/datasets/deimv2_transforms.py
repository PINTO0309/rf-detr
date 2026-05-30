# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""DEIMv2-compatible CPU transforms and collate-time augmentations."""

from __future__ import annotations

import copy
import random
from collections import defaultdict
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
import torchvision.transforms.v2 as T  # noqa: N812
import torchvision.transforms.v2.functional as TVF  # noqa: N812
from PIL import Image
from torch.utils._pytree import tree_flatten, tree_unflatten
from torchvision import tv_tensors

from rfdetr.datasets.mask_resize import compute_resized_mask_output_size, resize_masks
from rfdetr.utilities.box_ops import box_xyxy_to_cxcywh
from rfdetr.utilities.tensors import _collate_with_block_size

torchvision.disable_beta_transforms_warning()

_OBJECT_LEVEL_TARGET_KEYS = (
    "labels",
    "area",
    "iscrowd",
    "masks",
    "mask_valid",
    "segm_eval_valid",
    "mixup",
    "keypoints",
)


def _split_sample(inputs: Any) -> tuple[Any, dict[str, Any], Any | None]:
    """Normalize transform input into ``(image, target, dataset)``."""
    if isinstance(inputs, tuple) and len(inputs) == 1:
        inputs = inputs[0]
    if not isinstance(inputs, (tuple, list)) or len(inputs) < 2:
        raise TypeError("DEIMv2 transforms expect (image, target[, dataset])")
    dataset = inputs[2] if len(inputs) > 2 else None
    return inputs[0], inputs[1], dataset


def _pack_sample(
    image: Any,
    target: dict[str, Any],
    dataset: Any | None,
) -> tuple[Any, dict[str, Any], Any] | tuple[Any, dict[str, Any]]:
    """Restore a transform output tuple."""
    if dataset is None:
        return image, target
    return image, target, dataset


def _image_size(image: Any) -> tuple[int, int]:
    """Return image size as ``(height, width)``."""
    if isinstance(image, Image.Image):
        width, height = image.size
        return height, width
    return int(image.shape[-2]), int(image.shape[-1])


def _as_tv_target(target: dict[str, Any], spatial_size: tuple[int, int]) -> dict[str, Any]:
    """Convert boxes/masks to torchvision tv_tensors for v2 geometry transforms."""
    out = dict(target)
    if "boxes" in out and not isinstance(out["boxes"], tv_tensors.BoundingBoxes):
        out["boxes"] = tv_tensors.BoundingBoxes(out["boxes"], format="XYXY", canvas_size=spatial_size)
    if "masks" in out and not isinstance(out["masks"], tv_tensors.Mask):
        out["masks"] = tv_tensors.Mask(out["masks"].to(torch.bool))
    return out


def _plain_target(target: dict[str, Any]) -> dict[str, Any]:
    """Convert tv_tensors in a target dict back to plain tensors."""
    out = dict(target)
    if isinstance(out.get("boxes"), tv_tensors.BoundingBoxes):
        out["boxes"] = torch.as_tensor(out["boxes"], dtype=torch.float32)
    if isinstance(out.get("masks"), tv_tensors.Mask):
        out["masks"] = torch.as_tensor(out["masks"], dtype=torch.bool)
    return out


def _apply_keep(target: dict[str, Any], keep: torch.Tensor) -> dict[str, Any]:
    """Apply an object-level keep mask to target fields."""
    out = dict(target)
    num_boxes = int(target.get("boxes", torch.empty(0, 4)).shape[0])
    for key, value in list(target.items()):
        if key == "boxes" or key in _OBJECT_LEVEL_TARGET_KEYS:
            if torch.is_tensor(value) and value.ndim > 0 and int(value.shape[0]) == num_boxes:
                out[key] = value[keep]
    return out


class _TorchvisionWrapper:
    """Wrap a torchvision v2 transform for RF-DETR's ``(image, target, dataset)`` tuple."""

    def __init__(self, transform: Callable[..., Any]) -> None:
        self.transform: Any = transform

    def __call__(self, inputs: Any) -> Any:
        """Apply the wrapped transform."""
        image, target, dataset = _split_sample(inputs)
        tv_target = _as_tv_target(target, _image_size(image))
        image, tv_target = self.transform(image, tv_target)
        return _pack_sample(image, _plain_target(tv_target), dataset)


class RandomPhotometricDistort(_TorchvisionWrapper):
    """DEIMv2-compatible photometric distortion."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(T.RandomPhotometricDistort(**kwargs))


class RandomZoomOut(_TorchvisionWrapper):
    """DEIMv2-compatible random zoom-out."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(T.RandomZoomOut(**kwargs))


class RandomHorizontalFlip(_TorchvisionWrapper):
    """DEIMv2-compatible random horizontal flip."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(T.RandomHorizontalFlip(**kwargs))


class RandomCrop(_TorchvisionWrapper):
    """DEIMv2-compatible random crop."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(T.RandomCrop(**kwargs))


class RandomIoUCrop(_TorchvisionWrapper):
    """Random IoU crop with DEIMv2's explicit probability wrapper."""

    def __init__(
        self,
        min_scale: float = 0.3,
        max_scale: float = 1.0,
        min_aspect_ratio: float = 0.5,
        max_aspect_ratio: float = 2.0,
        sampler_options: list[float] | None = None,
        trials: int = 40,
        p: float = 1.0,
    ) -> None:
        self.p = p
        super().__init__(
            T.RandomIoUCrop(min_scale, max_scale, min_aspect_ratio, max_aspect_ratio, sampler_options, trials)
        )

    def __call__(self, inputs: Any) -> Any:
        """Apply the crop with probability ``p``."""
        if torch.rand(1) >= self.p:
            return inputs
        return super().__call__(inputs)


class RandomHorizontalFlipWithClass(RandomHorizontalFlip):
    """Horizontally flip and swap configured left/right class labels."""

    def __init__(self, p: float = 0.5, class_pairs: Sequence[Sequence[int]] | None = None) -> None:
        super().__init__(p=p)
        self._class_pairs = self._normalize_pairs(class_pairs)

    @staticmethod
    def _normalize_pairs(class_pairs: Sequence[Sequence[int]] | None) -> tuple[tuple[int, int], ...]:
        if class_pairs is None:
            return tuple()
        normalized: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for pair in class_pairs:
            if len(pair) != 2:
                raise ValueError("Each class pair must contain exactly two class indices.")
            left, right = int(pair[0]), int(pair[1])
            if left == right:
                continue
            key = (min(left, right), max(left, right))
            if key in seen:
                continue
            seen.add(key)
            normalized.append((left, right))
        return tuple(normalized)

    def __call__(self, inputs: Any) -> Any:
        """Apply a paired class swap only when the flip is actually applied."""
        image, target, dataset = _split_sample(inputs)
        flat_inputs, spec = tree_flatten((image, _as_tv_target(target, _image_size(image))))
        if torch.rand(1) >= self.transform.p:
            return inputs
        needs_transform_list = self.transform._needs_transform_list(flat_inputs)
        params = self.transform.make_params(
            [input_value for input_value, needs_transform in zip(flat_inputs, needs_transform_list) if needs_transform]
        )
        flat_outputs = [
            self.transform.transform(input_value, params) if needs_transform else input_value
            for input_value, needs_transform in zip(flat_inputs, needs_transform_list)
        ]
        image, target = tree_unflatten(flat_outputs, spec)
        target = _plain_target(target)
        if self._class_pairs and "labels" in target:
            labels = target["labels"]
            swapped = labels.clone()
            for left, right in self._class_pairs:
                left_mask = labels == left
                right_mask = labels == right
                swapped[left_mask] = right
                swapped[right_mask] = left
            target["labels"] = swapped
        return _pack_sample(image, target, dataset)


class Resize:
    """Resize images, boxes, and masks with DEIMv2-compatible mask origin handling."""

    def __init__(
        self,
        size: int | Sequence[int],
        interpolation: T.InterpolationMode = T.InterpolationMode.BILINEAR,
        max_size: int | None = None,
        antialias: bool | None = True,
        mask_resize_origin: str = "center",
    ) -> None:
        self.size = size
        self.interpolation = interpolation
        self.max_size = max_size
        self.antialias = antialias
        self.mask_resize_origin = mask_resize_origin

    def __call__(self, inputs: Any) -> Any:
        """Resize one sample."""
        image, target, dataset = _split_sample(inputs)
        in_height, in_width = _image_size(image)
        out_height, out_width = compute_resized_mask_output_size((in_height, in_width), self.size, self.max_size)
        image = TVF.resize(
            image,
            [out_height, out_width],
            interpolation=self.interpolation,
            max_size=None,
            antialias=self.antialias,
        )
        target = dict(target)
        if "boxes" in target:
            scale = target["boxes"].new_tensor([out_width / in_width, out_height / in_height] * 2)
            target["boxes"] = target["boxes"] * scale
        if "masks" in target:
            masks = target["masks"].to(torch.bool)
            if masks.numel() == 0:
                target["masks"] = masks.new_zeros((0, out_height, out_width), dtype=torch.bool)
            else:
                target["masks"] = resize_masks(
                    masks[:, None],
                    size=[out_height, out_width],
                    mode="nearest",
                    origin=self.mask_resize_origin,
                )[:, 0].to(torch.bool)
        target["size"] = torch.as_tensor([out_height, out_width], dtype=torch.int64)
        return _pack_sample(image, target, dataset)


class PadToSize:
    """Pad an image and aligned masks/boxes to an exact ``(width, height)`` size."""

    def __init__(self, size: int | Sequence[int], fill: int = 0, padding_mode: str = "constant") -> None:
        self.size = (int(size), int(size)) if isinstance(size, int) else (int(size[0]), int(size[1]))
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, inputs: Any) -> Any:
        """Pad one sample on the right and bottom."""
        image, target, dataset = _split_sample(inputs)
        height, width = _image_size(image)
        pad = [0, 0, max(self.size[0] - width, 0), max(self.size[1] - height, 0)]
        image = TVF.pad(image, padding=pad, fill=self.fill, padding_mode=self.padding_mode)
        target = dict(target)
        if "masks" in target and (pad[2] > 0 or pad[3] > 0):
            target["masks"] = F.pad(target["masks"], (0, pad[2], 0, pad[3]))
        target["padding"] = torch.tensor(pad)
        target["size"] = torch.as_tensor([height + pad[3], width + pad[2]], dtype=torch.int64)
        return _pack_sample(image, target, dataset)


class SanitizeBoundingBoxes:
    """Remove invalid boxes while keeping per-object target fields aligned."""

    def __init__(self, min_size: float = 1.0, min_area: float = 1.0, labels_getter: Any = "default") -> None:
        self.min_size = min_size
        self.min_area = min_area
        self.labels_getter = labels_getter

    def __call__(self, inputs: Any) -> Any:
        """Sanitize one sample."""
        image, target, dataset = _split_sample(inputs)
        if "boxes" not in target:
            return _pack_sample(image, target, dataset)
        boxes = torch.as_tensor(target["boxes"])
        if boxes.numel() == 0:
            return _pack_sample(image, _apply_keep(target, torch.zeros((0,), dtype=torch.bool)), dataset)
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        keep = (widths >= self.min_size) & (heights >= self.min_size) & (widths * heights >= self.min_area)
        return _pack_sample(image, _apply_keep(target, keep), dataset)


class ConvertPILImage:
    """Convert PIL images to float tensor images."""

    def __init__(self, dtype: str = "float32", scale: bool = True) -> None:
        self.dtype = dtype
        self.scale = scale

    def __call__(self, inputs: Any) -> Any:
        """Convert the sample image."""
        image, target, dataset = _split_sample(inputs)
        if isinstance(image, Image.Image):
            image = TVF.pil_to_tensor(image)
        if self.dtype == "float32":
            image = image.float()
        if self.scale:
            image = image / 255.0
        return _pack_sample(image, target, dataset)


class Normalize:
    """Normalize tensor images with ImageNet statistics."""

    def __init__(self, mean: Sequence[float], std: Sequence[float]) -> None:
        self.mean = list(mean)
        self.std = list(std)

    def __call__(self, inputs: Any) -> Any:
        """Normalize the sample image."""
        image, target, dataset = _split_sample(inputs)
        return _pack_sample(TVF.normalize(image, self.mean, self.std), target, dataset)


class ConvertBoxes:
    """Convert boxes to a requested format and optionally normalize by image size."""

    def __init__(self, fmt: str = "", normalize: bool = False) -> None:
        self.fmt = fmt.lower()
        self.normalize = normalize

    def __call__(self, inputs: Any) -> Any:
        """Convert target boxes."""
        image, target, dataset = _split_sample(inputs)
        target = dict(target)
        if "boxes" in target:
            boxes = target["boxes"].float()
            if self.fmt == "cxcywh":
                boxes = box_xyxy_to_cxcywh(boxes)
            elif self.fmt and self.fmt != "xyxy":
                boxes = torchvision.ops.box_convert(boxes, in_fmt="xyxy", out_fmt=self.fmt)
            if self.normalize:
                height, width = _image_size(image)
                boxes = boxes / boxes.new_tensor([width, height, width, height])
            target["boxes"] = boxes
        return _pack_sample(image, target, dataset)


class EmptyTransform:
    """No-op transform."""

    def __call__(self, inputs: Any) -> Any:
        """Return the input unchanged."""
        return inputs


class Mosaic:
    """DEIMv2-style four-image mosaic with optional cache and affine transform."""

    def __init__(
        self,
        output_size: int = 320,
        max_size: int | None = None,
        rotation_range: float = 0,
        translation_range: tuple[float, float] = (0.1, 0.1),
        scaling_range: tuple[float, float] = (0.5, 1.5),
        probability: float = 1.0,
        fill_value: int = 114,
        use_cache: bool = True,
        max_cached_images: int = 50,
        random_pop: bool = True,
        mask_resize_origin: str = "center",
    ) -> None:
        self.resize = Resize(output_size, max_size=max_size, mask_resize_origin=mask_resize_origin)
        self.probability = probability
        self.affine_transform = _TorchvisionWrapper(
            T.RandomAffine(
                degrees=rotation_range,
                translate=translation_range,
                scale=scaling_range,
                fill=fill_value,
            )
        )
        self.use_cache = use_cache
        self.mosaic_cache: list[dict[str, Any]] = []
        self.max_cached_images = max_cached_images
        self.random_pop = random_pop

    @staticmethod
    def _clone_target(target: dict[str, Any]) -> dict[str, Any]:
        return {key: value.clone() if torch.is_tensor(value) else copy.deepcopy(value) for key, value in target.items()}

    def state_dict(self) -> dict[str, Any]:
        """Return cache state for checkpointing."""
        return {
            "mosaic_cache": [
                {"img": sample["img"].copy(), "labels": self._clone_target(sample["labels"])}
                for sample in self.mosaic_cache
            ]
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore cache state."""
        cache = state_dict.get("mosaic_cache", [])
        self.mosaic_cache = [
            {"img": sample["img"].copy(), "labels": self._clone_target(sample["labels"])} for sample in cache
        ]

    def _samples_from_cache(self, image: Image.Image, target: dict[str, Any]) -> tuple[list[dict[str, Any]], int, int]:
        image, target = self.resize((image, target))
        self.mosaic_cache.append({"img": image, "labels": self._clone_target(target)})
        if len(self.mosaic_cache) > self.max_cached_images:
            self.mosaic_cache.pop(random.randint(0, len(self.mosaic_cache) - 2) if self.random_pop else 0)
        picks = random.choices(range(len(self.mosaic_cache)), k=3)
        samples = [{"img": image.copy(), "labels": self._clone_target(target)}]
        samples += [
            {
                "img": self.mosaic_cache[idx]["img"].copy(),
                "labels": self._clone_target(self.mosaic_cache[idx]["labels"]),
            }
            for idx in picks
        ]
        sizes = [_image_size(sample["img"]) for sample in samples]
        return samples, max(size[0] for size in sizes), max(size[1] for size in sizes)

    def _samples_from_dataset(
        self, image: Image.Image, target: dict[str, Any], dataset: Any
    ) -> tuple[list[dict[str, Any]], int, int]:
        image, target = self.resize((image, target))
        samples = [{"img": image, "labels": target}]
        for idx in random.choices(range(len(dataset)), k=3):
            sample_image, sample_target = self.resize(dataset.load_item(idx))
            samples.append({"img": sample_image, "labels": sample_target})
        sizes = [_image_size(sample["img"]) for sample in samples]
        return samples, max(size[0] for size in sizes), max(size[1] for size in sizes)

    def __call__(self, inputs: Any) -> Any:
        """Apply mosaic."""
        image, target, dataset = _split_sample(inputs)
        if self.probability < 1.0 and random.random() > self.probability:
            return inputs
        if self.use_cache:
            samples, max_height, max_width = self._samples_from_cache(image, target)
        else:
            if dataset is None:
                return inputs
            samples, max_height, max_width = self._samples_from_dataset(image, target, dataset)

        merged_image = Image.new(mode=samples[0]["img"].mode, size=(max_width * 2, max_height * 2), color=0)
        offsets = torch.tensor([[0, 0], [max_width, 0], [0, max_height], [max_width, max_height]], dtype=torch.float32)
        merged_targets: list[dict[str, Any]] = []
        mask_canvases: list[torch.Tensor] = []
        for idx, sample in enumerate(samples):
            x_off, y_off = int(offsets[idx, 0]), int(offsets[idx, 1])
            merged_image.paste(sample["img"], (x_off, y_off))
            sample_target = self._clone_target(sample["labels"])
            if "boxes" in sample_target:
                sample_target["boxes"] = sample_target["boxes"] + offsets[idx].repeat(2)
            if "masks" in sample_target:
                masks = sample_target["masks"]
                canvas = masks.new_zeros((masks.shape[0], max_height * 2, max_width * 2), dtype=torch.bool)
                canvas[:, y_off : y_off + masks.shape[-2], x_off : x_off + masks.shape[-1]] = masks
                sample_target["masks"] = canvas
            merged_targets.append(sample_target)

        merged_target: dict[str, Any] = {}
        for key in merged_targets[0]:
            values = [sample[key] for sample in merged_targets if key in sample]
            if values and torch.is_tensor(values[0]):
                merged_target[key] = torch.cat(values, dim=0)
            elif values:
                merged_target[key] = values
        if mask_canvases:
            merged_target["masks"] = torch.cat(mask_canvases, dim=0)
        merged_target["size"] = torch.as_tensor([max_height * 2, max_width * 2], dtype=torch.int64)
        return self.affine_transform(_pack_sample(merged_image, merged_target, dataset))


class Deimv2Compose:
    """Compose DEIMv2 transforms with epoch/sample policies."""

    def __init__(
        self,
        ops: Sequence[Any] | None,
        policy: dict[str, Any] | None = None,
        mosaic_prob: float = -0.1,
    ) -> None:
        self.transforms = list(ops) if ops is not None else [EmptyTransform()]
        self.mosaic_prob = mosaic_prob
        self.policy = policy or {"name": "default"}
        self.global_samples = 0

    def set_epoch(self, epoch: int) -> None:
        """Set the current epoch."""
        self._epoch = int(epoch)

    @property
    def epoch(self) -> int:
        """Current epoch, or -1 before training starts."""
        return getattr(self, "_epoch", -1)

    def state_dict(self) -> dict[str, Any]:
        """Return scheduler and transform state."""
        return {
            "global_samples": self.global_samples,
            "policy": copy.deepcopy(self.policy),
            "mosaic_prob": self.mosaic_prob,
            "transform_states": [
                transform.state_dict() if hasattr(transform, "state_dict") else None for transform in self.transforms
            ],
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore scheduler and transform state."""
        self.global_samples = state_dict.get("global_samples", 0)
        self.policy = copy.deepcopy(state_dict.get("policy", self.policy))
        self.mosaic_prob = state_dict.get("mosaic_prob", self.mosaic_prob)
        for transform, transform_state in zip(self.transforms, state_dict.get("transform_states", [])):
            if transform_state is not None and hasattr(transform, "load_state_dict"):
                transform.load_state_dict(transform_state)

    def __call__(self, *inputs: Any) -> Any:
        """Apply the configured policy."""
        sample = inputs if len(inputs) > 1 else inputs[0]
        name = self.policy.get("name", "default")
        if name == "stop_epoch":
            return self._stop_epoch_forward(sample)
        if name == "stop_sample":
            return self._stop_sample_forward(sample)
        for transform in self.transforms:
            sample = transform(sample)
        return sample

    def _stop_epoch_forward(self, sample: Any) -> Any:
        dataset = _split_sample(sample)[2]
        cur_epoch = int(getattr(dataset, "epoch", self.epoch))
        policy_ops = set(self.policy.get("ops", []))
        policy_epoch = self.policy.get("epoch", 0)
        if isinstance(policy_epoch, list) and len(policy_epoch) == 3:
            with_mosaic = policy_epoch[0] <= cur_epoch < policy_epoch[1] and random.random() <= self.mosaic_prob
            for transform in self.transforms:
                name = type(transform).__name__
                if name in policy_ops and (cur_epoch < policy_epoch[0] or cur_epoch >= policy_epoch[-1]):
                    continue
                if name == "Mosaic" and not with_mosaic:
                    continue
                if name in {"RandomZoomOut", "RandomIoUCrop"} and with_mosaic:
                    continue
                sample = transform(sample)
            return sample
        for transform in self.transforms:
            if type(transform).__name__ in policy_ops and cur_epoch >= int(policy_epoch):
                continue
            sample = transform(sample)
        return sample

    def _stop_sample_forward(self, sample: Any) -> Any:
        policy_ops = set(self.policy.get("ops", []))
        policy_sample = int(self.policy.get("sample", 0))
        for transform in self.transforms:
            if type(transform).__name__ in policy_ops and self.global_samples >= policy_sample:
                continue
            sample = transform(sample)
        self.global_samples += 1
        return sample


_TRANSFORM_REGISTRY: dict[str, type] = {
    cls.__name__: cls
    for cls in (
        RandomPhotometricDistort,
        RandomZoomOut,
        RandomIoUCrop,
        RandomHorizontalFlip,
        RandomHorizontalFlipWithClass,
        RandomCrop,
        Resize,
        PadToSize,
        SanitizeBoundingBoxes,
        ConvertPILImage,
        Normalize,
        ConvertBoxes,
        EmptyTransform,
        Mosaic,
    )
}


def build_deimv2_transform_ops(
    entries: Sequence[dict[str, Any]],
    *,
    mask_resize_origin: str,
) -> list[Any]:
    """Instantiate DEIMv2 transform ops from ``[{type: ...}]`` config entries."""
    transforms = []
    for entry in entries:
        params = dict(entry)
        name = params.pop("type")
        cls = _TRANSFORM_REGISTRY[name]
        if name in {"Resize", "Mosaic"}:
            params.setdefault("mask_resize_origin", mask_resize_origin)
        transforms.append(cls(**params))
    return transforms


class Deimv2CollateFunction:
    """RF-DETR collate wrapper with DEIMv2 MixUp/CopyBlend and no collate multi-scale resize."""

    def __init__(
        self,
        block_size: int | None,
        *,
        mixup_prob: float = 0.0,
        mixup_epochs: Sequence[int] = (4, 29),
        copyblend_prob: float = 0.0,
        copyblend_epochs: Sequence[int] = (4, 50),
        copyblend_type: str = "blend",
        conflict_with_mixup: bool = False,
        area_threshold: float = 100,
        num_objects: int = 3,
        with_expand: bool = False,
        expand_ratios: Sequence[float] = (0.1, 0.25),
        random_num_objects: bool = False,
    ) -> None:
        self.block_size = block_size
        self.mixup_prob = mixup_prob
        self.mixup_epochs = tuple(int(v) for v in mixup_epochs)
        self.copyblend_prob = copyblend_prob
        self.copyblend_epochs = tuple(int(v) for v in copyblend_epochs)
        self.copyblend_type = copyblend_type
        self.conflict_with_mixup = conflict_with_mixup
        self.area_threshold = area_threshold
        self.num_objects = num_objects
        self.with_expand = with_expand
        self.expand_ratios = tuple(float(v) for v in expand_ratios)
        self.random_num_objects = random_num_objects
        self._epoch = -1

    def set_epoch(self, epoch: int) -> None:
        """Set the current epoch."""
        self._epoch = int(epoch)

    @property
    def epoch(self) -> int:
        """Return the current epoch."""
        return self._epoch

    def __call__(self, items: list[tuple[torch.Tensor, dict[str, Any]]]) -> tuple[Any, ...]:
        """Apply collate-time augmentations and delegate padding to RF-DETR."""
        images = [item[0] for item in items]
        targets = [item[1] for item in items]
        if images and all(image.shape == images[0].shape for image in images):
            batch_images = torch.stack(images)
            batch_images, targets = self._apply_mixup_or_copyblend(batch_images, targets)
            items = list(zip(list(batch_images), targets))
        return _collate_with_block_size(items, block_size=self.block_size)

    def _apply_mixup_or_copyblend(
        self, images: torch.Tensor, targets: list[dict[str, Any]]
    ) -> tuple[torch.Tensor, list[dict[str, Any]]]:
        beta = round(random.uniform(0.45, 0.55), 6)
        if (
            self.mixup_prob > 0
            and self.mixup_epochs[0] <= self.epoch < self.mixup_epochs[-1]
            and random.random() < self.mixup_prob
        ):
            mixed = images.roll(shifts=1, dims=0).mul(1.0 - beta).add(images.mul(beta))
            shifted_targets = targets[-1:] + targets[:-1]
            return mixed, [self._merge_targets(a, b, beta) for a, b in zip(targets, shifted_targets)]

        if (
            self.copyblend_prob > 0
            and self.copyblend_epochs[0] <= self.epoch < self.copyblend_epochs[-1]
            and random.random() < self.copyblend_prob
        ):
            return self._apply_copyblend(images, targets, beta)
        return images, targets

    @staticmethod
    def _merge_targets(first: dict[str, Any], second: dict[str, Any], beta: float) -> dict[str, Any]:
        merged = copy.deepcopy(first)
        first_len = int(first["labels"].shape[0])
        second_len = int(second["labels"].shape[0])
        for key in _OBJECT_LEVEL_TARGET_KEYS:
            if key in first and key in second and torch.is_tensor(first[key]) and torch.is_tensor(second[key]):
                merged[key] = torch.cat([first[key], second[key]], dim=0)
        merged["boxes"] = torch.cat([first["boxes"], second["boxes"]], dim=0)
        merged["labels"] = torch.cat([first["labels"], second["labels"]], dim=0)
        merged["mixup"] = torch.tensor([beta] * first_len + [1.0 - beta] * second_len, dtype=torch.float32)
        return merged

    def _apply_copyblend(
        self, images: torch.Tensor, targets: list[dict[str, Any]], beta: float
    ) -> tuple[torch.Tensor, list[dict[str, Any]]]:
        objects_pool: defaultdict[str, list[Any]] = defaultdict(list)
        _, _, img_height, img_width = images.shape
        for image_idx, target in enumerate(targets):
            for object_idx, (box, label, area) in enumerate(zip(target["boxes"], target["labels"], target["area"])):
                if float(area) < self.area_threshold:
                    continue
                objects_pool["boxes"].append(box)
                objects_pool["labels"].append(label)
                objects_pool["areas"].append(area)
                objects_pool["image_idx"].append(image_idx)
                objects_pool["object_idx"].append(object_idx)
        if not objects_pool["boxes"]:
            return images, targets

        updated_images = images.clone()
        updated_targets = copy.deepcopy(targets)
        for dest_idx in range(len(images)):
            count = (
                random.randint(1, min(self.num_objects, len(objects_pool["boxes"])))
                if self.random_num_objects
                else self.num_objects
            )
            selected = random.sample(range(len(objects_pool["boxes"])), min(count, len(objects_pool["boxes"])))
            additions: defaultdict[str, list[torch.Tensor]] = defaultdict(list)
            for pool_idx in selected:
                box = objects_pool["boxes"][pool_idx]
                source_idx = int(objects_pool["image_idx"][pool_idx])
                object_idx = int(objects_pool["object_idx"][pool_idx])
                cx, cy, width, height = box
                x1_src = max(int((cx - width / 2) * img_width), 0)
                y1_src = max(int((cy - height / 2) * img_height), 0)
                x2_src = min(int((cx + width / 2) * img_width), img_width)
                y2_src = min(int((cy + height / 2) * img_height), img_height)
                patch_width, patch_height = x2_src - x1_src, y2_src - y1_src
                if patch_width <= 0 or patch_height <= 0:
                    continue
                x1 = random.randint(0, img_width - patch_width) if patch_width < img_width else 0
                y1 = random.randint(0, img_height - patch_height) if patch_height < img_height else 0
                x2, y2 = x1 + patch_width, y1 + patch_height
                source_patch = images[source_idx, :, y1_src:y2_src, x1_src:x2_src]
                if self.copyblend_type == "blend":
                    updated_images[dest_idx, :, y1:y2, x1:x2] = updated_images[
                        dest_idx, :, y1:y2, x1:x2
                    ] * beta + source_patch * (1.0 - beta)
                else:
                    updated_images[dest_idx, :, y1:y2, x1:x2] = source_patch
                additions["boxes"].append(
                    box.new_tensor(
                        [
                            (x1 + patch_width / 2) / img_width,
                            (y1 + patch_height / 2) / img_height,
                            patch_width / img_width,
                            patch_height / img_height,
                        ]
                    )
                )
                for key, source_key in (("labels", "labels"), ("area", "areas")):
                    additions[key].append(objects_pool[source_key][pool_idx].clone())
                for key in ("mask_valid", "segm_eval_valid", "iscrowd"):
                    if key in targets[source_idx]:
                        additions[key].append(targets[source_idx][key][object_idx].clone())
                if "masks" in targets[source_idx]:
                    mask = targets[source_idx]["masks"][object_idx]
                    canvas = mask.new_zeros((img_height, img_width), dtype=torch.bool)
                    canvas[y1:y2, x1:x2] = mask[y1_src:y2_src, x1_src:x2_src]
                    additions["masks"].append(canvas)
            for key, values in additions.items():
                if values:
                    updated_targets[dest_idx][key] = torch.cat([updated_targets[dest_idx][key], torch.stack(values)])
            if additions["boxes"]:
                base_len = len(updated_targets[dest_idx]["boxes"]) - len(additions["boxes"])
                updated_targets[dest_idx]["mixup"] = torch.tensor(
                    [1.0] * base_len + [1.0 - beta] * len(additions["boxes"]),
                    dtype=torch.float32,
                )
        return updated_images, updated_targets


def make_deimv2_collate_fn(block_size: int | None = None, **kwargs: Any) -> Deimv2CollateFunction:
    """Build a DEIMv2 collate function.

    This intentionally does not implement DEIMv2's ``base_size_repeat`` multi-scale resize; RF-DETR's model-side
    multi-scale path remains authoritative.
    """
    return Deimv2CollateFunction(block_size=block_size, **kwargs)


def make_standard_collate_fn(block_size: int | None = None) -> Callable[[list[tuple[Any, ...]]], tuple[Any, ...]]:
    """Return RF-DETR's normal block-size collate function."""
    return partial(_collate_with_block_size, block_size=block_size)
