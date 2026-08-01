#!/usr/bin/env python3
"""Generate V3 resize/patch fixtures with PyTorch's pinned uint8 AA kernel.

This tool needs only PyTorch 2.9.1, not the Gemma checkpoint or torchvision.
Its geometry is copied from the pinned Transformers Gemma 4 processor and the
resize operator is the exact ATen primitive used by torchvision for RGB8.
Large resized images live outside Git; only the script and compact report are
versioned.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch


TRANSFORMERS_COMMIT = "b3a36037d3feb22e3f0174b3dd4248fcc0f0f722"
PYTORCH_VERSION = "2.9.1"
PATCH_SIZE = 16
POOLING = 3
MAX_SOFT_TOKENS = 280
MAX_PATCHES = MAX_SOFT_TOKENS * POOLING**2
RESCALE_FACTOR = 1.0 / 255.0

FIXTURES = {
    "aligned": (960, 672),
    "square": (512, 512),
    "portrait": (333, 1000),
    "panorama": (1600, 300),
    "extreme": (5000, 1),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gemma 4 V3 preprocessing oracle")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.home() / ".cache/helios/gemma4_vision_preprocess_v3",
    )
    return parser.parse_args()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_fixture(width: int, height: int) -> torch.Tensor:
    y = torch.arange(height, dtype=torch.int64).reshape(height, 1)
    x = torch.arange(width, dtype=torch.int64).reshape(1, width)
    red = (17 * x + 3 * y + torch.bitwise_xor(x, y)) & 0xFF
    green = (5 * x + 29 * y + ((x * y) >> 5)) & 0xFF
    blue = (11 * x + 7 * y + torch.bitwise_xor(x >> 2, y << 1)) & 0xFF
    return torch.stack((red, green, blue), dim=0).to(torch.uint8)


def target_size(width: int, height: int) -> tuple[int, int]:
    target_px = MAX_PATCHES * PATCH_SIZE**2
    factor = math.sqrt(target_px / (height * width))
    side_mult = POOLING * PATCH_SIZE
    target_height = int(math.floor(factor * height / side_mult)) * side_mult
    target_width = int(math.floor(factor * width / side_mult)) * side_mult
    if target_height == 0 and target_width == 0:
        raise ValueError("both resized dimensions rounded to zero")
    max_side = (MAX_PATCHES // POOLING**2) * side_mult
    if target_height == 0:
        target_height = side_mult
        target_width = min(math.floor(width / height) * side_mult, max_side)
    elif target_width == 0:
        target_width = side_mult
        target_height = min(math.floor(height / width) * side_mult, max_side)
    if target_height * target_width > target_px:
        raise ValueError("target exceeds patch budget")
    return target_width, target_height


def raw_bytes(tensor: torch.Tensor) -> bytes:
    value = tensor.detach().cpu().contiguous()
    if value.storage_offset() != 0:
        value = value.clone()
    return bytes(value.untyped_storage())


def process(name: str, width: int, height: int, output_dir: Path) -> dict:
    source = make_fixture(width, height)
    target_width, target_height = target_size(width, height)
    if (target_width, target_height) == (width, height):
        resized = source
    else:
        resized = torch.ops.aten._upsample_bicubic2d_aa(
            source.unsqueeze(0), [target_height, target_width], False, None, None
        ).squeeze(0)
    resized_hwc = resized.permute(1, 2, 0).contiguous()
    resized_data = raw_bytes(resized_hwc)
    resized_path = output_dir / f"{name}.rgb"
    resized_path.write_bytes(resized_data)

    image = resized.float().mul(RESCALE_FACTOR)
    patch_height = target_height // PATCH_SIZE
    patch_width = target_width // PATCH_SIZE
    patches = image.reshape(3, patch_height, PATCH_SIZE, patch_width, PATCH_SIZE)
    patches = patches.permute(1, 3, 2, 4, 0).reshape(-1, PATCH_SIZE * PATCH_SIZE * 3)
    positions = torch.stack(torch.meshgrid(
        torch.arange(patch_width), torch.arange(patch_height), indexing="xy"
    ), dim=-1).reshape(-1, 2)
    soft_tokens = patches.shape[0] // POOLING**2
    if patches.shape[0] < MAX_PATCHES:
        patches = torch.nn.functional.pad(
            patches, (0, 0, 0, MAX_PATCHES - patches.shape[0]), value=0
        )
        positions = torch.nn.functional.pad(
            positions, (0, 0, 0, MAX_PATCHES - positions.shape[0]), value=-1
        )

    return {
        "source_width": width,
        "source_height": height,
        "target_width": target_width,
        "target_height": target_height,
        "patch_columns": patch_width,
        "patch_rows": patch_height,
        "real_patches": patch_width * patch_height,
        "max_patches": MAX_PATCHES,
        "soft_tokens": soft_tokens,
        "resized_file": resized_path.name,
        "resized_bytes": len(resized_data),
        "resized_sha256": sha256(resized_data),
        "patches_sha256": sha256(raw_bytes(patches)),
        "positions_sha256": sha256(raw_bytes(positions)),
    }


def main() -> int:
    if not torch.__version__.startswith(PYTORCH_VERSION):
        raise RuntimeError(f"expected PyTorch {PYTORCH_VERSION}, got {torch.__version__}")
    output_dir = parse_args().output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    fixtures = {
        name: process(name, width, height, output_dir)
        for name, (width, height) in FIXTURES.items()
    }
    manifest = {
        "contract": "gemma4-vision-preprocess-v3",
        "transformers_commit": TRANSFORMERS_COMMIT,
        "torch": torch.__version__,
        "operator": "aten._upsample_bicubic2d_aa",
        "input_dtype": "uint8",
        "patch_size": PATCH_SIZE,
        "pooling_kernel_size": POOLING,
        "max_soft_tokens": MAX_SOFT_TOKENS,
        "max_patches": MAX_PATCHES,
        "rescale_factor": RESCALE_FACTOR,
        "fixtures": fixtures,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
