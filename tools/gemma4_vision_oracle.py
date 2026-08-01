#!/usr/bin/env python3
"""Generate deterministic Gemma 4 vision reference activations.

This tool deliberately loads only the vision tower and its projection into the
text hidden size. Large binary outputs live outside Git; the generated manifest
contains the hashes and numerical summaries used by Hector's V0 contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
from safetensors import safe_open
from transformers import Gemma4Config
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4MultimodalEmbedder,
    Gemma4VisionModel,
)


UPSTREAM_COMMIT = "b3a36037d3feb22e3f0174b3dd4248fcc0f0f722"
FIXTURE_WIDTH = 960
FIXTURE_HEIGHT = 672
EXPECTED_PATCHES = 2520
EXPECTED_SOFT_TOKENS = 280
EXPECTED_VISION_TENSORS = 659


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the pinned Gemma 4 E2B vision oracle",
    )
    parser.add_argument("model_dir", type=Path, help="Local Gemma 4 checkpoint")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.home() / ".cache/helios/gemma4_vision_oracle/fp32",
        help="Directory for binary activations and manifest (outside Git)",
    )
    parser.add_argument("--device", default="cpu", choices=("cuda", "cpu"))
    parser.add_argument("--dtype", default="fp32", choices=("fp32", "fp16", "bf16"))
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_fixture() -> np.ndarray:
    """Return a deterministic, nontrivial uint8 RGB image."""
    y, x = np.indices((FIXTURE_HEIGHT, FIXTURE_WIDTH), dtype=np.uint32)
    red = (17 * x + 3 * y + np.bitwise_xor(x, y)) & 0xFF
    green = (5 * x + 29 * y + ((x * y) >> 5)) & 0xFF
    blue = (11 * x + 7 * y + np.bitwise_xor(x >> 2, y << 1)) & 0xFF
    return np.stack((red, green, blue), axis=-1).astype(np.uint8)


def tensor_from_output(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            try:
                return tensor_from_output(item)
            except TypeError:
                pass
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    raise TypeError(f"hook output has no tensor: {type(output)!r}")


def load_prefixed_state(
    module: torch.nn.Module,
    model_file: Path,
    prefix: str,
) -> int:
    state: dict[str, torch.Tensor] = {}
    with safe_open(str(model_file), framework="pt", device="cpu") as checkpoint:
        for name in checkpoint.keys():
            if name.startswith(prefix):
                state[name.removeprefix(prefix)] = checkpoint.get_tensor(name)

    incompatible = module.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"state mismatch for {prefix}: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    return len(state)


def tensor_stats(array: np.ndarray) -> dict[str, Any]:
    numeric = array.astype(np.float64, copy=False)
    return {
        "shape": list(array.shape),
        "stored_dtype": str(array.dtype),
        "finite": bool(np.isfinite(numeric).all()),
        "min": float(numeric.min()),
        "max": float(numeric.max()),
        "mean": float(numeric.mean()),
        "rms": float(math.sqrt(np.mean(numeric * numeric))),
    }


def save_array(output_dir: Path, name: str, tensor: torch.Tensor | np.ndarray) -> dict[str, Any]:
    if isinstance(tensor, torch.Tensor):
        value = tensor.detach().float().cpu().contiguous().numpy()
    else:
        value = np.ascontiguousarray(tensor)
    path = output_dir / f"{name}.npy"
    np.save(path, value, allow_pickle=False)
    result = tensor_stats(value)
    result.update({"file": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return result


def preprocess_fixture(model_dir: Path, fixture: np.ndarray) -> dict[str, Any]:
    """Apply the pinned processor contract to the already aligned fixture.

    The fixture needs no resize, so this is exactly the official rescale,
    patchify, XY-position and padding path without importing torchvision.
    """
    processor_path = model_dir / "processor_config.json"
    root = json.loads(processor_path.read_text(encoding="utf-8"))
    image_config = dict(root["image_processor"])
    expected = {
        "do_convert_rgb": True,
        "do_normalize": False,
        "do_rescale": True,
        "do_resize": True,
        "patch_size": 16,
        "pooling_kernel_size": 3,
        "max_soft_tokens": 280,
        "resample": 3,
    }
    mismatches = {
        key: (image_config.get(key), value)
        for key, value in expected.items()
        if image_config.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"processor contract changed: {mismatches}")

    patch_size = int(image_config["patch_size"])
    pooling = int(image_config["pooling_kernel_size"])
    max_patches = int(image_config["max_soft_tokens"]) * pooling**2
    image = torch.from_numpy(fixture).permute(2, 0, 1).float()
    image *= float(image_config["rescale_factor"])
    patch_height = image.shape[-2] // patch_size
    patch_width = image.shape[-1] // patch_size
    patches = image.reshape(3, patch_height, patch_size, patch_width, patch_size)
    patches = patches.permute(1, 3, 2, 4, 0).reshape(patch_height * patch_width, -1)
    grid = torch.meshgrid(torch.arange(patch_width), torch.arange(patch_height), indexing="xy")
    positions = torch.stack(grid, dim=-1).reshape(patches.shape[0], 2)
    soft_tokens = patches.shape[0] // pooling**2
    if patches.shape[0] < max_patches:
        patch_pad = max_patches - patches.shape[0]
        patches = torch.nn.functional.pad(patches, (0, 0, 0, patch_pad), value=0)
        positions = torch.nn.functional.pad(positions, (0, 0, 0, patch_pad), value=-1)
    return {
        "pixel_values": patches.unsqueeze(0),
        "image_position_ids": positions.unsqueeze(0),
        "num_soft_tokens_per_image": [soft_tokens],
    }


def main() -> int:
    args = parse_args()
    model_dir = args.model_dir.resolve()
    model_file = model_dir / "model.safetensors"
    config_file = model_dir / "config.json"
    if not model_file.is_file() or not config_file.is_file():
        raise FileNotFoundError(f"incomplete checkpoint directory: {model_dir}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot access a GPU")

    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"output directory is not empty: {output_dir}; choose a fresh directory"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[args.dtype]
    device = torch.device(args.device)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False

    started = time.perf_counter()
    config = Gemma4Config.from_pretrained(model_dir, local_files_only=True)
    config.vision_config._attn_implementation = "eager"

    fixture = make_fixture()
    processed = preprocess_fixture(model_dir, fixture)
    pixel_values = processed["pixel_values"]
    position_ids = processed["image_position_ids"]
    soft_tokens_value = processed["num_soft_tokens_per_image"]
    if isinstance(soft_tokens_value, torch.Tensor):
        soft_tokens = [int(v) for v in soft_tokens_value.flatten().tolist()]
    else:
        soft_tokens = [int(v) for v in soft_tokens_value]

    expected_patch_shape = (1, EXPECTED_PATCHES, 3 * config.vision_config.patch_size**2)
    if tuple(pixel_values.shape) != expected_patch_shape:
        raise RuntimeError(f"unexpected patch shape: {tuple(pixel_values.shape)} != {expected_patch_shape}")
    if soft_tokens != [EXPECTED_SOFT_TOKENS]:
        raise RuntimeError(f"unexpected soft-token count: {soft_tokens}")

    vision_model = Gemma4VisionModel(config.vision_config).to(dtype=dtype)
    projector = Gemma4MultimodalEmbedder(config.vision_config, config.text_config).to(dtype=dtype)
    vision_count = load_prefixed_state(vision_model, model_file, "model.vision_tower.")
    projector_count = load_prefixed_state(projector, model_file, "model.embed_vision.")
    if vision_count + projector_count != EXPECTED_VISION_TENSORS:
        raise RuntimeError(
            f"loaded {vision_count + projector_count} visual tensors, expected {EXPECTED_VISION_TENSORS}"
        )

    captured: dict[str, torch.Tensor] = {}

    def capture(name: str):
        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            captured[name] = tensor_from_output(output).detach().cpu()

        return hook

    handles = [
        vision_model.patch_embedder.register_forward_hook(capture("patch_embedder")),
        vision_model.encoder.layers[0].register_forward_hook(capture("layer00")),
        vision_model.encoder.layers[-1].register_forward_hook(capture("layer15")),
        vision_model.pooler.register_forward_hook(capture("pooler")),
    ]

    vision_model.eval().to(device)
    projector.eval().to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    forward_started = time.perf_counter()
    with torch.inference_mode():
        vision_output = vision_model(
            pixel_values=pixel_values.to(device),
            pixel_position_ids=position_ids.to(device),
        ).last_hidden_state
        projected = projector(vision_output)
        if device.type == "cuda":
            torch.cuda.synchronize()
    forward_seconds = time.perf_counter() - forward_started
    for handle in handles:
        handle.remove()

    if tuple(vision_output.shape) != (EXPECTED_SOFT_TOKENS, config.vision_config.hidden_size):
        raise RuntimeError(f"unexpected pooled vision shape: {tuple(vision_output.shape)}")
    if tuple(projected.shape) != (EXPECTED_SOFT_TOKENS, config.text_config.hidden_size):
        raise RuntimeError(f"unexpected projected shape: {tuple(projected.shape)}")
    expected_captures = {"patch_embedder", "layer00", "layer15", "pooler"}
    if set(captured) != expected_captures:
        raise RuntimeError(f"missing captures: {sorted(expected_captures - set(captured))}")

    artifacts = {
        "fixture_rgb": save_array(output_dir, "fixture_rgb", fixture),
        "patches": save_array(output_dir, "patches", pixel_values),
        "positions": save_array(output_dir, "positions", position_ids.numpy()),
        "patch_embedder": save_array(output_dir, "patch_embedder", captured["patch_embedder"]),
        "layer00": save_array(output_dir, "layer00", captured["layer00"]),
        "layer15": save_array(output_dir, "layer15", captured["layer15"]),
        "pooler": save_array(output_dir, "pooler", captured["pooler"]),
        "projected": save_array(output_dir, "projected", projected),
    }
    if not all(item["finite"] for item in artifacts.values()):
        raise RuntimeError("oracle produced non-finite values")

    manifest = {
        "contract": "gemma4-vision-oracle-v1",
        "upstream_transformers_commit": UPSTREAM_COMMIT,
        "checkpoint": {
            "directory": str(model_dir),
            "model_file": model_file.name,
            "model_bytes": model_file.stat().st_size,
            "model_sha256": sha256_file(model_file),
            "config_sha256": sha256_file(config_file),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "gpu_capability": list(torch.cuda.get_device_capability(device)) if device.type == "cuda" else None,
            "compute_dtype": str(dtype),
            "attention_implementation": "eager",
            "deterministic_algorithms": True,
            "tf32": False,
        },
        "fixture": {
            "width": FIXTURE_WIDTH,
            "height": FIXTURE_HEIGHT,
            "soft_tokens": soft_tokens,
            "patches": EXPECTED_PATCHES,
        },
        "loaded_tensors": {
            "vision_tower": vision_count,
            "projector": projector_count,
            "total": vision_count + projector_count,
        },
        "timing": {
            "forward_seconds": forward_seconds,
            "total_seconds": time.perf_counter() - started,
        },
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        "artifacts": artifacts,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "manifest": str(manifest_path),
        "projected_sha256": artifacts["projected"]["sha256"],
        "soft_tokens": soft_tokens,
        "peak_cuda_bytes": manifest["peak_cuda_bytes"],
        "forward_seconds": forward_seconds,
    }, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
