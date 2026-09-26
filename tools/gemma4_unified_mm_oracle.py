#!/usr/bin/env python3
"""Reference activations for the Gemma 4 12B «unified» image and audio embedders.

The 12B has no vision or audio tower: raw 48x48 pixel patches and raw 640-sample
waveform frames are projected straight into the decoder's hidden size. This tool
loads only those 11 tensors, runs the official processors and modules on
deterministic fixtures, and writes every stage the engine must reproduce.

Binary outputs live outside Git; the manifest carries hashes and summaries.
Needs torch, torchvision, transformers>=5.17 (gemma4_unified), safetensors, PIL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
from PIL import Image
from safetensors import safe_open
from transformers import AutoFeatureExtractor, AutoImageProcessor
from transformers.models.gemma4_unified.configuration_gemma4_unified import Gemma4UnifiedConfig
from transformers.models.gemma4_unified.modeling_gemma4_unified import (
    Gemma4UnifiedMultimodalEmbedder,
    Gemma4UnifiedVisionEmbedder,
)

# Two images: one already aligned to 48 px (no resize) and one that exercises
# the aspect-preserving resize. The audio is not a multiple of 640 samples, so
# the last frame is padded and masked.
IMAGES = {"aligned_960x672": (960, 672), "resized_1234x567": (1234, 567)}
AUDIO_SAMPLES = 32_000 + 333
SAMPLING_RATE = 16_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gemma 4 12B unified embedder oracle")
    parser.add_argument("model_dir", type=Path, help="Local Gemma 4 12B unified checkpoint")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.home() / ".cache/helios/gemma4_unified_mm_oracle/fp32",
        help="Directory for binary activations and manifest (outside Git)",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_image(width: int, height: int) -> np.ndarray:
    """Deterministic, nontrivial RGB8: every channel varies along both axes."""
    y, x = np.indices((height, width), dtype=np.uint32)
    red = (17 * x + 3 * y + np.bitwise_xor(x, y)) & 0xFF
    green = (5 * x + 29 * y + ((x * y) >> 5)) & 0xFF
    blue = (11 * x + 7 * y + np.bitwise_xor(x >> 2, y << 1)) & 0xFF
    return np.stack((red, green, blue), axis=-1).astype(np.uint8)


def make_audio() -> np.ndarray:
    """A chirp plus a seeded low-level noise floor, in [-1, 1]."""
    t = np.arange(AUDIO_SAMPLES, dtype=np.float64) / SAMPLING_RATE
    chirp = 0.6 * np.sin(2 * np.pi * (220 * t + 400 * t * t))
    noise = 0.05 * np.random.default_rng(1234).standard_normal(AUDIO_SAMPLES)
    return np.clip(chirp + noise, -1.0, 1.0).astype(np.float32)


def stats(array: np.ndarray) -> dict[str, Any]:
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


def save(output_dir: Path, name: str, value: torch.Tensor | np.ndarray) -> dict[str, Any]:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().contiguous().numpy()
    value = np.ascontiguousarray(value)
    path = output_dir / f"{name}.npy"
    np.save(path, value, allow_pickle=False)
    result = stats(value)
    result.update({"file": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return result


def load(module: torch.nn.Module, checkpoint: Path, names: dict[str, str]) -> None:
    """Load checkpoint tensors (bf16) into module parameters, upcast to fp32."""
    state = {}
    with safe_open(str(checkpoint), framework="pt", device="cpu") as source:
        for target, stored in names.items():
            state[target] = source.get_tensor(stored).float()
    result = module.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"state mismatch: {result}")


def main() -> None:
    args = parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = args.model_dir / "model.safetensors"
    config = Gemma4UnifiedConfig.from_pretrained(args.model_dir)
    torch.manual_seed(0)

    vision = Gemma4UnifiedVisionEmbedder(config.vision_config, config.text_config).float().eval()
    load(vision, checkpoint, {
        "patch_ln1.weight": "model.vision_embedder.patch_ln1.weight",
        "patch_ln1.bias": "model.vision_embedder.patch_ln1.bias",
        "patch_dense.weight": "model.vision_embedder.patch_dense.weight",
        "patch_dense.bias": "model.vision_embedder.patch_dense.bias",
        "patch_ln2.weight": "model.vision_embedder.patch_ln2.weight",
        "patch_ln2.bias": "model.vision_embedder.patch_ln2.bias",
        "pos_embedding": "model.vision_embedder.pos_embedding",
        "pos_norm.weight": "model.vision_embedder.pos_norm.weight",
        "pos_norm.bias": "model.vision_embedder.pos_norm.bias",
        "multimodal_embedder.embedding_projection.weight":
            "model.embed_vision.embedding_projection.weight",
    })
    audio = Gemma4UnifiedMultimodalEmbedder(config.audio_config, config.text_config).float().eval()
    load(audio, checkpoint, {"embedding_projection.weight": "model.embed_audio.embedding_projection.weight"})

    images = AutoImageProcessor.from_pretrained(args.model_dir)
    features = AutoFeatureExtractor.from_pretrained(args.model_dir)
    manifest: dict[str, Any] = {
        "schema": "helios.gemma4_unified_mm_oracle.v1",
        "model_config_sha256": sha256_file(args.model_dir / "config.json"),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "transformers": transformers.__version__},
        "compute_dtype": "fp32 (bf16 checkpoint upcast)",
        "images": {},
        "audio": {},
    }

    stages: dict[str, torch.Tensor] = {}
    hooks = [
        vision.patch_ln2.register_forward_hook(lambda m, i, o: stages.__setitem__("patch", o)),
        vision.pos_norm.register_forward_hook(lambda m, i, o: stages.__setitem__("pos", o)),
    ]
    with torch.no_grad():
        for name, (width, height) in IMAGES.items():
            rgb = make_image(width, height)
            batch = images(images=Image.fromarray(rgb), return_tensors="pt")
            pixels, positions = batch["pixel_values"], batch["image_position_ids"]
            result = vision(pixels, positions, return_dict=True)
            valid = (positions[0] != -1).all(-1)
            manifest["images"][name] = {
                "width": width, "height": height,
                "soft_tokens": int(valid.sum()),
                "rgb8": save(out, f"{name}.rgb8", rgb),
                "pixel_values": save(out, f"{name}.pixel_values", pixels[0][valid].float()),
                "positions": save(out, f"{name}.positions", positions[0][valid].to(torch.int32)),
                "after_patch_ln2": save(out, f"{name}.after_patch_ln2", stages["patch"][0][valid]),
                "after_pos_norm": save(out, f"{name}.after_pos_norm", stages["pos"][0][valid]),
                "embeddings": save(out, f"{name}.embeddings", result.pooler_output[0][valid]),
            }

        wave = make_audio()
        batch = features(wave, sampling_rate=SAMPLING_RATE, return_tensors="pt")
        frames, mask = batch["input_features"][0], batch["input_features_mask"][0]
        embedded = audio(frames[mask].unsqueeze(0))[0]
        manifest["audio"] = {
            "samples": AUDIO_SAMPLES, "sampling_rate": SAMPLING_RATE,
            "tokens": int(mask.sum()),
            "wave": save(out, "audio.wave_f32", wave),
            "frames": save(out, "audio.frames", frames[mask]),
            "embeddings": save(out, "audio.embeddings", embedded),
        }
    for hook in hooks:
        hook.remove()

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for name, image in manifest["images"].items():
        print(name, image["soft_tokens"], "tokens; embeddings rms", round(image["embeddings"]["rms"], 4))
    print("audio", manifest["audio"]["tokens"], "tokens; embeddings rms",
          round(manifest["audio"]["embeddings"]["rms"], 4))
    print("manifest:", out / "manifest.json")


if __name__ == "__main__":
    main()
