from __future__ import annotations

import numpy as np


def hex_to_rgb(color_hex: str) -> tuple[int, int, int]:
    color = color_hex.strip().lstrip("#")
    if len(color) != 6:
        raise ValueError(f"Expected a 6-digit hex color, got {color_hex!r}")
    return tuple(int(color[index : index + 2], 16) for index in range(0, 6, 2))


def recolor_masked_region(
    image: np.ndarray,
    mask: np.ndarray,
    color_hex: str,
    preserve_luminance: bool = True,
) -> np.ndarray:
    rgb = np.asarray(hex_to_rgb(color_hex), dtype=np.float32)
    base = image.astype(np.float32)
    alpha = np.clip(mask.astype(np.float32), 0.0, 1.0)
    if alpha.ndim == 2:
        alpha = alpha[..., None]
    target = np.tile(rgb.reshape(1, 1, 3), (image.shape[0], image.shape[1], 1))
    if preserve_luminance:
        original_luma = np.dot(base, np.array([0.2126, 0.7152, 0.0722], dtype=np.float32))
        target_luma = float(
            np.dot(rgb, np.array([0.2126, 0.7152, 0.0722], dtype=np.float32))
        )
        scale = (original_luma / max(target_luma, 1.0))[..., None]
        target = np.clip(target * scale, 0.0, 255.0)
    output = base * (1.0 - alpha) + target * alpha
    return np.clip(output, 0.0, 255.0).astype(np.uint8)
