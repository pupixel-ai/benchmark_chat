from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .models import LoadedImage


def load_image(path: Path) -> LoadedImage:
    cv2 = _try_import_cv2()
    if cv2 is not None:
        pixels = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if pixels is not None:
            rgb = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)
            return LoadedImage(pixels=rgb, width=int(rgb.shape[1]), height=int(rgb.shape[0]))

    image = _load_with_pillow(path)
    if image is None:
        raise ValueError("unable to decode image")
    return image


def resize_image(image: LoadedImage, max_side: int) -> LoadedImage:
    width = image.width
    height = image.height
    longest_side = max(width, height)
    if longest_side <= max_side:
        return image

    scale = float(max_side) / float(longest_side)
    target_width = max(1, int(round(width * scale)))
    target_height = max(1, int(round(height * scale)))
    pixels = image.pixels

    cv2 = _try_import_cv2()
    if cv2 is not None:
        resized = cv2.resize(pixels, (target_width, target_height), interpolation=cv2.INTER_AREA)
        return LoadedImage(pixels=resized, width=target_width, height=target_height)

    pillow_image = _pillow_from_array(pixels)
    if pillow_image is None:
        raise RuntimeError("resizing requires OpenCV or Pillow")
    resized = pillow_image.resize((target_width, target_height))
    return LoadedImage(pixels=_array_from_pillow(resized), width=target_width, height=target_height)


def _try_import_cv2() -> Any:
    try:
        import cv2  # type: ignore
    except ImportError:
        return None
    return cv2


def _load_with_pillow(path: Path) -> Optional[LoadedImage]:
    try:
        from PIL import Image  # type: ignore
    except ImportError:
        return None

    with Image.open(path) as raw:
        image = raw.convert("RGB")
        width, height = image.size
        return LoadedImage(pixels=_array_from_pillow(image), width=width, height=height)


def _pillow_from_array(pixels: object) -> Any:
    try:
        from PIL import Image  # type: ignore
    except ImportError:
        return None
    return Image.fromarray(pixels)


def _array_from_pillow(image: Any) -> object:
    try:
        import numpy as np  # type: ignore
    except ImportError as exc:
        raise RuntimeError("numpy is required for Pillow-based image decoding") from exc
    return np.asarray(image)
