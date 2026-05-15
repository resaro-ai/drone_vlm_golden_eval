"""Small image helpers that avoid mandatory Pillow/OpenCV dependencies."""

from __future__ import annotations

import struct
from pathlib import Path


_SOF_MARKERS = frozenset([
    *range(0xC0, 0xC4), *range(0xC5, 0xC8), *range(0xC9, 0xCC), *range(0xCD, 0xD0),
])


def jpeg_size(path: Path) -> tuple[int, int]:
    """Read JPEG dimensions from headers."""
    with path.open("rb") as f:
        if f.read(2) != b"\xff\xd8":
            raise ValueError(f"{path} is not a JPEG")
        while True:
            marker_start = f.read(1)
            if not marker_start:
                raise ValueError(f"No JPEG size marker found for {path}")
            if marker_start != b"\xff":
                continue
            marker = f.read(1)
            while marker == b"\xff":
                marker = f.read(1)
            if marker in {b"\xd8", b"\xd9"}:
                continue
            length_bytes = f.read(2)
            if len(length_bytes) != 2:
                raise ValueError(f"Bad JPEG marker length for {path}")
            length = struct.unpack(">H", length_bytes)[0]
            if marker[0] in _SOF_MARKERS:
                data = f.read(length - 2)
                if len(data) < 5:
                    raise ValueError(f"Bad JPEG SOF data for {path}")
                height, width = struct.unpack(">HH", data[1:5])
                return width, height
            f.seek(length - 2, 1)


def try_open_with_pillow(path: Path):
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise RuntimeError("Pillow is required for this operation. Install `Pillow`.") from exc
    return Image.open(path).convert("RGB")


def brightness_score(path: Path) -> float | None:
    """Return mean luminance in [0, 255] when Pillow is available."""
    try:
        image = try_open_with_pillow(path)
    except RuntimeError:
        return None
    gray = image.convert("L")
    pixels = list(gray.getdata())
    return sum(pixels) / max(len(pixels), 1)


def blur_score(path: Path) -> float | None:
    """Return a lightweight Laplacian-variance blur proxy when Pillow is available."""
    try:
        image = try_open_with_pillow(path).convert("L")
    except RuntimeError:
        return None
    width, height = image.size
    if width < 3 or height < 3:
        return 0.0
    pixels = image.load()
    values: list[float] = []
    for y in range(1, height - 1, max(1, height // 160)):
        for x in range(1, width - 1, max(1, width // 160)):
            center = float(pixels[x, y])
            lap = (
                float(pixels[x - 1, y])
                + float(pixels[x + 1, y])
                + float(pixels[x, y - 1])
                + float(pixels[x, y + 1])
                - 4.0 * center
            )
            values.append(lap)
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)


def image_to_base64(path: Path) -> str:
    """Read a JPEG file and return its base64-encoded string (no data URI prefix)."""
    import base64
    return base64.b64encode(path.read_bytes()).decode("ascii")

