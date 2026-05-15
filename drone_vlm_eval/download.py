"""Dataset download and extraction helpers."""

from __future__ import annotations

import shutil
import urllib.request
import zipfile
from pathlib import Path

TEST_SNIPPET_URL = "https://drive.usercontent.google.com/u/0/uc?id=1gw08wFoN8Aop6BQBbYuCPGHiI8vQmTgI&export=download"
TRAIN_SNIPPET_URL = "https://drive.usercontent.google.com/u/0/uc?id=1rRr821XZ6qvedc1x2ZFeZivEzESLLAOm&export=download"


def download_file(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as fh:  # noqa: S310
        shutil.copyfileobj(response, fh)
    return destination


def extract_if_zip(archive: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    if not zipfile.is_zipfile(archive):
        return archive
    marker = destination / ".extracted"
    if marker.exists():
        return destination
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(destination)
    marker.write_text(archive.name)
    return destination


def download_snippets(data_dir: Path) -> dict[str, Path]:
    downloads = data_dir / "downloads"
    raw = data_dir / "raw"
    test_archive = download_file(TEST_SNIPPET_URL, downloads / "drone_test_snippet.zip")
    train_archive = download_file(TRAIN_SNIPPET_URL, downloads / "drone_train_snippet.zip")
    return {
        "test": extract_if_zip(test_archive, raw / "test_snippet"),
        "train": extract_if_zip(train_archive, raw / "train_snippet"),
    }



