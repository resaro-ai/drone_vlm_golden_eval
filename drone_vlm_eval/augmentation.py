"""Targeted augmentation adapters."""

from __future__ import annotations

import base64
import hashlib
import io
import os
import shutil
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd


class Augmenter(ABC):
    name: str

    @abstractmethod
    def augment(self, df: pd.DataFrame, output_dir: Path, max_rows: int = 10) -> pd.DataFrame:
        """Return candidate augmented rows."""


class BlurAugmenter(Augmenter):
    name = "gaussian_blur"

    def augment(self, df: pd.DataFrame, output_dir: Path, max_rows: int = 10) -> pd.DataFrame:
        output_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for _, row in df.head(max_rows).iterrows():
            src = Path(str(row["image_path"]))
            out = output_dir / f"{row['image_id']}__blur.jpg"
            try:
                from PIL import Image, ImageFilter

                Image.open(src).convert("RGB").filter(
                    ImageFilter.GaussianBlur(radius=2.0)
                ).save(out, quality=92)
            except ModuleNotFoundError:
                shutil.copyfile(src, out)
            new_row = row.copy()
            new_row["image_id"] = f"{row['image_id']}__blur"
            new_row["image_path"] = str(out)
            new_row["augmentation_type"] = self.name
            new_row["source_image_id"] = row["image_id"]
            new_row["is_synthetic"] = True
            rows.append(new_row)
        return pd.DataFrame(rows)


class OpenAIImageAugmenter(Augmenter):
    """Generic image augmentation via OpenAI gpt-image-2 image editing.

    Pass a dict of {label: prompt} to define augmentation variants. Each
    label becomes a suffix on the generated image_id and is stored in the
    ``augment_label`` column for provenance.

    Bounding boxes are inherited unchanged from the source row. Generated
    rows are candidates; they must still pass quality checks.

    Environment variables:
        OPENAI_API_KEY: API key (required)
        OPENAI_IMAGE_CACHE_DIR: Local cache directory for generated images
    """

    name = "openai_image"

    def __init__(
        self,
        prompts: dict[str, str],
        api_key: str | None = None,
        cache_dir: Path | None = None,
        model: str = "gpt-image-2",
        max_workers: int = 4,
    ) -> None:
        self.prompts = prompts
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        cache_env = os.environ.get("OPENAI_IMAGE_CACHE_DIR")
        self.cache_dir = cache_dir or (Path(cache_env) if cache_env else None)
        self.model = model
        self.max_workers = max_workers
        self._client: Any = None

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        try:
            from openai import OpenAI
        except ModuleNotFoundError as exc:
            raise RuntimeError("OpenAIImageAugmenter requires the `openai` package.") from exc
        self._client = OpenAI(api_key=self.api_key)

    def _cache_key(self, image_b64: str, prompt: str) -> str:
        return hashlib.sha256(f"{image_b64}|{prompt}".encode()).hexdigest()[:16]

    def _edit_image(self, src: Path, prompt: str) -> bytes:
        """Call the OpenAI images.edit API and return raw image bytes."""
        self._ensure_client()

        image_bytes = src.read_bytes()
        image_b64 = base64.b64encode(image_bytes).decode("ascii")

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = self.cache_dir / f"{self._cache_key(image_b64, prompt)}.jpg"
            if cache_path.exists():
                return cache_path.read_bytes()

        with open(src, "rb") as image_file:
            response = self._client.images.edit(
                model=self.model,
                image=image_file,
                prompt=prompt,
                quality="low",
                n=1,
            )
        result_bytes = base64.b64decode(response.data[0].b64_json)

        if self.cache_dir:
            cache_path.write_bytes(result_bytes)

        return result_bytes

    def augment(self, df: pd.DataFrame, output_dir: Path, max_rows: int = 10) -> pd.DataFrame:
        """Augment source rows with every prompt.

        Args:
            max_rows: Number of source rows to process (0 = all). Each row
                receives every prompt, so total outputs = max_rows × len(prompts).
        """
        from PIL import Image

        output_dir.mkdir(parents=True, exist_ok=True)

        source = df if max_rows == 0 else df.head(max_rows)

        tasks = [
            (label, prompt, row)
            for label, prompt in self.prompts.items()
            for _, row in source.iterrows()
        ]

        def _process(task: tuple[str, str, Any]) -> Any:
            label, prompt, row = task
            src = Path(str(row["image_path"]))
            image_id = str(row["image_id"])
            out = output_dir / f"{image_id}__{label}.jpg"

            status = "ok"
            try:
                result_bytes = self._edit_image(src, prompt)
                Image.open(io.BytesIO(result_bytes)).convert("RGB").save(out, quality=92)
            except Exception as exc:
                print(f"  OpenAI image '{label}' failed for {image_id}: {exc}")
                status = f"failed: {exc}"

            new_row = row.copy()
            new_row["image_id"] = f"{image_id}__openai_{label}"
            new_row["source_image_id"] = image_id
            if out.exists():
                new_row["image_path"] = str(out)
                new_row["augmentation_type"] = self.name
                new_row["augment_label"] = label
                new_row["is_synthetic"] = True
            else:
                new_row["image_path"] = None
                new_row["augmentation_type"] = f"{self.name}_failed"
                new_row["augment_label"] = label
                new_row["is_synthetic"] = False
            new_row["openai_augment_status"] = status
            return new_row

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            rows = list(executor.map(_process, tasks))

        for label in self.prompts:
            label_rows = [r for r in rows if r.get("augment_label") == label]
            ok_count = sum(1 for r in label_rows if r.get("openai_augment_status") == "ok")
            print(f"  OpenAI image '{label}': {ok_count}/{len(label_rows)} succeeded")

        ok_total = sum(1 for r in rows if r.get("openai_augment_status") == "ok")
        print(f"  OpenAI image total: {ok_total}/{len(rows)} augmented candidates succeeded")
        return pd.DataFrame(rows)
