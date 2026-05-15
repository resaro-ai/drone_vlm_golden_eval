"""Load and normalize the drone image/XML snippets."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

from drone_vlm_eval.image_io import jpeg_size
from drone_vlm_eval.schemas import BoundingBox, DroneRecord


def parse_voc_xml(path: Path) -> tuple[int, int, list[BoundingBox]]:
    root = ET.parse(path).getroot()
    width = int(float(root.findtext("size/width", "0")))
    height = int(float(root.findtext("size/height", "0")))
    bbox: list[BoundingBox] = []
    for obj in root.findall("object"):
        b = obj.find("bndbox")
        if b is None:
            continue
        bbox.append(
            BoundingBox(
                xmin=float(b.findtext("xmin", "0")),
                ymin=float(b.findtext("ymin", "0")),
                xmax=float(b.findtext("xmax", "0")),
                ymax=float(b.findtext("ymax", "0")),
            )
        )
    return width, height, bbox


def find_image_dirs(root: Path) -> list[Path]:
    return sorted(
        p
        for p in root.rglob("*")
        if p.is_dir() and any(child.suffix.lower() in {".jpg", ".jpeg", ".png"} for child in p.iterdir())
    )


def find_xml_dirs(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_dir() and any(child.suffix.lower() == ".xml" for child in p.iterdir()))


def load_snippet(root: Path, source_split: str) -> list[DroneRecord]:
    image_dirs = find_image_dirs(root)
    xml_dirs = find_xml_dirs(root)
    xml_by_stem = {xml.stem: xml for xml_dir in xml_dirs for xml in xml_dir.glob("*.xml")}
    records: list[DroneRecord] = []
    for image_dir in image_dirs:
        for image_path in sorted(image_dir.iterdir()):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            image_id = image_path.stem
            xml_path = xml_by_stem.get(image_id)
            if xml_path is not None:
                xml_width, xml_height, bbox = parse_voc_xml(xml_path)
                width, height = xml_width, xml_height
            else:
                width, height = jpeg_size(image_path)
                bbox = []
            records.append(
                DroneRecord(
                    image_id=image_id,
                    image_path=image_path,
                    width=width,
                    height=height,
                    drone_present=bool(bbox) or image_id.startswith("VS_P"),
                    bbox=bbox,
                    source_split=source_split,
                    xml_path=xml_path,
                )
            )
    return records


def load_snippets(paths: dict[str, Path]) -> pd.DataFrame:
    records: list[DroneRecord] = []
    for split, path in paths.items():
        if path.exists():
            records.extend(load_snippet(path, split))
    return pd.DataFrame([record.to_dict() for record in records])


def discover_existing_snippets(data_dir: Path) -> dict[str, Path]:
    """Return local snippets, preferring checked-in data then Downloads fallback."""
    raw = data_dir / "raw"
    paths: dict[str, Path] = {}
    for split in ("test", "train"):
        candidates = [
            raw / f"{split}_snippet",
            raw / f"drone_{split}_snippet",
        ]
        for candidate in candidates:
            if candidate.exists():
                paths[split] = candidate
                break
    downloads_test = Path.home() / "Downloads" / "DroneDatasetTestSnippet"
    if "test" not in paths and downloads_test.exists():
        paths["test"] = downloads_test
    return paths

