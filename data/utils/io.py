"""Generic file I/O helpers shared across `data/`: HTTP download and
JSON-lines read/write.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, TypeVar

import requests

from data.utils.constants import HTTP_DOWNLOAD_CHUNK_BYTES, HTTP_DOWNLOAD_TIMEOUT_SECONDS

T = TypeVar("T")


def download_file(url: str, destination: Path, overwrite: bool = False) -> Path:
    if destination.exists() and not overwrite:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=HTTP_DOWNLOAD_TIMEOUT_SECONDS) as response:
        response.raise_for_status()
        tmp_destination = destination.with_suffix(destination.suffix + ".part")
        with open(tmp_destination, "wb") as f:
            for chunk in response.iter_content(chunk_size=HTTP_DOWNLOAD_CHUNK_BYTES):
                f.write(chunk)
        tmp_destination.replace(destination)
    return destination


def write_jsonl(items: Iterable, destination: Path, to_json_line: Callable[[object], str]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "w", encoding="utf-8") as f:
        for item in items:
            f.write(to_json_line(item))
            f.write("\n")


def read_jsonl(source: Path, from_json_line: Callable[[str], T]) -> list[T]:
    with open(source, encoding="utf-8") as f:
        return [from_json_line(line) for line in f if line.strip()]
