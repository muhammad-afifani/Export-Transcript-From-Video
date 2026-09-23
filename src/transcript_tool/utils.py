from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Callable, TypeVar

T = TypeVar("T")


class ToolError(RuntimeError):
    """Kesalahan yang terjadi di dalam pipeline transcript-tool."""


def run_cmd(args: list[str]) -> subprocess.CompletedProcess:
    """Jalankan perintah eksternal (ffmpeg/ffprobe) dengan argv list, tanpa shell."""
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        raise ToolError(
            f"Perintah gagal ({' '.join(args[:2])}...): {result.stderr.strip()[-2000:]}"
        )
    return result


def ffprobe_duration(path: Path) -> float:
    result = run_cmd(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ]
    )
    data = json.loads(result.stdout)
    return float(data["format"]["duration"])


def format_timestamp(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def retry(func: Callable[[], T], attempts: int = 4, base_delay: float = 2.0, label: str = "operasi") -> T:
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return func()
        except Exception as exc:  # noqa: BLE001 - retry generik untuk panggilan API eksternal
            last_exc = exc
            if attempt == attempts:
                break
            delay = base_delay * (2 ** (attempt - 1))
            print(f"  [retry] {label} gagal (percobaan {attempt}/{attempts}): {exc}. Coba lagi dalam {delay:.0f}s...")
            time.sleep(delay)
    raise ToolError(f"{label} tetap gagal setelah {attempts} percobaan: {last_exc}")
