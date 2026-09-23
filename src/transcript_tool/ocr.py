from __future__ import annotations

import difflib
from dataclasses import dataclass
from pathlib import Path

from .utils import ToolError, run_cmd


@dataclass
class ScreenTextEntry:
    timestamp: float
    text: str


def extract_frames(video_path: Path, out_dir: Path, interval_seconds: int) -> list[tuple[Path, float]]:
    """Sample satu frame setiap `interval_seconds` detik dari video (diproses lokal)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = out_dir / "frame_%06d.jpg"
    run_cmd(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            f"fps=1/{interval_seconds}",
            "-qscale:v",
            "2",
            str(pattern),
        ]
    )
    frames = sorted(out_dir.glob("frame_*.jpg"))
    return [(frame, idx * interval_seconds) for idx, frame in enumerate(frames)]


def _similar(a: str, b: str, threshold: float = 0.85) -> bool:
    if not a or not b:
        return a == b
    return difflib.SequenceMatcher(None, a, b).ratio() >= threshold


def ocr_frames(frames: list[tuple[Path, float]], lang: str = "ind+eng") -> list[ScreenTextEntry]:
    """OCR tiap frame lokal via Tesseract, lalu dedupe teks yang berturut-turut mirip (slide sama)."""
    try:
        import pytesseract
        from PIL import Image
    except ImportError as exc:
        raise ToolError(
            "Package 'pytesseract'/'Pillow' belum terinstall, atau binary tesseract-ocr belum ada di sistem. "
            "Lihat README bagian 'Setup OCR'."
        ) from exc

    entries: list[ScreenTextEntry] = []
    last_text = ""
    for frame_path, ts in frames:
        raw = pytesseract.image_to_string(Image.open(frame_path), lang=lang)
        text = " ".join(raw.split())
        if not text:
            continue
        if _similar(text, last_text):
            continue
        entries.append(ScreenTextEntry(timestamp=ts, text=text))
        last_text = text

    return entries
