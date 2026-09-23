from __future__ import annotations

from .ocr import ScreenTextEntry
from .transcribe import TranscriptSegment
from .utils import format_timestamp


def build_full_transcript(
    transcript_segments: list[TranscriptSegment],
    screen_entries: list[ScreenTextEntry],
    video_name: str,
) -> str:
    """Gabungkan transcript audio (dgn speaker label jika ada) + teks layar OCR jadi satu dokumen kronologis."""
    lines: list[str] = [f"# Transcript: {video_name}", ""]

    events: list[tuple[float, str]] = []
    for seg in transcript_segments:
        speaker_prefix = f"{seg.speaker}: " if seg.speaker else ""
        events.append((seg.start, f"[{format_timestamp(seg.start)}] {speaker_prefix}{seg.text}"))
    for entry in screen_entries:
        events.append((entry.timestamp, f"[{format_timestamp(entry.timestamp)}] [LAYAR] {entry.text}"))

    events.sort(key=lambda e: e[0])
    for _, line in events:
        lines.append(line)

    return "\n".join(lines) + "\n"
