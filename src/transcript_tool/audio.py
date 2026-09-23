from __future__ import annotations

from pathlib import Path

from .utils import ffprobe_duration, run_cmd


def extract_audio(video_path: Path, out_path: Path) -> Path:
    """Extract audio dari video jadi mono 16kHz WAV (untuk diarization) menggunakan ffmpeg."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    run_cmd(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(out_path),
        ]
    )
    return out_path


def split_audio_for_upload(wav_path: Path, out_dir: Path, chunk_seconds: int) -> list[tuple[Path, float]]:
    """Split audio jadi beberapa chunk mp3 (untuk memenuhi batas ukuran file Whisper API).

    Return list of (chunk_path, start_offset_seconds) terurut berdasarkan waktu.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = out_dir / "chunk_%03d.mp3"
    run_cmd(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(wav_path),
            "-f",
            "segment",
            "-segment_time",
            str(chunk_seconds),
            "-reset_timestamps",
            "1",
            "-c:a",
            "libmp3lame",
            "-q:a",
            "4",
            str(pattern),
        ]
    )
    chunks = sorted(out_dir.glob("chunk_*.mp3"))
    if not chunks:
        raise RuntimeError("Gagal membagi audio menjadi beberapa chunk.")

    result = []
    offset = 0.0
    for chunk in chunks:
        result.append((chunk, offset))
        offset += ffprobe_duration(chunk)
    return result


def audio_needs_split(wav_path: Path, max_bytes: int) -> bool:
    return wav_path.stat().st_size > max_bytes
