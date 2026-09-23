from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import config
from .audio import audio_needs_split, split_audio_for_upload
from .utils import ToolError, retry


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str
    speaker: str | None = None


def transcribe_audio(wav_path: Path, work_dir: Path, language: str | None) -> list[TranscriptSegment]:
    """Transkripsi audio via OpenAI Whisper API. Audio otomatis dipecah jika > 24MB."""
    if not config.OPENAI_API_KEY:
        raise ToolError(
            "OPENAI_API_KEY belum diset. Isi file .env (lihat .env.example) sebelum menjalankan transkripsi."
        )

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ToolError("Package 'openai' belum terinstall. Jalankan: pip install -r requirements.txt") from exc

    client = OpenAI(api_key=config.OPENAI_API_KEY)

    if audio_needs_split(wav_path, config.WHISPER_MAX_BYTES):
        chunk_dir = work_dir / "audio_chunks"
        # ~10 menit per chunk cukup aman di bawah batas 25MB untuk mp3 quality 4
        chunks = split_audio_for_upload(wav_path, chunk_dir, chunk_seconds=600)
    else:
        chunks = [(wav_path, 0.0)]

    all_segments: list[TranscriptSegment] = []
    for idx, (chunk_path, offset) in enumerate(chunks, start=1):
        print(f"  Transkripsi chunk {idx}/{len(chunks)} ({chunk_path.name})...")

        def call() -> object:
            with open(chunk_path, "rb") as f:
                kwargs = dict(
                    model=config.WHISPER_MODEL,
                    file=f,
                    response_format="verbose_json",
                )
                if language:
                    kwargs["language"] = language
                return client.audio.transcriptions.create(**kwargs)

        response = retry(call, label=f"Whisper transcription chunk {idx}")

        segments = getattr(response, "segments", None)
        if segments:
            for seg in segments:
                all_segments.append(
                    TranscriptSegment(
                        start=seg["start"] + offset if isinstance(seg, dict) else seg.start + offset,
                        end=seg["end"] + offset if isinstance(seg, dict) else seg.end + offset,
                        text=(seg["text"] if isinstance(seg, dict) else seg.text).strip(),
                    )
                )
        else:
            text = getattr(response, "text", "").strip()
            if text:
                all_segments.append(TranscriptSegment(start=offset, end=offset, text=text))

    return all_segments
