from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import config
from .transcribe import TranscriptSegment
from .utils import ToolError


@dataclass
class SpeakerSegment:
    start: float
    end: float
    speaker: str


def run_diarization(wav_path: Path) -> list[SpeakerSegment]:
    """Jalankan speaker diarization 100% lokal via pyannote.audio. Audio tidak dikirim kemana pun."""
    if not config.HF_TOKEN:
        raise ToolError(
            "HF_TOKEN belum diset. Diarization butuh token Hugging Face gratis (untuk download model "
            "sekali saja) - lihat README bagian 'Setup speaker diarization'."
        )

    try:
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise ToolError(
            "Package 'pyannote.audio' (dan torch) belum terinstall. Jalankan: pip install -r requirements.txt"
        ) from exc

    pipeline = Pipeline.from_pretrained(config.DIARIZATION_MODEL, use_auth_token=config.HF_TOKEN)
    diarization = pipeline(str(wav_path))

    segments: list[SpeakerSegment] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append(SpeakerSegment(start=turn.start, end=turn.end, speaker=speaker))
    return segments


def assign_speakers(
    transcript_segments: list[TranscriptSegment], speaker_segments: list[SpeakerSegment]
) -> list[TranscriptSegment]:
    """Untuk tiap segmen transcript, cari speaker dengan overlap waktu terbesar."""
    if not speaker_segments:
        return transcript_segments

    speaker_names: dict[str, str] = {}

    def friendly_name(raw_label: str) -> str:
        if raw_label not in speaker_names:
            speaker_names[raw_label] = f"Speaker {len(speaker_names) + 1}"
        return speaker_names[raw_label]

    for seg in transcript_segments:
        best_overlap = 0.0
        best_speaker = None
        for sp in speaker_segments:
            overlap = min(seg.end, sp.end) - max(seg.start, sp.start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = sp.speaker
        if best_speaker is not None:
            seg.speaker = friendly_name(best_speaker)

    return transcript_segments
