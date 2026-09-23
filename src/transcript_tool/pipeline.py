from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .audio import extract_audio
from .diarize import assign_speakers, run_diarization
from .merge import build_full_transcript
from .mom import generate_mom
from .ocr import extract_frames, ocr_frames
from .transcribe import transcribe_audio, transcribe_audio_local
from .utils import ToolError

ALLOWED_VIDEO_EXT = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}


@dataclass
class PipelineOptions:
    video_path: Path
    output_dir: Path
    language: str | None = "id"
    engine: str = "openai"  # "openai" (butuh OPENAI_API_KEY) atau "local" (faster-whisper, gratis)
    local_model_size: str = "small"
    do_diarization: bool = True
    do_ocr: bool = True
    do_mom: bool = True
    ocr_interval: int = 5
    keep_temp: bool = False


@dataclass
class PipelineResult:
    transcript_path: Path
    mom_path: Path | None
    warnings: list[str]


def _validate_video(video_path: Path) -> None:
    if not video_path.exists() or not video_path.is_file():
        raise ToolError(f"File video tidak ditemukan: {video_path}")
    if video_path.suffix.lower() not in ALLOWED_VIDEO_EXT:
        raise ToolError(
            f"Ekstensi '{video_path.suffix}' tidak didukung. Gunakan salah satu dari: "
            f"{', '.join(sorted(ALLOWED_VIDEO_EXT))}"
        )


def run_pipeline(opts: PipelineOptions) -> PipelineResult:
    _validate_video(opts.video_path)
    opts.output_dir.mkdir(parents=True, exist_ok=True)

    work_dir = Path(tempfile.mkdtemp(prefix="transcript_tool_"))
    warnings: list[str] = []

    try:
        print("[1/5] Extract audio dari video (lokal, via ffmpeg)...")
        wav_path = extract_audio(opts.video_path, work_dir / "audio.wav")

        if opts.engine == "local":
            print("[2/5] Transkripsi audio lokal (faster-whisper, tanpa API key)...")
            transcript_segments = transcribe_audio_local(wav_path, opts.language, opts.local_model_size)
        else:
            print("[2/5] Transkripsi audio via OpenAI Whisper API...")
            transcript_segments = transcribe_audio(wav_path, work_dir, opts.language)
        print(f"  Selesai: {len(transcript_segments)} segmen transcript.")

        if opts.do_diarization:
            print("[3/5] Speaker diarization (lokal, pyannote.audio)...")
            try:
                speaker_segments = run_diarization(wav_path)
                transcript_segments = assign_speakers(transcript_segments, speaker_segments)
                print(f"  Selesai: {len(speaker_segments)} segmen speaker terdeteksi.")
            except ToolError as exc:
                msg = f"Diarization dilewati: {exc}"
                print(f"  [WARNING] {msg}")
                warnings.append(msg)
        else:
            print("[3/5] Speaker diarization dilewati (sesuai opsi).")

        screen_entries = []
        if opts.do_ocr:
            print("[4/5] Extract teks layar via OCR (lokal, tesseract)...")
            try:
                frames = extract_frames(opts.video_path, work_dir / "frames", opts.ocr_interval)
                screen_entries = ocr_frames(frames)
                print(f"  Selesai: {len(screen_entries)} potongan teks layar unik terdeteksi.")
            except ToolError as exc:
                msg = f"OCR layar dilewati: {exc}"
                print(f"  [WARNING] {msg}")
                warnings.append(msg)
        else:
            print("[4/5] OCR layar dilewati (sesuai opsi).")

        full_transcript = build_full_transcript(transcript_segments, screen_entries, opts.video_path.name)
        transcript_path = opts.output_dir / "transcript.md"
        transcript_path.write_text(full_transcript, encoding="utf-8")
        print(f"  Transcript tersimpan: {transcript_path}")

        mom_path = None
        if opts.do_mom:
            print("[5/5] Generate Minutes of Meeting via Claude API...")
            try:
                mom_text = generate_mom(full_transcript)
                mom_path = opts.output_dir / "mom.md"
                mom_path.write_text(mom_text, encoding="utf-8")
                print(f"  MoM tersimpan: {mom_path}")
            except ToolError as exc:
                msg = f"Generate MoM dilewati: {exc}"
                print(f"  [WARNING] {msg}")
                warnings.append(msg)
        else:
            print("[5/5] Generate MoM dilewati (sesuai opsi).")

        return PipelineResult(transcript_path=transcript_path, mom_path=mom_path, warnings=warnings)
    finally:
        if not opts.keep_temp:
            shutil.rmtree(work_dir, ignore_errors=True)
        else:
            print(f"File sementara disimpan di: {work_dir}")
