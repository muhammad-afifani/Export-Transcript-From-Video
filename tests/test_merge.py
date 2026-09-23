import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transcript_tool.merge import build_full_transcript  # noqa: E402
from transcript_tool.ocr import ScreenTextEntry  # noqa: E402
from transcript_tool.transcribe import TranscriptSegment  # noqa: E402


def test_build_full_transcript_orders_chronologically_and_labels_speakers():
    segments = [
        TranscriptSegment(start=5.0, end=8.0, text="Selamat siang semua.", speaker="Speaker 1"),
        TranscriptSegment(start=12.0, end=15.0, text="Baik, saya lanjutkan.", speaker="Speaker 2"),
    ]
    screen = [ScreenTextEntry(timestamp=9.0, text="Slide 1: Ringkasan Insiden")]

    out = build_full_transcript(segments, screen, "rapat.mp4")

    lines = out.strip().splitlines()
    body = [line for line in lines if line.startswith("[")]
    assert body == [
        "[00:00:05] Speaker 1: Selamat siang semua.",
        "[00:00:09] [LAYAR] Slide 1: Ringkasan Insiden",
        "[00:00:12] Speaker 2: Baik, saya lanjutkan.",
    ]


def test_build_full_transcript_without_speaker_labels():
    segments = [TranscriptSegment(start=0.0, end=2.0, text="Halo.", speaker=None)]
    out = build_full_transcript(segments, [], "rapat.mp4")
    assert "[00:00:00] Halo." in out
