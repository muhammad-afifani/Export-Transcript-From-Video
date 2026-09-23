import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transcript_tool.ocr import _similar  # noqa: E402


def test_similar_texts_detected():
    a = "PT Pertamina Hulu Mahakam wajib menyelesaikan penanggulangan kedaruratan kebocoran pipa"
    b = "PT Pertamina Hulu Mahakam wajib menyelesaikan penanggulangan kedaruratan kebocoran pipa minyak"
    assert _similar(a, b) is True


def test_different_texts_not_similar():
    a = "Ringkasan Insiden Kebocoran Pipa"
    b = "Daftar Hadir Peserta Rapat"
    assert _similar(a, b) is False


def test_empty_strings():
    assert _similar("", "") is True
    assert _similar("", "abc") is False
