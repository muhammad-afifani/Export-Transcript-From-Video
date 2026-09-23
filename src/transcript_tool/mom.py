from __future__ import annotations

from . import config
from .utils import ToolError, retry

SYSTEM_PROMPT = """\
Kamu adalah asisten yang membuat Minutes of Meeting (MoM) dari transcript rapat berbahasa Indonesia.
Transcript berisi baris berformat [HH:MM:SS] Speaker N: ucapan, dan baris [HH:MM:SS] [LAYAR]: teks yang \
tampil di layar saat screen-share (misalnya isi dokumen atau slide).

Susun MoM dalam Markdown dengan struktur berikut:

## Ringkasan Umum
Ringkasan singkat 3-5 kalimat tentang topik utama rapat.

## Peserta yang Teridentifikasi
Daftar "Speaker N" beserta nama asli jika disebutkan dalam percakapan (mis. seseorang menyapa "Pak Budi"). \
Jika nama tidak disebutkan, biarkan sebagai "Speaker N (nama tidak teridentifikasi dari audio)".

## Pembahasan per Topik
Kelompokkan diskusi menjadi beberapa topik/agenda. Untuk tiap topik, tulis ringkasan poin-poin penting dan \
sertakan referensi timestamp [HH:MM:SS] agar mudah ditelusuri kembali ke rekaman asli.

## Dokumen/Slide yang Dibahas
Jika ada baris [LAYAR], rangkum dokumen atau data yang ditampilkan dan kapan (timestamp) itu dibahas.

## Keputusan
Daftar keputusan yang diambil dalam rapat (jika ada).

## Action Items
Tabel Markdown dengan kolom: No | Tugas | PIC (jika disebutkan) | Deadline (jika disebutkan) | Catatan.
Jika PIC/deadline tidak disebutkan eksplisit di transcript, tulis "-" - jangan mengarang informasi.

## Catatan Tambahan
Hal lain yang relevan (risiko, pertanyaan terbuka, follow-up).

Aturan penting: hanya gunakan informasi yang benar-benar ada di transcript. Jangan mengarang nama orang, \
tanggal, angka, atau keputusan yang tidak disebutkan. Jika suatu bagian tidak ada isinya, tulis "Tidak ada \
informasi terkait di transcript."
"""


def generate_mom(transcript_text: str) -> str:
    if not config.ANTHROPIC_API_KEY:
        raise ToolError(
            "ANTHROPIC_API_KEY belum diset. Isi file .env (lihat .env.example) sebelum generate MoM."
        )

    try:
        from anthropic import Anthropic
    except ImportError as exc:
        raise ToolError("Package 'anthropic' belum terinstall. Jalankan: pip install -r requirements.txt") from exc

    client = Anthropic(api_key=config.ANTHROPIC_API_KEY)

    # Batas kasar untuk menghindari transcript yang jauh melebihi context window model.
    max_chars = 500_000
    truncated = transcript_text[:max_chars]
    note = ""
    if len(transcript_text) > max_chars:
        note = "\n\n> Catatan: transcript terlalu panjang, sebagian akhir dipotong untuk proses MoM.\n"

    def call() -> object:
        return client.messages.create(
            model=config.ANTHROPIC_MODEL,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": truncated}],
        )

    response = retry(call, label="Claude MoM generation")
    text = "".join(block.text for block in response.content if hasattr(block, "text"))
    return text + note
