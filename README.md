# Export Transcript From Video

Tool CLI untuk mengubah video rapat (termasuk yang ada screen-share, misalnya rekaman Zoom/Teams) jadi:

1. **Transcript lengkap** dengan timestamp, label pembicara ("Speaker 1", "Speaker 2", dst), dan teks yang
   tampil di layar saat screen-share (misalnya isi dokumen/slide yang di-share).
2. **Draft Minutes of Meeting (MoM)** terstruktur: ringkasan, peserta, pembahasan per topik, keputusan, dan
   action items — dibuat otomatis oleh Claude berdasarkan transcript.

Setiap tahap dirancang untuk **gagal secara aman (graceful degradation)**: kalau salah satu komponen (mis.
diarization atau OCR) tidak bisa jalan — dependency belum diinstall, API key belum diisi, dsb — tool tetap
melanjutkan proses dan minimal tetap menghasilkan transcript teks. Bukan gagal total.

## Alur data & keamanan

Penting untuk video rapat internal/rahasia — berikut kemana saja data kamu mengalir:

| Data | Diproses di mana | Dikirim ke mana |
|---|---|---|
| File video mentah | Lokal (mesin kamu) | **Tidak pernah diupload ke mana pun** |
| Audio hasil extract | Lokal | Dikirim ke **OpenAI** (Whisper API) untuk transkripsi |
| Speaker diarization (siapa bicara) | 100% **lokal** (`pyannote.audio`, model didownload sekali) | Tidak dikirim kemana pun |
| Frame video / OCR teks layar | 100% **lokal** (`tesseract`) | Tidak dikirim kemana pun |
| Transcript teks (audio + layar) | Lokal | Dikirim ke **Anthropic** (Claude API) untuk generate MoM |

File `.env` (API key) dan folder `output/` sudah masuk `.gitignore` secara default supaya rekaman/transkrip
rapat tidak sengaja ter-commit ke git. Pastikan kamu memang punya izin untuk merekam & mentranskrip rapat
yang kamu proses.

## Instalasi

### 1. System dependencies

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg tesseract-ocr tesseract-ocr-ind
```

`tesseract-ocr-ind` menambah dukungan bahasa Indonesia untuk OCR teks di layar (dokumen/slide).

### 2. Python dependencies

Untuk coba dulu versi paling ringan (transcript + MoM saja, tanpa speaker ID / OCR layar):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-core.txt
```

Lalu jalankan dengan flag `--no-diarization --no-ocr`.

Kalau nanti mau tambah fitur speaker ID dan/atau OCR layar, install tambahan sesuai kebutuhan:

```bash
pip install -r requirements-diarization.txt   # speaker ID (berat: torch + pyannote, ~1-2GB)
pip install -r requirements-ocr.txt           # OCR teks layar
# atau langsung semuanya:
pip install -r requirements.txt
```

### 3. Setup API key

```bash
cp .env.example .env
```

Isi `.env`:

- `OPENAI_API_KEY` — untuk transkripsi (Whisper API). Dapatkan di https://platform.openai.com/api-keys
- `ANTHROPIC_API_KEY` — untuk generate MoM (Claude API). Dapatkan di https://console.anthropic.com/settings/keys
- `HF_TOKEN` — untuk speaker diarization lokal. Dapatkan token gratis di
  https://huggingface.co/settings/tokens, lalu **accept license** (login dulu) di dua halaman model berikut
  dengan akun yang sama:
  - https://huggingface.co/pyannote/speaker-diarization-3.1
  - https://huggingface.co/pyannote/segmentation-3.0

Kalau salah satu key tidak diisi, tool tetap jalan tapi bagian yang butuh key tersebut otomatis dilewati
dengan pesan warning (bukan error fatal).

## Cara pakai

```bash
python main.py --video rapat.mp4 --output-dir hasil/
```

Hasil:

- `hasil/transcript.md` — transcript lengkap (audio + speaker label + teks layar), selalu dibuat.
- `hasil/mom.md` — draft Minutes of Meeting, dibuat kalau `ANTHROPIC_API_KEY` tersedia.

### Opsi lain

```bash
# Hanya extract transcript teks, tanpa speaker ID, OCR, atau MoM (paling cepat & minim dependency)
python main.py --video rapat.mp4 --no-diarization --no-ocr --no-mom

# Video berbahasa Inggris / campuran, biarkan Whisper auto-detect bahasa
python main.py --video rapat.mp4 --language ""

# OCR sampling tiap 10 detik (default 5 detik) - lebih cepat untuk video panjang
python main.py --video rapat.mp4 --ocr-interval 10

# Simpan file sementara (audio wav, frame) untuk debugging
python main.py --video rapat.mp4 --keep-temp
```

Jalankan `python main.py --help` untuk daftar lengkap opsi.

## Format output

`transcript.md` berupa timeline kronologis, contoh:

```
[00:01:05] Speaker 1: Baik, kita mulai rapat evaluasi penanganan kebocoran pipa.
[00:01:40] [LAYAR] PT Pertamina Hulu Mahakam wajib menyelesaikan penanggulangan kedaruratan...
[00:02:10] Speaker 2: Untuk section 1 sampai 3 sudah dipasang oil boom tambahan.
```

`mom.md` berisi bagian: Ringkasan Umum, Peserta yang Teridentifikasi, Pembahasan per Topik (dengan
referensi timestamp), Dokumen/Slide yang Dibahas, Keputusan, Action Items (tabel), dan Catatan Tambahan.
Claude diinstruksikan untuk **tidak mengarang** nama/tanggal/keputusan yang tidak ada di transcript.

## Batasan yang perlu diketahui

- **Nama peserta**: diarization hanya memberi label "Speaker 1", "Speaker 2", dst — bukan nama asli, kecuali
  nama disebutkan langsung dalam percakapan (mis. seseorang dipanggil "Pak Budi").
- **Akurasi speaker ID** tergantung kualitas audio (satu mic per orang lebih akurat daripada semua orang
  pakai satu mic laptop).
- **Video panjang**: audio otomatis dipecah jadi chunk ~10 menit untuk memenuhi batas ukuran file Whisper
  API (25MB); ada kemungkinan kata di ujung batas chunk sedikit terpotong.
- **OCR** sebaiknya dipakai untuk video dengan screen-share teks/dokumen yang cukup jelas terbaca; hasil
  OCR bisa kurang akurat untuk teks kecil/blur atau font tidak umum.

## Menjalankan test

```bash
pip install pytest
python -m pytest tests/ -v
```

Test yang ada memverifikasi logic murni (penggabungan timeline transcript+OCR, dedupe teks layar) tanpa
butuh ffmpeg/API key, jadi bisa dijalankan di mana saja.
