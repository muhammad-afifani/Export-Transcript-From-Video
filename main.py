#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from transcript_tool.pipeline import PipelineOptions, run_pipeline  # noqa: E402
from transcript_tool.utils import ToolError  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract transcript + Minutes of Meeting (MoM) dari video rapat.",
    )
    parser.add_argument("--video", required=True, type=Path, help="Path ke file video (mp4/mkv/mov/...).")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output"), help="Folder output (default: ./output)."
    )
    parser.add_argument(
        "--language", default="id", help="Kode bahasa audio untuk Whisper (default: id). Kosongkan untuk auto-detect."
    )
    parser.add_argument(
        "--engine",
        choices=["openai", "local"],
        default="openai",
        help="Mesin transkripsi: 'openai' (Whisper API, butuh OPENAI_API_KEY) atau "
        "'local' (faster-whisper, gratis & tanpa API key, jalan di CPU). Default: openai.",
    )
    parser.add_argument(
        "--local-model-size",
        default="small",
        help="Ukuran model faster-whisper saat --engine local (tiny/base/small/medium/large-v3). "
        "Default: small (cukup akurat & masih cepat di CPU).",
    )
    parser.add_argument("--no-diarization", action="store_true", help="Lewati speaker diarization.")
    parser.add_argument("--no-ocr", action="store_true", help="Lewati OCR teks layar.")
    parser.add_argument("--no-mom", action="store_true", help="Lewati generate MoM, hanya buat transcript.")
    parser.add_argument(
        "--ocr-interval", type=int, default=5, help="Interval sampling frame untuk OCR, dalam detik (default: 5)."
    )
    parser.add_argument("--keep-temp", action="store_true", help="Simpan file sementara (audio/frame) untuk debug.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    opts = PipelineOptions(
        video_path=args.video,
        output_dir=args.output_dir,
        language=args.language or None,
        engine=args.engine,
        local_model_size=args.local_model_size,
        do_diarization=not args.no_diarization,
        do_ocr=not args.no_ocr,
        do_mom=not args.no_mom,
        ocr_interval=args.ocr_interval,
        keep_temp=args.keep_temp,
    )

    try:
        result = run_pipeline(opts)
    except ToolError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    print("\n=== Selesai ===")
    print(f"Transcript: {result.transcript_path}")
    print(f"MoM: {result.mom_path if result.mom_path else '(tidak dibuat)'}")
    if result.warnings:
        print("\nPeringatan selama proses:")
        for w in result.warnings:
            print(f"  - {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
