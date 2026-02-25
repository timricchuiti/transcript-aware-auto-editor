#!/usr/bin/env python3
"""CLI orchestrator for the transcript-aware auto-editor pipeline."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from auto_transcript import transcribe
from transcript_diff import find_deleted_ranges, parse_srt
from merge_cutlists import build_auto_editor_cmd, run_auto_editor


def get_video_duration(video_path):
    """Get video duration in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        str(video_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            data = json.loads(result.stdout)
            return float(data["format"]["duration"])
    except (json.JSONDecodeError, KeyError, ValueError):
        pass
    return None


def print_summary(video_path, original_srt, deleted_ranges):
    """Print editing statistics."""
    print("\n--- Edit Summary ---")

    duration = get_video_duration(video_path)
    if duration:
        print(f"  Input duration:        {_format_duration(duration)}")

    original_blocks = parse_srt(original_srt)
    total_blocks = len(original_blocks)

    cut_duration = sum(end - start for start, end in deleted_ranges)
    num_cuts = len(deleted_ranges)

    # Count deleted blocks (blocks in original not accounted for by remaining)
    # We approximate: number of cuts ~ number of deleted blocks
    print(f"  Transcript blocks:     {total_blocks}")
    print(f"  Blocks removed:        {num_cuts}")
    print(f"  Time removed:          {_format_duration(cut_duration)}")

    if duration:
        remaining = duration - cut_duration
        pct = (cut_duration / duration) * 100 if duration > 0 else 0
        print(f"  Estimated output:      {_format_duration(remaining)}")
        print(f"  Reduction:             {pct:.1f}%")

    print("--------------------\n")


def _format_duration(seconds):
    """Format seconds into MM:SS.s or HH:MM:SS.s."""
    if seconds < 0:
        seconds = 0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if h > 0:
        return f"{h}:{m:02d}:{s:05.2f}"
    return f"{m}:{s:05.2f}"


def open_in_editor(filepath):
    """Open a file in the user's default editor."""
    editor = os.environ.get("EDITOR", os.environ.get("VISUAL"))

    if editor:
        subprocess.run([editor, str(filepath)])
    elif sys.platform == "darwin":
        subprocess.run(["open", "-t", str(filepath)])
    elif sys.platform.startswith("linux"):
        subprocess.run(["xdg-open", str(filepath)])
    else:
        print(f"Please open {filepath} in your text editor.", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Transcript-aware video editing via auto-editor.",
        epilog="Example: python3 main.py video.mp4 --transcript video.srt --whisper-json video.json --export final-cut-pro",
    )
    parser.add_argument("video", help="Path to the input video file")

    # Transcript inputs
    parser.add_argument("--transcript", help="Path to the edited .srt file")
    parser.add_argument("--whisper-json", help="Path to the WhisperX .json file")
    parser.add_argument("--original-srt", help="Path to the original .srt.orig file (default: <transcript>.orig)")

    # Workflow flags
    parser.add_argument("--transcribe-only", action="store_true",
                        help="Generate transcript and stop")
    parser.add_argument("--edit-transcript", action="store_true",
                        help="Open SRT in default text editor")
    parser.add_argument("--summary", action="store_true",
                        help="Print edit statistics")

    # auto-editor options
    parser.add_argument("--margin", default=None,
                        help="Margin around cuts (passed directly to auto-editor, e.g., 0.15sec)")
    parser.add_argument("--export", default=None,
                        help="Export format: final-cut-pro, premiere, resolve, clip-sequence, or video")
    parser.add_argument("--ffmpeg-args", default=None,
                        help='FFmpeg args for re-encoding (e.g., "-crf 22 -preset veryfast")')

    # Transcription options (for --transcribe-only)
    parser.add_argument("--model", default="medium", help="WhisperX model size (default: medium)")
    parser.add_argument("--language", default="en", help="Language code (default: en)")
    parser.add_argument("--output-dir", default=None, help="Output directory for transcription")

    args = parser.parse_args()

    video = Path(args.video).resolve()
    if not video.exists():
        print(f"Error: Video file not found: {video}", file=sys.stderr)
        sys.exit(1)

    # Phase 1: Transcribe only
    if args.transcribe_only:
        transcribe(str(video), model=args.model, language=args.language,
                   output_dir=args.output_dir)
        return

    # Determine SRT paths
    if args.transcript:
        edited_srt = Path(args.transcript).resolve()
    else:
        edited_srt = video.with_suffix(".srt")

    if args.original_srt:
        original_srt = Path(args.original_srt).resolve()
    else:
        original_srt = Path(str(edited_srt) + ".orig")

    if args.whisper_json:
        whisper_json = Path(args.whisper_json).resolve()
    else:
        whisper_json = video.with_suffix(".json")

    # Open editor if requested
    if args.edit_transcript:
        if not edited_srt.exists():
            print(f"Error: SRT file not found: {edited_srt}", file=sys.stderr)
            print("Run with --transcribe-only first to generate the transcript.")
            sys.exit(1)
        open_in_editor(edited_srt)
        print(f"Editor closed. Re-run without --edit-transcript to apply cuts.")
        return

    # Validate required files
    for path, label in [(edited_srt, "Edited SRT"), (original_srt, "Original SRT"),
                        (whisper_json, "WhisperX JSON")]:
        if not path.exists():
            print(f"Error: {label} not found: {path}", file=sys.stderr)
            if label == "Original SRT":
                print("Hint: Run auto_transcript.py first, or provide --original-srt.", file=sys.stderr)
            sys.exit(1)

    # Phase 2: Diff transcripts
    print(f"Comparing transcripts...")
    print(f"  Original: {original_srt}")
    print(f"  Edited:   {edited_srt}")

    deleted_ranges = find_deleted_ranges(str(original_srt), str(edited_srt), str(whisper_json))

    if not deleted_ranges:
        print("No transcript edits detected. Running auto-editor with default settings only.")

    if deleted_ranges:
        print(f"\nFound {len(deleted_ranges)} transcript cut(s):")
        for start, end in deleted_ranges:
            print(f"  {start:.3f}s — {end:.3f}s  ({end - start:.3f}s)")

    # Summary
    if args.summary:
        print_summary(str(video), str(original_srt), deleted_ranges)

    # Phase 3: Build and run auto-editor
    extra_args = []
    if args.ffmpeg_args:
        extra_args.extend(args.ffmpeg_args.split())

    cmd = build_auto_editor_cmd(
        video_path=str(video),
        transcript_cuts=deleted_ranges if deleted_ranges else None,
        margin=args.margin,
        export=args.export,
        extra_args=extra_args if extra_args else None,
    )

    run_auto_editor(cmd)


if __name__ == "__main__":
    main()
