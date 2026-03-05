#!/usr/bin/env python3
"""Silence detection via ffmpeg + numpy. Replaces auto-editor's audio analysis."""

import subprocess
import sys

import numpy as np


def detect_silence(media_path, threshold=0.04, frame_rate=30, sample_rate=48000):
    """Return boolean array where True=loud, False=silent, one entry per video frame.

    Args:
        media_path: Path to audio or video file.
        threshold: Max absolute amplitude below which a frame is silent (0.0–1.0).
        frame_rate: Video frame rate (used to chunk audio into per-frame windows).
        sample_rate: Audio sample rate for extraction.

    Returns:
        numpy boolean array of length ceil(duration * frame_rate).
    """
    # Extract raw PCM audio via ffmpeg
    cmd = [
        "ffmpeg", "-i", str(media_path),
        "-vn",                          # no video
        "-ac", "1",                     # mono
        "-ar", str(sample_rate),        # resample
        "-f", "s16le",                  # raw 16-bit signed little-endian
        "-loglevel", "error",
        "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed: {result.stderr.decode()[:500]}")

    # Convert raw bytes to float32 samples in [-1, 1]
    samples = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0

    if len(samples) == 0:
        return np.array([], dtype=bool)

    # Chunk into per-frame windows
    samples_per_frame = int(sample_rate / frame_rate)
    n_frames = int(np.ceil(len(samples) / samples_per_frame))

    # Pad to fill last chunk
    padded = np.zeros(n_frames * samples_per_frame, dtype=np.float32)
    padded[:len(samples)] = samples

    chunks = padded.reshape(n_frames, samples_per_frame)
    max_amp = np.max(np.abs(chunks), axis=1)

    return max_amp >= threshold


def apply_margin(is_loud, margin_frames):
    """Expand loud regions by margin_frames in each direction (in-place).

    Args:
        is_loud: Boolean numpy array (modified in place).
        margin_frames: Number of frames to expand on each side.
    """
    if margin_frames <= 0 or len(is_loud) == 0:
        return

    n = len(is_loud)
    # Find transitions: False→True (start of loud) and True→False (end of loud)
    # Expand each loud region by margin_frames on both sides
    result = is_loud.copy()
    for i in range(n):
        if is_loud[i]:
            lo = max(0, i - margin_frames)
            hi = min(n, i + margin_frames + 1)
            result[lo:hi] = True
    is_loud[:] = result


def get_kept_ranges(is_loud, frame_rate):
    """Convert boolean array to list of (start_sec, end_sec) kept ranges.

    Args:
        is_loud: Boolean numpy array (True=keep, False=cut).
        frame_rate: Video frame rate.

    Returns:
        List of (start_sec, end_sec) tuples for contiguous loud regions.
    """
    if len(is_loud) == 0:
        return []

    ranges = []
    in_range = False
    start = 0

    for i, loud in enumerate(is_loud):
        if loud and not in_range:
            start = i
            in_range = True
        elif not loud and in_range:
            ranges.append((start / frame_rate, i / frame_rate))
            in_range = False

    if in_range:
        ranges.append((start / frame_rate, len(is_loud) / frame_rate))

    return ranges


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Detect silence in media files.")
    parser.add_argument("media", help="Path to audio/video file")
    parser.add_argument("--threshold", type=float, default=0.04, help="Amplitude threshold (0.0–1.0)")
    parser.add_argument("--frame-rate", type=float, default=30, help="Frame rate")
    parser.add_argument("--margin", type=float, default=0.1, help="Margin in seconds")
    args = parser.parse_args()

    is_loud = detect_silence(args.media, args.threshold, args.frame_rate)
    margin_frames = int(args.margin * args.frame_rate)
    apply_margin(is_loud, margin_frames)
    ranges = get_kept_ranges(is_loud, args.frame_rate)

    total_frames = len(is_loud)
    loud_frames = int(np.sum(is_loud))
    duration = total_frames / args.frame_rate

    print(f"Duration: {duration:.1f}s  |  {len(ranges)} kept ranges  |  "
          f"{loud_frames}/{total_frames} frames loud ({100*loud_frames/max(total_frames,1):.1f}%)")
    for s, e in ranges:
        print(f"  {s:.3f}s – {e:.3f}s  ({e-s:.3f}s)")
