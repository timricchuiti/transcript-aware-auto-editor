#!/usr/bin/env python3
"""Diff engine — compares original vs edited SRT and extracts cut ranges from WhisperX JSON."""

import json
import re
import sys
from collections import namedtuple

SrtBlock = namedtuple("SrtBlock", ["index", "start", "end", "text"])


def parse_srt(filepath):
    """Parse an SRT file into a list of SrtBlocks.

    Handles messy edits: missing blank lines, non-sequential indices,
    extra whitespace, missing block numbers.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    blocks = []
    # Split on blank lines (one or more)
    raw_blocks = re.split(r"\n\s*\n", content.strip())

    for raw in raw_blocks:
        lines = [line.strip() for line in raw.strip().split("\n") if line.strip()]
        if not lines:
            continue

        # Find the timestamp line (contains " --> ")
        ts_idx = None
        for i, line in enumerate(lines):
            if " --> " in line:
                ts_idx = i
                break

        if ts_idx is None:
            # No timestamp found — skip this block
            print(f"Warning: Skipping block with no timestamp: {lines[:2]}", file=sys.stderr)
            continue

        # Parse index (line before timestamp, if present and numeric)
        index = None
        if ts_idx > 0:
            try:
                index = int(lines[ts_idx - 1])
            except ValueError:
                pass

        # Parse timestamps
        ts_match = re.match(
            r"(\d{1,2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{1,2}:\d{2}:\d{2}[,\.]\d{3})",
            lines[ts_idx],
        )
        if not ts_match:
            print(f"Warning: Skipping block with malformed timestamp: {lines[ts_idx]}", file=sys.stderr)
            continue

        start = _srt_time_to_seconds(ts_match.group(1))
        end = _srt_time_to_seconds(ts_match.group(2))

        if start is None or end is None:
            print(f"Warning: Skipping block with unparseable timestamp: {lines[ts_idx]}", file=sys.stderr)
            continue

        # Text is everything after the timestamp line
        text_lines = lines[ts_idx + 1 :]
        text = " ".join(text_lines)

        blocks.append(SrtBlock(index=index, start=start, end=end, text=text))

    return blocks


def _srt_time_to_seconds(ts):
    """Convert SRT timestamp (HH:MM:SS,mmm or HH:MM:SS.mmm) to seconds."""
    ts = ts.replace(",", ".")
    match = re.match(r"(\d+):(\d{2}):(\d{2})\.(\d{3})", ts)
    if not match:
        return None
    h, m, s, ms = int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))
    return h * 3600 + m * 60 + s + ms / 1000.0


def _normalize_text(text):
    """Normalize text for comparison: lowercase, collapse whitespace, strip punctuation."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_whisper_json(filepath):
    """Load WhisperX JSON output."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def _find_segment_times(whisper_data, block_text, srt_start=None, claimed_segments=None):
    """Find the best matching segment in WhisperX JSON for a given SRT block text.

    When multiple segments match the same text (duplicates), uses proximity to
    the SRT block's timestamp to pick the closest unclaimed segment.

    Args:
        whisper_data: Parsed WhisperX JSON dict.
        block_text: Text content of the SRT block.
        srt_start: Start time from the SRT block (used for proximity matching).
        claimed_segments: Set of already-claimed segment indices (mutated in place).

    Returns (start, end) in seconds, or None if no match found.
    """
    norm_block = _normalize_text(block_text)
    if not norm_block:
        return None

    if claimed_segments is None:
        claimed_segments = set()

    segments = whisper_data.get("segments", [])
    candidates = []  # (seg_index, start, end, match_quality, time_distance)

    for i, seg in enumerate(segments):
        if i in claimed_segments:
            continue

        seg_text = seg.get("text", "")
        norm_seg = _normalize_text(seg_text)
        if not norm_seg:
            continue

        # Exact match
        if norm_seg == norm_block:
            dist = abs(seg["start"] - srt_start) if srt_start is not None else 0
            candidates.append((i, seg["start"], seg["end"], 1.0, dist))
            continue

        # Containment/overlap match
        if norm_block in norm_seg or norm_seg in norm_block:
            overlap = len(set(norm_block.split()) & set(norm_seg.split()))
            total = max(len(norm_block.split()), len(norm_seg.split()))
            score = overlap / total if total > 0 else 0
            if score >= 0.5:
                dist = abs(seg["start"] - srt_start) if srt_start is not None else 0
                candidates.append((i, seg["start"], seg["end"], score, dist))

    if not candidates:
        return None

    # Sort by: best match quality (descending), then closest to SRT timestamp (ascending)
    candidates.sort(key=lambda c: (-c[3], c[4]))
    best = candidates[0]
    claimed_segments.add(best[0])
    return (best[1], best[2])


def find_deleted_ranges(original_srt, edited_srt, whisper_json):
    """Find time ranges that were deleted from the SRT.

    Compares original SRT against edited SRT by text content, counting duplicates.
    If the same text appears N times in the original and M times in the edited
    (where M < N), then N - M instances are considered deleted.

    Timestamps come from the WhisperX JSON (source of truth).

    Args:
        original_srt: Path to the original .srt.orig file.
        edited_srt: Path to the user-edited .srt file.
        whisper_json: Path to the WhisperX .json file.

    Returns:
        List of (start_sec, end_sec) tuples, sorted and non-overlapping.
    """
    original_blocks = parse_srt(original_srt)
    edited_blocks = parse_srt(edited_srt)
    whisper_data = load_whisper_json(whisper_json)

    # Count occurrences of each normalized text in the edited SRT
    from collections import Counter
    edited_counts = Counter(_normalize_text(b.text) for b in edited_blocks)

    # Walk through original blocks in order. For each normalized text,
    # "spend" edited counts first (those are kept), then the rest are deletions.
    remaining_edited = dict(edited_counts)
    deleted_blocks = []
    for block in original_blocks:
        norm = _normalize_text(block.text)
        if not norm:
            continue
        if remaining_edited.get(norm, 0) > 0:
            # This block is still present in edited — consume one count
            remaining_edited[norm] -= 1
        else:
            # This block was deleted
            deleted_blocks.append(block)

    if not deleted_blocks:
        print("No deleted blocks found — nothing to cut.")
        return []

    # Look up timestamps from JSON for each deleted block.
    # Track which JSON segments have been claimed so duplicate text
    # maps to distinct segments by proximity to the SRT block's timestamp.
    claimed_segments = set()  # indices into whisper_data["segments"]
    ranges = []
    for block in deleted_blocks:
        times = _find_segment_times(whisper_data, block.text, block.start, claimed_segments)
        if times:
            ranges.append(times)
        else:
            # Fall back to SRT timestamps with a warning
            print(
                f"Warning: Could not match block to JSON, using SRT timestamps: "
                f'"{block.text[:60]}..."',
                file=sys.stderr,
            )
            ranges.append((block.start, block.end))

    # Sort and merge overlapping ranges
    ranges.sort()
    merged = _merge_ranges(ranges)

    return merged


def _merge_ranges(ranges):
    """Merge overlapping or adjacent time ranges."""
    if not ranges:
        return []

    merged = [ranges[0]]
    for start, end in ranges[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    return merged


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Diff SRT files and find deleted time ranges.")
    parser.add_argument("original", help="Path to original .srt.orig file")
    parser.add_argument("edited", help="Path to edited .srt file")
    parser.add_argument("json", help="Path to WhisperX .json file")
    args = parser.parse_args()

    ranges = find_deleted_ranges(args.original, args.edited, args.json)
    if ranges:
        print(f"\nFound {len(ranges)} cut range(s):")
        for start, end in ranges:
            print(f"  {start:.3f}s — {end:.3f}s  ({end - start:.3f}s)")
