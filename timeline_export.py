#!/usr/bin/env python3
"""Timeline export — FCPXML, Premiere XML, and ffmpeg video export.

Replaces auto-editor's export functionality with support for reordered clips.
"""

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape


@dataclass
class Clip:
    """A single clip referencing a region of the source media."""
    source_in: float   # seconds — start in source
    source_out: float  # seconds — end in source

    @property
    def duration(self):
        return self.source_out - self.source_in


def get_media_info(path):
    """Probe media file with ffprobe and return metadata dict.

    Returns:
        {
            "duration": float,
            "frame_rate": float,
            "frame_rate_num": int,
            "frame_rate_den": int,
            "width": int | None,
            "height": int | None,
            "sample_rate": int,
            "has_video": bool,
            "has_audio": bool,
        }
    """
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-show_entries", "stream=codec_type,width,height,r_frame_rate,sample_rate",
        "-print_format", "json",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr[:500]}")

    data = json.loads(result.stdout)

    info = {
        "duration": float(data.get("format", {}).get("duration", 0)),
        "frame_rate": 30.0,
        "frame_rate_num": 30,
        "frame_rate_den": 1,
        "width": None,
        "height": None,
        "sample_rate": 48000,
        "has_video": False,
        "has_audio": False,
    }

    for stream in data.get("streams", []):
        codec_type = stream.get("codec_type")
        if codec_type == "video":
            info["has_video"] = True
            if stream.get("width"):
                info["width"] = int(stream["width"])
            if stream.get("height"):
                info["height"] = int(stream["height"])
            rfr = stream.get("r_frame_rate", "30/1")
            if "/" in rfr:
                num, den = rfr.split("/")
                num, den = int(num), int(den)
                if den > 0:
                    info["frame_rate"] = num / den
                    info["frame_rate_num"] = num
                    info["frame_rate_den"] = den
        elif codec_type == "audio":
            info["has_audio"] = True
            if stream.get("sample_rate"):
                info["sample_rate"] = int(stream["sample_rate"])

    return info


def build_clip_list(ordered_blocks, silence_kept_ranges, margin=0.0):
    """Build final clip list from user's ordered blocks + silence-detected kept ranges.

    For each kept block (in user's order), find silence-detected kept ranges that
    overlap with the block's time span, clip them to block boundaries, and apply margin.

    Args:
        ordered_blocks: List of dicts with 'start' and 'end' (seconds), in user's desired order.
        silence_kept_ranges: List of (start_sec, end_sec) from silence detection.
        margin: Extra padding in seconds to add around each clip boundary.

    Returns:
        List of Clip objects in playback order.
    """
    clips = []

    for block in ordered_blocks:
        block_start = block["start"]
        block_end = block["end"]

        # Find silence-kept ranges overlapping this block
        block_clips = []
        for rng_start, rng_end in silence_kept_ranges:
            # Check overlap
            overlap_start = max(rng_start, block_start)
            overlap_end = min(rng_end, block_end)
            if overlap_start < overlap_end:
                # Apply margin (expand clip, but clamp to block boundaries)
                clip_in = max(block_start, overlap_start - margin)
                clip_out = min(block_end, overlap_end + margin)
                block_clips.append(Clip(source_in=clip_in, source_out=clip_out))

        if block_clips:
            # Merge overlapping clips within this block
            merged = [block_clips[0]]
            for c in block_clips[1:]:
                if c.source_in <= merged[-1].source_out:
                    merged[-1] = Clip(merged[-1].source_in, max(merged[-1].source_out, c.source_out))
                else:
                    merged.append(c)
            clips.extend(merged)
        else:
            # No silence data overlaps — keep entire block (minus margin clamp)
            clips.append(Clip(source_in=block_start, source_out=block_end))

    return clips


def _seconds_to_rational(seconds, frame_rate_num, frame_rate_den):
    """Convert seconds to FCPXML rational time string (e.g. '3003/30000s')."""
    # Use frame_rate_num as timebase denominator for frame-accurate times
    # FCPXML uses rational time: numerator/denominator format
    timebase = frame_rate_num * 100  # high-precision timebase
    ticks = round(seconds * timebase / frame_rate_den)
    return f"{ticks}/{timebase}s"


def generate_fcpxml(media_path, clips, media_info):
    """Generate FCPXML 1.11 string for Final Cut Pro / DaVinci Resolve.

    Args:
        media_path: Path to the source media file.
        clips: List of Clip objects.
        media_info: Dict from get_media_info().

    Returns:
        FCPXML string.
    """
    p = Path(media_path)
    filename = p.name
    fr_num = media_info["frame_rate_num"]
    fr_den = media_info["frame_rate_den"]
    duration = media_info["duration"]
    has_video = media_info["has_video"]
    has_audio = media_info["has_audio"]

    # Calculate total timeline duration
    timeline_dur = sum(c.duration for c in clips)

    # Format spec
    if has_video:
        w = media_info["width"] or 1920
        h = media_info["height"] or 1080
        format_el = f'    <format id="r1" name="FFVideoFormat{h}p{round(fr_num/fr_den)}" frameDuration="{fr_den}/{fr_num}s" width="{w}" height="{h}"/>'
    else:
        format_el = f'    <format id="r1" name="FFVideoFormatRateUndefined" frameDuration="{fr_den}/{fr_num}s"/>'

    # Asset
    src_str = _seconds_to_rational(duration, fr_num, fr_den)
    if has_video and has_audio:
        asset_el = f'    <asset id="r2" name="{xml_escape(p.stem)}" src="file://{xml_escape(str(p.resolve()))}" start="0/1s" duration="{src_str}" hasVideo="1" hasAudio="1" format="r1" audioSources="1" audioChannels="2"/>'
    elif has_audio:
        asset_el = f'    <asset id="r2" name="{xml_escape(p.stem)}" src="file://{xml_escape(str(p.resolve()))}" start="0/1s" duration="{src_str}" hasAudio="1" format="r1" audioSources="1" audioChannels="2"/>'
    else:
        asset_el = f'    <asset id="r2" name="{xml_escape(p.stem)}" src="file://{xml_escape(str(p.resolve()))}" start="0/1s" duration="{src_str}" hasVideo="1" format="r1"/>'

    # Build spine clips
    spine_items = []
    timeline_pos = 0.0
    for clip in clips:
        offset = _seconds_to_rational(timeline_pos, fr_num, fr_den)
        start = _seconds_to_rational(clip.source_in, fr_num, fr_den)
        dur = _seconds_to_rational(clip.duration, fr_num, fr_den)
        spine_items.append(
            f'          <asset-clip ref="r2" offset="{offset}" name="{xml_escape(p.stem)}" start="{start}" duration="{dur}"/>'
        )
        timeline_pos += clip.duration

    tl_dur = _seconds_to_rational(timeline_dur, fr_num, fr_den)
    spine_xml = "\n".join(spine_items)

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE fcpxml>
<fcpxml version="1.11">
  <resources>
{format_el}
{asset_el}
  </resources>
  <library>
    <event name="PaperCut Import">
      <project name="{xml_escape(p.stem)}_ALTERED">
        <sequence format="r1" duration="{tl_dur}" tcStart="0/1s" tcFormat="NDF">
          <spine>
{spine_xml}
          </spine>
        </sequence>
      </project>
    </event>
  </library>
</fcpxml>
"""


def generate_premiere_xml(media_path, clips, media_info):
    """Generate FCP7 XML (Premiere Pro compatible) string.

    Args:
        media_path: Path to the source media file.
        clips: List of Clip objects.
        media_info: Dict from get_media_info().

    Returns:
        FCP7 XML string.
    """
    p = Path(media_path)
    filename = p.name
    fr = media_info["frame_rate"]
    duration_frames = round(media_info["duration"] * fr)
    has_video = media_info["has_video"]
    has_audio = media_info["has_audio"]
    w = media_info.get("width") or 1920
    h = media_info.get("height") or 1080
    timebase = round(fr)

    # Build clip items
    video_clips = []
    audio_clips = []
    timeline_frame = 0
    for i, clip in enumerate(clips, 1):
        in_frame = round(clip.source_in * fr)
        out_frame = round(clip.source_out * fr)
        clip_dur = out_frame - in_frame
        start_frame = timeline_frame
        end_frame = timeline_frame + clip_dur

        clip_xml = f"""          <clipitem id="clipitem-{i}">
            <name>{xml_escape(filename)}</name>
            <duration>{duration_frames}</duration>
            <rate><timebase>{timebase}</timebase><ntsc>FALSE</ntsc></rate>
            <in>{in_frame}</in>
            <out>{out_frame}</out>
            <start>{start_frame}</start>
            <end>{end_frame}</end>
            <file id="file-1"/>
          </clipitem>"""

        if has_video:
            video_clips.append(clip_xml)
        if has_audio:
            audio_clips.append(clip_xml.replace("clipitem", "clipitem").replace(
                f'id="clipitem-{i}"', f'id="clipitem-audio-{i}"'
            ))

        timeline_frame = end_frame

    total_tl_frames = timeline_frame

    video_track = ""
    if has_video and video_clips:
        video_track = f"""      <track>
{chr(10).join(video_clips)}
      </track>"""

    audio_track = ""
    if has_audio and audio_clips:
        audio_track = f"""      <track>
{chr(10).join(audio_clips)}
      </track>"""

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE xmeml>
<xmeml version="5">
  <sequence>
    <name>{xml_escape(p.stem)}_ALTERED</name>
    <duration>{total_tl_frames}</duration>
    <rate><timebase>{timebase}</timebase><ntsc>FALSE</ntsc></rate>
    <media>
      <video>
{video_track}
      </video>
      <audio>
{audio_track}
      </audio>
    </media>
    <timecode>
      <string>00:00:00:00</string>
      <frame>0</frame>
      <rate><timebase>{timebase}</timebase><ntsc>FALSE</ntsc></rate>
    </timecode>
  </sequence>
  <bin>
    <children>
      <clip id="masterclip-1">
        <name>{xml_escape(filename)}</name>
        <duration>{duration_frames}</duration>
        <rate><timebase>{timebase}</timebase><ntsc>FALSE</ntsc></rate>
        <file id="file-1">
          <name>{xml_escape(filename)}</name>
          <pathurl>file://localhost{xml_escape(str(p.resolve()))}</pathurl>
          <duration>{duration_frames}</duration>
          <rate><timebase>{timebase}</timebase><ntsc>FALSE</ntsc></rate>
        </file>
      </clip>
    </children>
  </bin>
</xmeml>
"""


def export_video(media_path, clips, output_path, extra_args=None):
    """Export video/audio by concatenating clips via ffmpeg.

    Uses filter_complex with trim/atrim + concat for each clip.

    Args:
        media_path: Path to source media.
        clips: List of Clip objects.
        output_path: Path for output file.
        extra_args: Optional list of extra ffmpeg arguments.
    """
    if not clips:
        raise ValueError("No clips to export")

    p = Path(media_path)
    media_info = get_media_info(str(p))
    has_video = media_info["has_video"]
    has_audio = media_info["has_audio"]

    filter_parts = []
    concat_inputs = []

    for i, clip in enumerate(clips):
        if has_video and has_audio:
            filter_parts.append(
                f"[0:v]trim=start={clip.source_in:.6f}:end={clip.source_out:.6f},setpts=PTS-STARTPTS[v{i}];"
            )
            filter_parts.append(
                f"[0:a]atrim=start={clip.source_in:.6f}:end={clip.source_out:.6f},asetpts=PTS-STARTPTS[a{i}];"
            )
            concat_inputs.append(f"[v{i}][a{i}]")
        elif has_audio:
            filter_parts.append(
                f"[0:a]atrim=start={clip.source_in:.6f}:end={clip.source_out:.6f},asetpts=PTS-STARTPTS[a{i}];"
            )
            concat_inputs.append(f"[a{i}]")
        else:
            filter_parts.append(
                f"[0:v]trim=start={clip.source_in:.6f}:end={clip.source_out:.6f},setpts=PTS-STARTPTS[v{i}];"
            )
            concat_inputs.append(f"[v{i}]")

    n = len(clips)
    if has_video and has_audio:
        concat_filter = f"{''.join(concat_inputs)}concat=n={n}:v=1:a=1[outv][outa]"
        map_args = ["-map", "[outv]", "-map", "[outa]"]
    elif has_audio:
        concat_filter = f"{''.join(concat_inputs)}concat=n={n}:v=0:a=1[outa]"
        map_args = ["-map", "[outa]"]
    else:
        concat_filter = f"{''.join(concat_inputs)}concat=n={n}:v=1:a=0[outv]"
        map_args = ["-map", "[outv]"]

    filter_complex = "".join(filter_parts) + concat_filter

    cmd = [
        "ffmpeg", "-y",
        "-i", str(p),
        "-filter_complex", filter_complex,
    ] + map_args

    if extra_args:
        cmd.extend(extra_args)

    cmd.append(str(output_path))

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg export failed: {result.stderr[:1000]}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export timeline from clip list.")
    parser.add_argument("media", help="Source media file")
    parser.add_argument("--clips", required=True, help='JSON array of [{"in": 1.0, "out": 3.5}, ...]')
    parser.add_argument("--format", choices=["fcpxml", "premiere", "video"], default="fcpxml")
    parser.add_argument("--output", help="Output file path")
    args = parser.parse_args()

    clip_data = json.loads(args.clips)
    clips = [Clip(source_in=c["in"], source_out=c["out"]) for c in clip_data]
    info = get_media_info(args.media)

    if args.format == "fcpxml":
        print(generate_fcpxml(args.media, clips, info))
    elif args.format == "premiere":
        print(generate_premiere_xml(args.media, clips, info))
    elif args.format == "video":
        out = args.output or str(Path(args.media).stem) + "_ALTERED" + Path(args.media).suffix
        export_video(args.media, clips, out)
        print(f"Exported to {out}")
