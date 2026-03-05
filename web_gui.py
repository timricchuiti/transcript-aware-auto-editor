#!/usr/bin/env python3
"""Flask web GUI for PaperCut."""

import argparse
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_file, send_from_directory

from transcript_diff import find_deleted_ranges, parse_srt, load_whisper_json
from silence import detect_silence, apply_margin, get_kept_ranges
from timeline_export import (
    Clip, build_clip_list, get_media_info,
    generate_fcpxml, generate_premiere_xml, export_video as export_video_file,
)

app = Flask(__name__, static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024 * 1024  # 10 GB

UPLOAD_DIR = Path(tempfile.gettempdir()) / "papercut"
UPLOAD_DIR.mkdir(exist_ok=True)


@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({"error": "File too large (max 10 GB)"}), 413


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/landing")
def landing():
    return send_from_directory(app.static_folder, "landing.html")


@app.route("/api/upload", methods=["POST"])
def upload_file():
    """Handle video file upload via drag-and-drop."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400

    dest = UPLOAD_DIR / f.filename
    f.save(str(dest))

    return jsonify({
        "path": str(dest),
        "filename": f.filename,
    })


@app.route("/media")
def serve_media():
    """Serve a local video file for HTML5 playback with range request support."""
    path = request.args.get("path", "")
    p = Path(path).resolve()
    if not p.exists() or not p.is_file():
        return jsonify({"error": f"File not found: {path}"}), 404
    return send_file(str(p), conditional=True)


@app.route("/api/transcribe", methods=["POST"])
def transcribe_sse():
    """Start transcription and stream progress via SSE."""
    data = request.get_json()
    video_path = data.get("video_path", "")
    model = data.get("model", "medium")
    language = data.get("language", "en")
    engine = data.get("engine", "whisperx")

    video = Path(video_path).resolve()
    if not video.exists():
        return jsonify({"error": f"File not found: {video_path}"}), 404

    out_dir = video.parent
    stem = video.stem

    if engine == "crisperwhisper":
        return _transcribe_crisper_sse(video, out_dir, stem, language)

    # Default: WhisperX via subprocess
    cmd = [
        "whisperx",
        str(video),
        "--model", model,
        "--language", language,
        "--output_format", "all",
        "--compute_type", "float32",
        "--output_dir", str(out_dir),
    ]

    def generate():
        yield f"data: {json.dumps({'type': 'start', 'message': 'Starting WhisperX...', 'command': ' '.join(cmd)})}\n\n"

        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1
        )

        for line in iter(proc.stdout.readline, ""):
            line = line.rstrip()
            if line:
                yield f"data: {json.dumps({'type': 'progress', 'message': line})}\n\n"

        proc.wait()

        if proc.returncode != 0:
            yield f"data: {json.dumps({'type': 'error', 'message': f'WhisperX failed (exit code {proc.returncode})'})}\n\n"
            return

        json_path = out_dir / f"{stem}.json"
        srt_path = out_dir / f"{stem}.srt"
        orig_srt_path = out_dir / f"{stem}.srt.orig"

        # Copy original SRT for diffing
        if srt_path.exists() and not orig_srt_path.exists():
            import shutil
            shutil.copy2(srt_path, orig_srt_path)

        result = {
            "type": "done",
            "message": "Transcription complete.",
            "json_path": str(json_path),
            "srt_path": str(srt_path),
            "orig_srt_path": str(orig_srt_path),
        }
        yield f"data: {json.dumps(result)}\n\n"

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _transcribe_crisper_sse(video, out_dir, stem, language):
    """Run CrisperWhisper transcription with SSE progress streaming."""
    from auto_transcript import transcribe_crisper
    import queue
    import threading

    progress_queue = queue.Queue()

    def run_transcription():
        try:
            result = transcribe_crisper(
                str(video), language=language, output_dir=str(out_dir),
                progress_callback=lambda msg: progress_queue.put(("progress", msg)),
            )
            progress_queue.put(("done", result))
        except Exception as e:
            progress_queue.put(("error", str(e)))

    def generate():
        yield f"data: {json.dumps({'type': 'start', 'message': 'Starting CrisperWhisper (verbatim mode)...'})}\n\n"

        thread = threading.Thread(target=run_transcription, daemon=True)
        thread.start()

        while True:
            try:
                kind, payload = progress_queue.get(timeout=0.5)
            except queue.Empty:
                if not thread.is_alive():
                    break
                continue

            if kind == "progress":
                yield f"data: {json.dumps({'type': 'progress', 'message': payload})}\n\n"
            elif kind == "error":
                yield f"data: {json.dumps({'type': 'error', 'message': payload})}\n\n"
                return
            elif kind == "done":
                json_path, srt_path, orig_srt_path = payload
                result = {
                    "type": "done",
                    "message": "Transcription complete (CrisperWhisper).",
                    "json_path": str(json_path),
                    "srt_path": str(srt_path),
                    "orig_srt_path": str(orig_srt_path),
                }
                yield f"data: {json.dumps(result)}\n\n"
                return

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/srt")
def get_srt():
    """Parse an SRT file and return blocks as JSON."""
    path = request.args.get("path", "")
    p = Path(path).resolve()
    if not p.exists():
        return jsonify({"error": f"File not found: {path}"}), 404

    blocks = parse_srt(str(p))
    return jsonify([
        {"index": b.index, "start": b.start, "end": b.end, "text": b.text}
        for b in blocks
    ])


@app.route("/api/diff", methods=["POST"])
def diff_transcript():
    """Accept edited block list, compute cut ranges via WhisperX JSON."""
    data = request.get_json()
    orig_srt_path = data.get("orig_srt_path", "")
    json_path = data.get("json_path", "")
    kept_blocks = data.get("kept_blocks", [])  # list of {index, start, end, text}

    for path, label in [(orig_srt_path, "Original SRT"), (json_path, "WhisperX JSON")]:
        if not Path(path).resolve().exists():
            return jsonify({"error": f"{label} not found: {path}"}), 404

    # Write kept blocks to a temp SRT file
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".srt", delete=False, encoding="utf-8")
    try:
        for i, block in enumerate(kept_blocks, 1):
            start_ts = _seconds_to_srt_time(block["start"])
            end_ts = _seconds_to_srt_time(block["end"])
            tmp.write(f"{i}\n{start_ts} --> {end_ts}\n{block['text']}\n\n")
        tmp.close()

        deleted_ranges = find_deleted_ranges(orig_srt_path, tmp.name, json_path)

        # Compute summary
        original_blocks = parse_srt(orig_srt_path)
        total_blocks = len(original_blocks)
        kept_count = len(kept_blocks)
        removed_count = total_blocks - kept_count
        cut_duration = sum(end - start for start, end in deleted_ranges)

        # Get video duration from JSON if available
        whisper_data = load_whisper_json(json_path)
        segments = whisper_data.get("segments", [])
        total_duration = segments[-1]["end"] if segments else 0

        return jsonify({
            "deleted_ranges": [{"start": s, "end": e} for s, e in deleted_ranges],
            "summary": {
                "total_blocks": total_blocks,
                "kept_blocks": kept_count,
                "removed_blocks": removed_count,
                "cut_duration": round(cut_duration, 3),
                "total_duration": round(total_duration, 3),
                "reduction_pct": round((cut_duration / total_duration) * 100, 1) if total_duration > 0 else 0,
            }
        })
    finally:
        os.unlink(tmp.name)


@app.route("/api/export", methods=["POST"])
def export_video():
    """Build clip list from ordered blocks + silence detection, then export."""
    data = request.get_json()
    video_path = data.get("video_path", "")
    ordered_blocks = data.get("ordered_blocks", [])  # [{id, start, end}, ...]
    margin = data.get("margin", 0.1)
    export_format = data.get("export", "final-cut-pro")
    ffmpeg_args = data.get("ffmpeg_args", "")
    edit_method = data.get("edit_method", "")
    export_folder = data.get("export_folder", "")

    video = Path(video_path).resolve()
    if not video.exists():
        return jsonify({"error": f"Video not found: {video_path}"}), 404

    if not ordered_blocks:
        return jsonify({"success": False, "error": "No blocks to export."}), 400

    try:
        # Probe media
        media_info = get_media_info(str(video))
        frame_rate = media_info["frame_rate"]

        # Parse threshold from edit_method (e.g. "audio:threshold=0.04")
        threshold = 0.04
        if edit_method:
            import re
            m = re.search(r"threshold=([0-9.]+)", edit_method)
            if m:
                threshold = float(m.group(1))

        # Silence detection
        is_loud = detect_silence(str(video), threshold=threshold, frame_rate=frame_rate,
                                 sample_rate=media_info["sample_rate"])
        margin_frames = int(margin * frame_rate)
        apply_margin(is_loud, margin_frames)
        kept_ranges = get_kept_ranges(is_loud, frame_rate)

        # Build clip list from user's ordered blocks + silence data
        clips = build_clip_list(ordered_blocks, kept_ranges, margin=margin)

        if not clips:
            return jsonify({"success": False, "error": "No audio detected in kept blocks."})

        # Determine output path
        ext_map = {
            "final-cut-pro": ".fcpxml",
            "premiere": ".xml",
            "resolve": ".fcpxml",
            "video": video.suffix,
        }
        ext = ext_map.get(export_format, video.suffix)
        output_name = video.stem + "_ALTERED" + ext

        if export_folder:
            export_dir = Path(export_folder).resolve()
            export_dir.mkdir(parents=True, exist_ok=True)
        else:
            export_dir = video.parent

        output_path = export_dir / output_name

        # Generate export
        if export_format in ("final-cut-pro", "resolve"):
            xml = generate_fcpxml(str(video), clips, media_info)
            output_path.write_text(xml, encoding="utf-8")
        elif export_format == "premiere":
            xml = generate_premiere_xml(str(video), clips, media_info)
            output_path.write_text(xml, encoding="utf-8")
        elif export_format == "video":
            extra = ffmpeg_args.split() if ffmpeg_args else None
            export_video_file(str(video), clips, str(output_path), extra_args=extra)
        else:
            return jsonify({"success": False, "error": f"Unknown export format: {export_format}"})

        clip_count = len(clips)
        total_dur = sum(c.duration for c in clips)
        return jsonify({
            "success": True,
            "message": f"Export completed: {output_path.name} ({clip_count} clips, {total_dur:.1f}s)",
            "output_path": str(output_path),
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


def _seconds_to_srt_time(seconds):
    """Convert seconds to SRT timestamp format HH:MM:SS,mmm."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    whole_s = int(s)
    ms = int(round((s - whole_s) * 1000))
    return f"{h:02d}:{m:02d}:{whole_s:02d},{ms:03d}"


def main():
    parser = argparse.ArgumentParser(description="Web GUI for PaperCut")
    parser.add_argument("--port", type=int, default=5000, help="Port to run on (default: 5000)")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    args = parser.parse_args()

    print(f"Starting PaperCut at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
