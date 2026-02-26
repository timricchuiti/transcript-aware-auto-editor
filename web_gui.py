#!/usr/bin/env python3
"""Flask web GUI for transcript-aware auto-editor."""

import argparse
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_file, send_from_directory

from transcript_diff import find_deleted_ranges, parse_srt, load_whisper_json
from merge_cutlists import build_auto_editor_cmd, run_auto_editor

app = Flask(__name__, static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024 * 1024  # 10 GB

UPLOAD_DIR = Path(tempfile.gettempdir()) / "auto-editor-gui"
UPLOAD_DIR.mkdir(exist_ok=True)


@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({"error": "File too large (max 10 GB)"}), 413


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


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
    """Start WhisperX transcription and stream progress via SSE."""
    data = request.get_json()
    video_path = data.get("video_path", "")
    model = data.get("model", "medium")
    language = data.get("language", "en")

    video = Path(video_path).resolve()
    if not video.exists():
        return jsonify({"error": f"File not found: {video_path}"}), 404

    out_dir = video.parent
    stem = video.stem

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
    """Run auto-editor with transcript cuts and export settings."""
    data = request.get_json()
    video_path = data.get("video_path", "")
    deleted_ranges = data.get("deleted_ranges", [])  # [{start, end}, ...]
    margin = data.get("margin")
    export_format = data.get("export")
    silent_speed = data.get("silent_speed")
    sounded_speed = data.get("sounded_speed")
    video_codec = data.get("video_codec")
    audio_codec = data.get("audio_codec")
    ffmpeg_args = data.get("ffmpeg_args")
    edit_method = data.get("edit_method")

    video = Path(video_path).resolve()
    if not video.exists():
        return jsonify({"error": f"Video not found: {video_path}"}), 404

    transcript_cuts = [(r["start"], r["end"]) for r in deleted_ranges] if deleted_ranges else None

    extra_args = []
    if silent_speed is not None and silent_speed != "":
        extra_args.extend(["--video-speed", str(silent_speed)])
    if sounded_speed is not None and sounded_speed != "":
        extra_args.extend(["--video-speed", str(sounded_speed)])  # auto-editor uses different flags
    if video_codec:
        extra_args.extend(["--video-codec", video_codec])
    if audio_codec:
        extra_args.extend(["--audio-codec", audio_codec])
    if ffmpeg_args:
        extra_args.extend(ffmpeg_args.split())
    if edit_method:
        extra_args.extend(["--edit", edit_method])

    # Format margin for auto-editor (e.g., "0.15sec")
    margin_str = None
    if margin is not None:
        margin_str = f"{margin}sec"

    cmd = build_auto_editor_cmd(
        video_path=str(video),
        transcript_cuts=transcript_cuts,
        margin=margin_str,
        export=export_format if export_format else None,
        extra_args=extra_args if extra_args else None,
    )

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            return jsonify({
                "success": False,
                "command": " ".join(cmd),
                "error": result.stderr or "auto-editor failed",
            })
        return jsonify({
            "success": True,
            "command": " ".join(cmd),
            "stdout": result.stdout,
            "message": "Export completed successfully.",
        })
    except subprocess.TimeoutExpired:
        return jsonify({"success": False, "error": "Export timed out (10 min limit)."})


def _seconds_to_srt_time(seconds):
    """Convert seconds to SRT timestamp format HH:MM:SS,mmm."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    whole_s = int(s)
    ms = int(round((s - whole_s) * 1000))
    return f"{h:02d}:{m:02d}:{whole_s:02d},{ms:03d}"


def main():
    parser = argparse.ArgumentParser(description="Web GUI for transcript-aware auto-editor")
    parser.add_argument("--port", type=int, default=5000, help="Port to run on (default: 5000)")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    args = parser.parse_args()

    print(f"Starting transcript-aware auto-editor GUI at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
