# CLAUDE.md — PaperCut

## Project Overview

A browser-based transcript editing tool for video and audio. Users transcribe media via WhisperX (or CrisperWhisper), cut/reorder/edit blocks in a two-pane UI, and export as FCPXML (Final Cut Pro / DaVinci Resolve), Premiere XML, or re-encoded video/audio via ffmpeg.

**Primary interface:** `python3 web_gui.py` → browser GUI at localhost:5000

## Architecture

```
Media file → WhisperX/CrisperWhisper → .json + .srt + .srt.orig
     ↓
Browser UI: two-pane editor (original | edit)
  - Cut, reorder (drag-and-drop), text edit blocks
  - Free Edit mode: edit as raw SRT text
     ↓
On export:
  1. silence.py — amplitude-based silence detection (numpy + ffmpeg)
  2. timeline_export.py — build ordered clip list, generate output
  3. Output: FCPXML / Premiere XML / video via ffmpeg concat
```

## Key Files

| File | Purpose |
|---|---|
| `web_gui.py` | Flask backend — upload, transcribe (SSE), diff, export endpoints |
| `static/index.html` | Full single-page editor UI |
| `static/landing.html` | Marketing landing page |
| `silence.py` | Silence detection: `detect_silence()`, `apply_margin()`, `get_kept_ranges()` |
| `timeline_export.py` | `Clip` dataclass, `build_clip_list()`, `generate_fcpxml()`, `generate_premiere_xml()`, `export_video()` |
| `transcript_diff.py` | SRT parser, WhisperX JSON loader, deleted range detection |
| `merge_cutlists.py` | Legacy auto-editor command builder (no longer used by web GUI) |
| `auto_transcript.py` | WhisperX/CrisperWhisper wrapper for transcription |
| `main.py` | CLI orchestrator (legacy, wraps the old pipeline) |

## Tech Stack

- Python 3.13+ (Homebrew), Flask
- Dependencies: `numpy`, `whisperx`, `flask` (see `requirements.txt`)
- WhisperX requires Python 3.12: `pipx reinstall whisperx --python /opt/homebrew/bin/python3.12`
- FFmpeg must be installed (`brew install ffmpeg`)
- No auto-editor dependency — silence detection and export are handled natively

## Running

```bash
python3 web_gui.py --port 5009          # start on custom port
python3 web_gui.py                       # default: localhost:5000
```

## Export Details

- FCPXML imports into FCP as event "PaperCut Import" with project `{filename}_ALTERED`
- Silence detection threshold parsed from `edit_method` field (e.g. `audio:threshold=0.04`)
- Export pipeline: ordered_blocks from frontend → silence detection → clip list → output format

## Design Constraints

- **JSON is source of truth** for word-level timestamps — SRT timestamps are approximate
- **Block-level operations** in block mode; free-form editing in Free Edit mode
- **Non-monotonic/corrupt SRT timestamps** → log warning, skip block, don't crash
- FCPXML exports reference media by absolute file path; original file must stay in place
- Supports both video and audio-only files across all export formats

## Development Guidelines

- **Do NOT commit or push unless explicitly asked.** Wait for Tim to say when he wants a commit.
- Keep modules independent and testable — each file should work as a standalone unit
- SRT parsing must be robust against messy edits (extra blank lines, missing block numbers, etc.)
- Write to stdout for status messages, stderr for warnings/errors

## Frontend Features

- Two-pane view: original (read-only diff display) | edit (interactive)
- Block mode: cut/restore, drag-and-drop reorder, inline text editing
- Free Edit mode: toggle to edit the entire transcript as raw SRT text
- Undo/redo stack (supports cut, restore, edit, reorder actions)
- Search/find across transcript blocks
- Auto-edit tools: Clean Fillers, Dedupe Takes
- Auto-save to localStorage (preserves cuts, edits, and block order)
- Export presets saved to localStorage
- Video/audio sync playback with scroll sync
- Dark mode toggle
- Keyboard shortcuts: Ctrl+Z/Y undo/redo, Ctrl+F search, arrow keys nav, Delete to cut
