# Video Transcriber

**NAS Video → AI-Powered CSV Timeline** — Automatically analyse security/webcam footage and produce structured, searchable timelines with visual captions, tags, anomaly scores, and optional audio transcription.

Built for Windows with local GPU inference (no cloud APIs). Designed for IR webcam footage but works with any video.

---

## What It Does

```
NAS video folder ──► local copy ──► downscale ──► chunk into windows ──► per-chunk analysis ──► CSV timeline
```

For each video:
1. **Copies** from NAS to a local working directory (pipelined for speed)
2. **Downscales** to an analysis copy (480p by default, configurable)
3. **Chunks** into fixed time windows (e.g., 40 seconds)
4. For each chunk:
   - Extracts representative frames (evenly spaced)
   - Computes cheap signals: **motion score**, **luma stats**, **IR/day mode** detection
   - Feeds frames to a local **Qwen2.5-VL** multimodal model for visual captioning + tagging
   - Optionally runs **faster-whisper** audio transcription (with Hungarian → English translation support)
5. Computes **anomaly scores** (robust z-scores on motion + person detection)
6. Writes per-video CSVs + a combined timeline CSV

### Output Columns

| Column | Description |
|---|---|
| `abs_start` / `abs_end` | Absolute timestamps (from embedded media creation time) |
| `clip_file` | Source video filename |
| `motion_score` | Mean frame-to-frame pixel difference |
| `luma_mean` / `luma_std` | Brightness statistics |
| `ir_mode` | `ir` or `day` (detected via saturation) |
| `caption_visual` | AI-generated description of visual activity |
| `tags_visual` | Structured tags (e.g., `person_present, entered_room, sat_down`) |
| `transcript_audio` | Whisper transcription (if `--audio` enabled) |
| `anomaly_score_combined` | 0–1 anomaly score combining motion + person detection |

---

## Requirements

- **OS**: Windows (tested), should work on Linux/macOS
- **Python**: 3.10+
- **FFmpeg**: `ffmpeg` and `ffprobe` must be in PATH
- **GPU**: NVIDIA GPU with CUDA (for Qwen2.5-VL and faster-whisper)

## Installation

```bash
# Create virtual environment
python -m venv venv
.\venv\Scripts\activate     # Windows
# source venv/bin/activate  # Linux/macOS

# Core dependencies
pip install pandas numpy opencv-python pillow tqdm

# GPU PyTorch (CUDA 12.1 example)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Qwen2.5-VL model
pip install transformers accelerate

# Optional: audio transcription
pip install faster-whisper
```

Or install everything at once:

```bash
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## Usage

### Basic

```bash
python video-transcribe.py --in "\\NAS\camera\clips" --out "./output"
```

### With Audio Transcription

```bash
python video-transcribe.py --in "\\NAS\camera\clips" --out "./output" --audio
```

### Optimised for Long Recordings

```bash
python video-transcribe.py \
  --in "\\NAS\camera\clips" --out "./output" \
  --chunk 40 --frames 3 --height 480 --hq-height 720 \
  --skim --skim-motion 0.02 --skim-faces \
  --dynamic-frames --frames-min 2 --frames-max 6 \
  --audio --audio-vad
```

### Dry Run (Preview Without Processing)

```bash
python video-transcribe.py --in "\\NAS\camera\clips" --out "./output" --dry-run
```

Lists all videos, their durations, and estimated chunk counts without processing anything.

---

## CLI Reference

| Flag | Default | Description |
|---|---|---|
| `--in` | *(required)* | Input folder containing videos (NAS UNC paths supported) |
| `--out` | *(required)* | Output folder for CSV files |
| `--chunk` | `40` | Chunk window size in seconds |
| `--frames` | `3` | Frames per chunk for AI captioning |
| `--height` | `480` | Downscale height for analysis (signal computation) |
| `--hq-height` | `720` | Height for frames sent to the AI model |
| `--model` | `Qwen/Qwen2.5-VL-7B-Instruct` | HuggingFace model ID for visual captioning |
| `--language` | `hu` | Language mode: `hu` (Hungarian) or `en` (English) |
| `--audio` | off | Enable audio transcription via faster-whisper |
| `--whisper_model` | `small` | Whisper model size (`tiny`, `small`, `medium`, `large`) |
| `--skim` | off | Skip AI captioning on low-activity chunks |
| `--skim-motion` | `0.02` | Motion threshold to trigger AI on a chunk |
| `--skim-faces` | off | Also trigger AI when faces are detected |
| `--audio-vad` | off | Gate Whisper with a cheap RMS energy check |
| `--audio-vad-threshold` | `0.01` | RMS threshold for the VAD gate |
| `--audio-faces-only` | off | Only run Whisper on chunks with detected faces |
| `--dynamic-frames` | off | Adjust frame count per chunk based on motion |
| `--frames-min` | `2` | Min frames when motion is low (with `--dynamic-frames`) |
| `--frames-max` | `6` | Max frames when motion is high |
| `--motion-low` | `0.01` | Motion score threshold for "low activity" |
| `--motion-high` | `0.05` | Motion score threshold for "high activity" |
| `--no-skip` | off | Reprocess even if a CSV already exists |
| `--dry-run` | off | Preview mode — list videos without processing |

---

## Features

### Smart Skimming (`--skim`)

Computes cheap motion and face-detection signals *before* running the expensive AI model. Chunks below the motion threshold (and without faces, if `--skim-faces` is enabled) get a `(skipped low activity)` caption, saving substantial GPU time on quiet footage.

### Resume Support

If a run is interrupted mid-video (crash, Ctrl+C, power loss), restart with the same command — it automatically picks up from the last completed chunk. Progress is tracked via `.progress.jsonl` files in the output directory, which are cleaned up on successful completion.

### Dynamic Frame Count (`--dynamic-frames`)

Instead of a fixed number of frames per chunk, scales the frame count based on detected motion:
- Low motion → fewer frames (saves GPU)
- High motion → more frames (better captions)

### Dual Resolution

Signal computation (motion, luma, IR detection) runs on a low-res copy (`--height`), while AI captioning uses a higher-res copy (`--hq-height`). This balances speed and caption quality.

### Audio VAD Gate (`--audio-vad`)

Before running Whisper on a chunk, a cheap RMS energy check filters out silence. This avoids wasting compute on chunks with no speech.

---

## Project Structure

```
video-transcriber/
├── video-transcribe.py       # Main pipeline script
├── test_video_transcribe.py  # pytest test suite (65 tests, all mocked)
├── requirements.txt          # Python dependencies
├── .gitignore
├── _work/                    # Auto-created working directory
│   ├── videos/               #   Local copies of NAS videos
│   └── analysis/             #   Downscaled analysis copies
└── output/                   # Default output directory (CSVs)
```

---

## Testing

All tests are pure-unit with mocked externals — no GPU, ffmpeg, or media files needed.

```bash
pip install pytest
python -m pytest test_video_transcribe.py -v
```

```
65 passed in ~1.6s
```

---

## Tips

- **Start conservative**: Use `--chunk 40 --frames 3` for initial runs to limit GPU cost
- **Use `--skim`** on long recordings to skip AI on boring chunks
- **Sort output** by `anomaly_score_combined` descending to find interesting moments
- **Filter by tags**: Look for `person_present`, `entered_room`, `picked_up_object`, etc.
- **Hungarian footage**: Use `--language hu --audio` to transcribe Hungarian speech and translate to English

---

## License

This project is licensed under the **Business Source License 1.1 (BSL 1.1)**. 

- **Non-Production Use:** Free for non-production use (e.g., testing, internal development, personal use).
- **Production Use:** Requires a commercial license.
- **Change License:** On **March 4, 2030** (or 4 years after a specific version's release), the license for that version converts to the **GNU General Public License v3.0 or later (GPL-3.0-or-later)**.

See the [LICENSE](LICENSE) file for the full text.
