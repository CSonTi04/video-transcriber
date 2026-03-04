""" 
NAS Video → AI Happenings CSV Timeline (Windows, local GPU, IR webcam footage)

What it does:
- Recursively scans a NAS folder for videos
- Copies each video to a local working folder next to this script (./_work/videos)
- Reads embedded media creation time (ffprobe creation_time); falls back to file mtime
- Creates a downscaled "analysis copy" (720p by default)
- Chunks video into fixed windows (e.g., 20s)
- For each chunk:
    - extracts N representative frames (e.g., 6 frames evenly spaced)
    - computes cheap signals (motion_score, luma stats, ir_mode via saturation)
    - feeds the frames to a local multimodal model (Qwen2.5-VL) to produce a visual-only caption + tags
    - optionally runs audio transcription for the chunk (faster-whisper)
- Writes one CSV per input video + a combined CSV for the whole folder

Requirements (Windows):
- Install FFmpeg and ensure `ffmpeg` and `ffprobe` are in PATH
- Python 3.10+ recommended

Install:
  pip install --upgrade pip
  pip install pandas numpy opencv-python pillow tqdm

GPU PyTorch (CUDA 12.1 example):
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

Transformers + Qwen2.5-VL:
  pip install transformers accelerate

Optional audio transcription:
  pip install faster-whisper

Run:
  python video-transcribe.py --in "\\\\NAS\\camera\\clips" --out "./output" --chunk 20 --frames 6

With audio:
  python video-transcribe.py --in "\\\\NAS\\camera\\clips" --out "./output" --chunk 20 --frames 6 --audio

Notes:
- Processing 8 hours can be compute-heavy. Start with chunk=30 and frames=4 if needed.
- Downscaling to 720p speeds up everything and is usually fine for "what happened" summaries.
- This script uses embedded media creation time when available. Xiaomi clips often have it.
"""

import argparse
import hashlib
import json
import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi"}

# Root-relative working directories
SCRIPT_ROOT = os.path.dirname(os.path.abspath(__file__))
WORK_ROOT = os.path.join(SCRIPT_ROOT, "_work")
LOCAL_VIDEO_DIR = os.path.join(WORK_ROOT, "videos")
LOCAL_ANALYSIS_DIR = os.path.join(WORK_ROOT, "analysis")

# Lazy-loaded Haar cascade singleton
_haar_cascade: Optional[cv2.CascadeClassifier] = None


def _get_haar_cascade() -> Optional[cv2.CascadeClassifier]:
    """Load and cache the Haar face cascade classifier (lazy singleton)."""
    global _haar_cascade
    if _haar_cascade is None:
        try:
            cascade_path = os.path.join(
                cv2.data.haarcascades, "haarcascade_frontalface_default.xml"
            )
            _haar_cascade = cv2.CascadeClassifier(cascade_path)
            if _haar_cascade.empty():
                logger.warning("Haar cascade loaded but is empty")
                _haar_cascade = None
        except Exception as e:
            logger.warning("Failed to load Haar cascade: %s", e)
            _haar_cascade = None
    return _haar_cascade


# ----------------------------- Config -----------------------------


@dataclass
class PipelineConfig:
    """Consolidates all pipeline parameters into a single configuration object."""

    # Chunking
    chunk_s: int = 20
    frames_per_chunk: int = 6
    analysis_height: int = 720
    hq_height: Optional[int] = None

    # Skim / skip logic
    skim: bool = False
    skim_motion: float = 0.02
    skim_faces: bool = False

    # Audio
    with_audio: bool = False
    audio_faces_only: bool = False
    audio_vad: bool = False
    audio_vad_threshold: float = 0.01

    # Dynamic frame count
    dynamic_frames: bool = False
    frames_min: int = 2
    frames_max: int = 6
    motion_low: float = 0.01
    motion_high: float = 0.05

    # Language & processing
    language: str = "hu"
    skip_if_done: bool = True


# ------------------------- Utilities -------------------------


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def run_cmd(cmd: List[str]) -> str:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed:\n{' '.join(cmd)}\n\nSTDERR:\n{p.stderr}")
    return p.stdout.strip()


def list_videos(root: str) -> List[str]:
    vids: List[str] = []
    for r, _, files in os.walk(root):
        for fn in files:
            if os.path.splitext(fn)[1].lower() in VIDEO_EXTS:
                vids.append(os.path.join(r, fn))
    vids.sort()
    return vids


def ffprobe_duration_seconds(path: str) -> float:
    out = run_cmd([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path
    ])
    return float(out)


def ffprobe_creation_time_local(path: str) -> datetime:
    """Prefer embedded media creation_time (container tag). 

    Returns local timezone-aware datetime when possible.
    Falls back to file modified time.
    """
    try:
        out = run_cmd([
            "ffprobe", "-v", "error",
            "-print_format", "json",
            "-show_entries", "format_tags=creation_time",
            path
        ])
        data = json.loads(out)
        tags = ((data.get("format") or {}).get("tags") or {})
        ct = tags.get("creation_time")
        if ct:
            ct = ct.strip()
            # Normalize " " -> "T" if needed
            if " " in ct and "T" not in ct:
                ct = ct.replace(" ", "T")

            # Handle Z suffix (UTC)
            if ct.endswith("Z"):
                ct_iso = ct[:-1] + "+00:00"
                dt = datetime.fromisoformat(ct_iso)
                return dt.astimezone()  # local tz

            # Handle explicit offsets
            if re.search(r"[+-]\d\d:\d\d$", ct):
                dt = datetime.fromisoformat(ct)
                return dt.astimezone()

            # No timezone info -> treat as local naive (best effort)
            return datetime.fromisoformat(ct)
    except Exception as e:
        logger.debug("Could not read creation_time from %s: %s", path, e)

    # Fallback: filesystem modified time
    ts = os.path.getmtime(path)
    return datetime.fromtimestamp(ts).astimezone()


def copy_to_local_workdir(src_video: str) -> str:
    """Copy a NAS video to local ./_work/videos if not already present.

    Uses a hash prefix to avoid basename collisions from different source paths.
    """
    ensure_dir(LOCAL_VIDEO_DIR)
    # Hash the full source path to avoid collisions
    path_hash = hashlib.md5(src_video.encode()).hexdigest()[:8]
    basename = os.path.basename(src_video)
    dst_name = f"{path_hash}_{basename}"
    dst = os.path.join(LOCAL_VIDEO_DIR, dst_name)

    if os.path.exists(dst):
        # Already copied
        return dst

    logger.info("Copying from NAS → local: %s", src_video)
    shutil.copy2(src_video, dst)  # preserves mtime
    return dst


def downscale_to_analysis_copy(src: str, dst: str, height: int = 720) -> None:
    """Create a downscaled no-audio analysis copy."""
    if os.path.exists(dst):
        return
    ensure_dir(os.path.dirname(dst))
    run_cmd([
        "ffmpeg", "-y", "-i", src,
        "-vf", f"scale=-2:{height}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-an",
        dst
    ])


def extract_frames_for_chunk(
    video_path: str, start_s: float, dur_s: float, n_frames: int, out_dir: str
) -> List[str]:
    """Extract N frames evenly spaced across the chunk using a single ffmpeg call."""
    ensure_dir(out_dir)

    frame_paths: List[str] = []
    if n_frames <= 0:
        return frame_paths

    # Avoid very edges
    eps = min(0.2, dur_s / 10.0)
    t0 = start_s + eps
    t1 = start_s + max(eps, dur_s - eps)
    if t1 <= t0:
        t0 = start_s
        t1 = start_s + dur_s

    times = np.linspace(t0, t1, num=n_frames, endpoint=True)

    # Build a select filter for all frames in one ffmpeg call
    # select='eq(t,T0)+eq(t,T1)+...', outputting each selected frame
    select_expr = "+".join(f"lte(prev_pts*TB,{ts:.3f})*gte(pts*TB,{ts:.3f})" for ts in times)

    output_pattern = os.path.join(out_dir, "frame_%02d.jpg")
    try:
        run_cmd([
            "ffmpeg", "-y",
            "-ss", f"{t0:.3f}",
            "-i", video_path,
            "-t", f"{(t1 - t0 + 1.0):.3f}",
            "-vf", f"fps=1/{max(0.1, dur_s / n_frames):.3f}",
            "-frames:v", str(n_frames),
            "-q:v", "2",
            "-pix_fmt", "yuvj420p",
            output_pattern,
        ])
    except Exception as e:
        logger.warning("Batch frame extraction failed, falling back to per-frame: %s", e)
        # Fallback: extract frames one by one
        for i, ts in enumerate(times):
            fp = os.path.join(out_dir, f"frame_{i:02d}.jpg")
            try:
                run_cmd([
                    "ffmpeg", "-y",
                    "-ss", f"{ts:.3f}",
                    "-i", video_path,
                    "-frames:v", "1",
                    "-q:v", "2",
                    "-pix_fmt", "yuvj420p",
                    fp
                ])
            except Exception as e2:
                logger.debug("Frame extraction failed at t=%.3f: %s", ts, e2)
                continue

    # Collect whatever frames were actually written
    for i in range(n_frames):
        fp = os.path.join(out_dir, f"frame_{i:02d}.jpg")
        if os.path.exists(fp):
            frame_paths.append(fp)

    return frame_paths


def read_frames_at_timestamps(
    video_path: str, timestamps: List[float]
) -> List[Optional[np.ndarray]]:
    """Open a VideoCapture once and read frames at multiple timestamps.

    Returns a list of frames (or None for failed reads), one per timestamp.
    """
    results: List[Optional[np.ndarray]] = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [None] * len(timestamps)
    try:
        for ts_s in timestamps:
            cap.set(cv2.CAP_PROP_POS_MSEC, ts_s * 1000.0)
            ok, frame = cap.read()
            results.append(frame if ok else None)
    finally:
        cap.release()
    return results


def read_frame_for_signals(video_path: str, ts_s: float) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_MSEC, ts_s * 1000.0)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def compute_chunk_signals(
    video_path: str, start_s: float, dur_s: float, sample_points: int = 5
) -> Dict[str, float]:
    """Cheap signals: motion, luma stats, IR mode via saturation."""
    if sample_points < 2:
        sample_points = 2

    times = np.linspace(start_s, start_s + dur_s, num=sample_points, endpoint=False)
    results = read_frames_at_timestamps(video_path, times.tolist())
    frames = [f for f in results if f is not None]

    if len(frames) < 2:
        return {
            "motion_score": 0.0,
            "luma_mean": 0.0,
            "luma_std": 0.0,
            "ir_mode": "unknown",
            "sat_mean": 0.0,
        }

    # Resize for fast signals
    small_frames = [cv2.resize(fr, (320, 180), interpolation=cv2.INTER_AREA) for fr in frames]

    diffs: List[float] = []
    lumas: List[float] = []
    sats: List[float] = []

    prev_gray = None
    for fr in small_frames:
        hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1].astype(np.float32) / 255.0
        sats.append(float(np.mean(sat)))

        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        lumas.append(float(np.mean(gray)))

        if prev_gray is not None:
            diffs.append(float(np.mean(np.abs(gray - prev_gray))))
        prev_gray = gray

    motion_score = float(np.mean(diffs)) if diffs else 0.0
    luma_mean = float(np.mean(lumas))
    luma_std = float(np.std(lumas))
    sat_mean = float(np.mean(sats))

    # IR heuristic: near-grayscale -> low saturation
    ir_mode = "ir" if sat_mean < 0.10 else "day"

    return {
        "motion_score": motion_score,
        "luma_mean": luma_mean,
        "luma_std": luma_std,
        "ir_mode": ir_mode,
        "sat_mean": sat_mean,
    }


def sample_frames_for_skim(
    video_path: str, start_s: float, dur_s: float, sample_points: int = 3
) -> List[np.ndarray]:
    if sample_points < 1:
        sample_points = 1
    times = np.linspace(start_s, start_s + dur_s, num=sample_points, endpoint=False)
    results = read_frames_at_timestamps(video_path, times.tolist())
    return [f for f in results if f is not None]


def detect_faces_in_frames(
    frames: List[np.ndarray],
    scale_factor: float = 1.1,
    min_neighbors: int = 4,
) -> int:
    if not frames:
        return 0
    cascade = _get_haar_cascade()
    if cascade is None:
        return 0

    total = 0
    for fr in frames:
        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=(30, 30)
        )
        total += len(faces)
    return total


def choose_frames_per_chunk(
    base_frames: int,
    motion_score: float,
    dynamic: bool = False,
    motion_low: float = 0.01,
    motion_high: float = 0.05,
    min_frames: int = 2,
    max_frames: int = 6
) -> int:
    if not dynamic:
        return int(base_frames)
    if motion_score <= motion_low:
        return int(min_frames)
    if motion_score >= motion_high:
        return int(max_frames)
    t = (motion_score - motion_low) / max(1e-6, (motion_high - motion_low))
    frames = min_frames + t * (max_frames - min_frames)
    return int(max(min_frames, min(max_frames, round(frames))))


def audio_has_speech(
    video_path: str,
    start_s: float,
    dur_s: float,
    rms_threshold: float = 0.01
) -> bool:
    """Cheap audio gate: compute RMS energy from a mono 16kHz wav clip."""
    try:
        with tempfile.TemporaryDirectory() as td:
            wav = os.path.join(td, "gate.wav")
            run_cmd([
                "ffmpeg", "-y",
                "-ss", f"{start_s:.3f}",
                "-t", f"{dur_s:.3f}",
                "-i", video_path,
                "-vn",
                "-ac", "1",
                "-ar", "16000",
                "-f", "wav",
                wav
            ])

            with wave.open(wav, "rb") as wf:
                frames = wf.readframes(wf.getnframes())
                if not frames:
                    return False
                samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                rms = float(np.sqrt(np.mean(samples ** 2)))
                return rms >= rms_threshold
    except Exception as e:
        logger.warning("Audio VAD check failed for %s at t=%.1f: %s", video_path, start_s, e)
        return False


def robust_z_scores(values: List[float]) -> List[float]:
    """Robust z-score using median absolute deviation (MAD)."""
    if not values:
        return []
    x = np.array(values, dtype=np.float32)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad < 1e-8:
        return [0.0 for _ in values]
    z = 0.6745 * (x - med) / mad
    return z.tolist()


# ------------------------- AI Captioning -------------------------

@dataclass
class CaptionResult:
    caption: str
    tags: str
    confidence: Optional[float] = None


class VisualCaptionerQwen25VL:
    """Caption a sequence of frames with Qwen2.5-VL."""

    def __init__(self, model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct", use_fp16: bool = True):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_id)
        dtype = torch.float16 if use_fp16 else torch.float32

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto"
        )
        self.model.eval()

    def caption_frames(self, frame_paths: List[str], prompt: str, max_new_tokens: int = 220) -> CaptionResult:
        from PIL import Image

        images = [Image.open(p).convert("RGB") for p in frame_paths]

        # Structured output -> better CSV filtering
        full_prompt = (
            prompt
            + "\n\nReturn JSON with keys: "
              "caption (string, 1-3 sentences), "
              "tags (comma-separated, <=8 tags). "
              "Tags examples: person_present, entered_room, left_frame, sat_down, stood_up, "
              "lights_changed, picked_up_object, phone_visible, camera_obstructed, nothing_happened."
        )

        messages = [{"role": "user", "content": []}]
        for img in images:
            messages[0]["content"].append({"type": "image", "image": img})
        messages[0]["content"].append({"type": "text", "text": full_prompt})

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=images, return_tensors="pt")

        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        decoded = self.processor.batch_decode(out, skip_special_tokens=True)[0]
        caption, tags = self._extract_json(decoded)
        return CaptionResult(caption=caption, tags=tags)

    @staticmethod
    def _extract_json(text: str) -> Tuple[str, str]:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
                cap = str(obj.get("caption", "")).strip()
                tags = str(obj.get("tags", "")).strip()
                if cap:
                    return cap, tags
            except Exception as e:
                logger.debug("JSON parse failed in model output: %s", e)
        return text.strip(), ""


# ------------------------- Optional Audio -------------------------

class ChunkTranscriber:
    """Transcribe chunks with faster-whisper. Supports transcription and translation."""

    def __init__(self, model_size: str = "small", device: str = "cuda", language: str = "hu", task: str = "translate"):
        from faster_whisper import WhisperModel
        self.model = WhisperModel(model_size, device=device, compute_type="float16")
        self.language = language  # 'hu' for Hungarian, 'en' for English, None for auto-detect
        self.task = task  # 'transcribe' or 'translate'

    def transcribe_chunk(self, video_path: str, start_s: float, dur_s: float) -> str:
        with tempfile.TemporaryDirectory() as td:
            wav = os.path.join(td, "chunk.wav")
            run_cmd([
                "ffmpeg", "-y",
                "-ss", f"{start_s:.3f}",
                "-t", f"{dur_s:.3f}",
                "-i", video_path,
                "-vn",
                "-ac", "1",
                "-ar", "16000",
                wav
            ])
            segments, _info = self.model.transcribe(wav, vad_filter=True, language=self.language, task=self.task)
            parts: List[str] = []
            for seg in segments:
                parts.append(seg.text.strip())
            return " ".join([p for p in parts if p])


# -------------------- Decomposed Processing --------------------


def prepare_analysis_copies(
    video_path: str, config: PipelineConfig
) -> Tuple[str, str]:
    """Create analysis and optional HQ copies. Returns (analysis_path, hq_path)."""
    base = os.path.splitext(os.path.basename(video_path))[0]
    analysis_path = os.path.join(
        LOCAL_ANALYSIS_DIR, f"{base}__analysis_{config.analysis_height}p.mp4"
    )
    downscale_to_analysis_copy(video_path, analysis_path, height=config.analysis_height)

    hq_height = config.hq_height if config.hq_height is not None else config.analysis_height
    hq_path = analysis_path
    if hq_height > config.analysis_height:
        hq_path = os.path.join(
            LOCAL_ANALYSIS_DIR, f"{base}__analysis_{hq_height}p.mp4"
        )
        downscale_to_analysis_copy(video_path, hq_path, height=hq_height)

    return analysis_path, hq_path


def process_single_chunk(
    idx: int,
    start_s: float,
    this_dur: float,
    clip_start: datetime,
    analysis_path: str,
    hq_path: str,
    video_path: str,
    temp_root: str,
    captioner: "VisualCaptionerQwen25VL",
    config: PipelineConfig,
    transcriber: Optional["ChunkTranscriber"] = None,
) -> Dict[str, Any]:
    """Process a single chunk and return a row dict."""
    abs_start = clip_start + timedelta(seconds=float(start_s))
    abs_end = clip_start + timedelta(seconds=float(start_s + this_dur))

    signals = compute_chunk_signals(analysis_path, float(start_s), float(this_dur), sample_points=5)

    skim_trigger = True
    faces_detected = 0
    if config.skim:
        skim_frames = sample_frames_for_skim(
            analysis_path, float(start_s), float(this_dur), sample_points=3
        )
        if config.skim_faces:
            faces_detected = detect_faces_in_frames(skim_frames)

        motion_trigger = float(signals["motion_score"]) >= float(config.skim_motion)
        faces_trigger = faces_detected > 0
        skim_trigger = motion_trigger or faces_trigger

    cap_res = CaptionResult(caption="(skipped low activity)", tags="")
    if skim_trigger:
        frames_to_use = choose_frames_per_chunk(
            config.frames_per_chunk,
            motion_score=float(signals["motion_score"]),
            dynamic=config.dynamic_frames,
            motion_low=config.motion_low,
            motion_high=config.motion_high,
            min_frames=config.frames_min,
            max_frames=config.frames_max,
        )
        chunk_frame_dir = os.path.join(temp_root, f"chunk_{idx:05d}")
        frame_paths = extract_frames_for_chunk(
            hq_path, float(start_s), float(this_dur), frames_to_use, chunk_frame_dir
        )

        if config.language == "hu":
            prompt = (
                "Írj le, hogy mi történik ebben az időszakban, KÉP ALAPJÁN. Figyelmen kívül hagyj minden hangot/beszédet. "
                "Legyél konkrét emberekről/tárgyakról és cselekedetekről. Ha bizonytalan vagy, írd azt, hogy 'unclear'."
            )
        else:  # English
            prompt = (
                "Describe what happens in this time segment using VISUALS ONLY. Ignore any audio/speech. "
                "Be concrete about people/objects and actions. If uncertain, say 'unclear'."
            )
        cap_res = captioner.caption_frames(frame_paths, prompt=prompt, max_new_tokens=220)

    transcript = ""
    if config.with_audio and transcriber is not None and (not config.skim or skim_trigger):
        speech_ok = True
        if config.audio_vad:
            speech_ok = audio_has_speech(
                video_path, float(start_s), float(this_dur),
                rms_threshold=config.audio_vad_threshold,
            )

        if (not config.audio_faces_only or faces_detected > 0) and speech_ok:
            try:
                transcript = transcriber.transcribe_chunk(video_path, float(start_s), float(this_dur))
            except Exception as e:
                logger.warning("Audio transcription failed at chunk %d: %s", idx, e)
                transcript = ""

    return {
        "abs_start": abs_start.strftime("%Y-%m-%d %H:%M:%S"),
        "abs_end": abs_end.strftime("%Y-%m-%d %H:%M:%S"),
        "clip_file": os.path.basename(video_path),
        "clip_start_media_time": clip_start.strftime("%Y-%m-%d %H:%M:%S"),
        "offset_start_s": float(start_s),
        "offset_end_s": float(start_s + this_dur),

        "motion_score": float(signals["motion_score"]),
        "luma_mean": float(signals["luma_mean"]),
        "luma_std": float(signals["luma_std"]),
        "sat_mean": float(signals["sat_mean"]),
        "ir_mode": signals["ir_mode"],

        "skim_triggered": bool(skim_trigger),
        "faces_detected": int(faces_detected),

        "caption_visual": cap_res.caption,
        "tags_visual": cap_res.tags,

        "transcript_audio": transcript,
    }


def compute_anomaly_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Add anomaly scoring columns to a DataFrame of chunk rows."""
    if not df.empty:
        motion_z = robust_z_scores(df["motion_score"].tolist())
        df["anomaly_z_motion"] = motion_z
        z = np.abs(df["anomaly_z_motion"].astype(float).to_numpy())
        df["anomaly_score_motion"] = (z / (z + 3.0)).astype(float)  # 0..1-ish

        def has_person(tags: str) -> int:
            t = (tags or "").lower()
            return 1 if "person_present" in t else 0

        person_flag = df["tags_visual"].fillna("").map(has_person).astype(float)
        df["person_present_tag"] = person_flag
        df["anomaly_score_combined"] = np.clip(
            df["anomaly_score_motion"].astype(float) + 0.25 * person_flag,
            0.0, 1.0
        )
    else:
        df["anomaly_z_motion"] = []
        df["anomaly_score_motion"] = []
        df["person_present_tag"] = []
        df["anomaly_score_combined"] = []
    return df


def _load_progress(progress_path: str) -> List[Dict[str, Any]]:
    """Load existing chunk rows from a JSONL progress file."""
    rows: List[Dict[str, Any]] = []
    if os.path.exists(progress_path):
        with open(progress_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning("Malformed progress line: %s", e)
    return rows


def _save_chunk_row(progress_path: str, row: Dict[str, Any]) -> None:
    """Append a single chunk row to the JSONL progress file."""
    with open(progress_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ------------------------- Main Processing -------------------------


def process_video(
    video_path: str,
    out_dir: str,
    captioner: "VisualCaptionerQwen25VL",
    chunk_s: int = 20,
    frames_per_chunk: int = 6,
    analysis_height: int = 720,
    with_audio: bool = False,
    transcriber: Optional["ChunkTranscriber"] = None,
    skip_if_done: bool = True,
    language: str = "hu",
    skim: bool = False,
    skim_motion: float = 0.02,
    skim_faces: bool = False,
    hq_height: Optional[int] = None,
    audio_faces_only: bool = False,
    audio_vad: bool = False,
    audio_vad_threshold: float = 0.01,
    dynamic_frames: bool = False,
    frames_min: int = 2,
    frames_max: int = 6,
    motion_low: float = 0.01,
    motion_high: float = 0.05
) -> pd.DataFrame:
    """Process a video file into a timeline CSV.

    This function preserves the original call signature for backwards compatibility.
    Internally it delegates to the decomposed helpers and PipelineConfig.
    """
    base = os.path.splitext(os.path.basename(video_path))[0]
    ensure_dir(out_dir)

    per_csv = os.path.join(out_dir, f"{base}.timeline.csv")
    if skip_if_done and os.path.exists(per_csv):
        logger.info("Skipping (already processed): %s", base)
        return pd.read_csv(per_csv)

    config = PipelineConfig(
        chunk_s=chunk_s,
        frames_per_chunk=frames_per_chunk,
        analysis_height=analysis_height,
        hq_height=hq_height,
        skim=skim,
        skim_motion=skim_motion,
        skim_faces=skim_faces,
        with_audio=with_audio,
        audio_faces_only=audio_faces_only,
        audio_vad=audio_vad,
        audio_vad_threshold=audio_vad_threshold,
        dynamic_frames=dynamic_frames,
        frames_min=frames_min,
        frames_max=frames_max,
        motion_low=motion_low,
        motion_high=motion_high,
        language=language,
        skip_if_done=skip_if_done,
    )

    clip_start = ffprobe_creation_time_local(video_path)
    dur = ffprobe_duration_seconds(video_path)

    analysis_path, hq_path = prepare_analysis_copies(video_path, config)

    n_chunks = int(math.ceil(dur / chunk_s))

    # Resume support: load previously processed chunks
    progress_path = os.path.join(out_dir, f".{base}.progress.jsonl")
    existing_rows = _load_progress(progress_path)
    completed_indices = {int(r.get("_chunk_idx", -1)) for r in existing_rows}

    temp_root = tempfile.mkdtemp(prefix=f"frames_{base}_")

    rows: List[Dict[str, Any]] = list(existing_rows)
    try:
        for idx in tqdm(range(n_chunks), desc=f"Chunks: {base}", unit="chunk"):
            start_s = idx * chunk_s
            this_dur = min(chunk_s, dur - start_s)
            if this_dur <= 0.05:
                break

            # Skip already-processed chunks (resume support)
            if idx in completed_indices:
                continue

            row = process_single_chunk(
                idx=idx,
                start_s=start_s,
                this_dur=this_dur,
                clip_start=clip_start,
                analysis_path=analysis_path,
                hq_path=hq_path,
                video_path=video_path,
                temp_root=temp_root,
                captioner=captioner,
                config=config,
                transcriber=transcriber,
            )
            row["_chunk_idx"] = idx
            rows.append(row)
            _save_chunk_row(progress_path, row)

        # Sort by chunk index and remove the internal tracking field
        rows.sort(key=lambda r: r.get("_chunk_idx", 0))
        for r in rows:
            r.pop("_chunk_idx", None)

        df = pd.DataFrame(rows)
        df = compute_anomaly_scores(df)

        df.to_csv(per_csv, index=False, encoding="utf-8-sig")

        # Clean up progress file on successful completion
        if os.path.exists(progress_path):
            os.remove(progress_path)

        return df

    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


# ------------------------- Entry Point -------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="NAS Video → AI Happenings CSV Timeline"
    )
    ap.add_argument("--in", dest="in_dir", required=True, help="Input folder containing videos (NAS path ok)")
    ap.add_argument("--out", dest="out_dir", required=True, help="Output folder (relative ok)")
    ap.add_argument("--chunk", type=int, default=40, help="Chunk/window size in seconds (default: 40)")
    ap.add_argument("--frames", type=int, default=3, help="Frames per chunk for captioning (default: 3)")
    ap.add_argument("--height", type=int, default=480, help="Downscale height for analysis copy (default: 480)")
    ap.add_argument("--hq-height", type=int, default=720, help="Height for HQ AI frames (use > --height for HQ pass)")
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help="VLM model id")
    ap.add_argument("--audio", action="store_true", help="Enable audio transcription column")
    ap.add_argument("--whisper_model", type=str, default="small", help="faster-whisper size")
    ap.add_argument("--language", type=str, default="hu", choices=["hu", "en"], help="Language mode: 'hu' for Hungarian or 'en' for English (default: hu)")
    ap.add_argument("--skim", action="store_true", help="Enable low-cost skim to skip AI on low-activity chunks")
    ap.add_argument("--skim-motion", type=float, default=0.02, help="Motion threshold to trigger HQ AI (default: 0.02)")
    ap.add_argument("--skim-faces", action="store_true", help="Use Haar face detection to trigger HQ AI")
    ap.add_argument("--audio-faces-only", action="store_true", help="Only transcribe audio if faces are detected during skim")
    ap.add_argument("--audio-vad", action="store_true", help="Enable cheap audio VAD gate before Whisper")
    ap.add_argument("--audio-vad-threshold", type=float, default=0.01, help="RMS threshold for audio VAD (default: 0.01)")
    ap.add_argument("--dynamic-frames", action="store_true", help="Adjust frame count based on motion")
    ap.add_argument("--frames-min", type=int, default=2, help="Min frames per chunk when motion is low")
    ap.add_argument("--frames-max", type=int, default=6, help="Max frames per chunk when motion is high")
    ap.add_argument("--motion-low", type=float, default=0.01, help="Motion score for low activity (default: 0.01)")
    ap.add_argument("--motion-high", type=float, default=0.05, help="Motion score for high activity (default: 0.05)")
    ap.add_argument("--no-skip", action="store_true", help="Reprocess even if CSV exists")
    ap.add_argument("--dry-run", action="store_true", help="List videos and estimated work without processing")

    args = ap.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    ensure_dir(WORK_ROOT)
    ensure_dir(LOCAL_VIDEO_DIR)
    ensure_dir(LOCAL_ANALYSIS_DIR)
    ensure_dir(args.out_dir)

    videos_on_nas = list_videos(args.in_dir)
    if not videos_on_nas:
        logger.info("No videos found.")
        return

    # ---- Dry-run mode ----
    if args.dry_run:
        logger.info("=== DRY RUN === (no processing will occur)")
        total_chunks = 0
        for vp in videos_on_nas:
            try:
                dur = ffprobe_duration_seconds(vp)
                n_chunks = int(math.ceil(dur / args.chunk))
                total_chunks += n_chunks
                logger.info(
                    "  %s  |  %.1fs  |  %d chunks",
                    os.path.basename(vp), dur, n_chunks,
                )
            except Exception as e:
                logger.warning("  %s  |  ERROR: %s", os.path.basename(vp), e)
        logger.info(
            "Total: %d videos, ~%d chunks (chunk=%ds)",
            len(videos_on_nas), total_chunks, args.chunk,
        )
        return

    # ---- Pipeline NAS copy with processing ----
    # Copy video N+1 in background while processing video N
    local_videos: List[str] = []

    with ThreadPoolExecutor(max_workers=1) as copy_pool:
        # Submit all copy jobs upfront
        copy_futures = {
            copy_pool.submit(copy_to_local_workdir, vp): vp
            for vp in videos_on_nas
        }
        for future in copy_futures:
            try:
                local_videos.append(future.result())
            except Exception as e:
                logger.warning("Failed to copy %s: %s", copy_futures[future], e)

    # Load captioner once
    captioner = VisualCaptionerQwen25VL(model_id=args.model, use_fp16=True)

    transcriber = None
    if args.audio:
        # For Hungarian: transcribe Hungarian and translate to English
        # For English: transcribe English
        if args.language == "hu":
            transcriber = ChunkTranscriber(model_size=args.whisper_model, device="cuda", language="hu", task="translate")
        else:
            transcriber = ChunkTranscriber(model_size=args.whisper_model, device="cuda", language="en", task="transcribe")

    all_dfs: List[pd.DataFrame] = []
    for i, local_vp in enumerate(
        tqdm(local_videos, desc="Videos", unit="video"),
        start=1,
    ):
        logger.info("Processing video %d/%d: %s", i, len(local_videos), local_vp)
        df = process_video(
            local_vp,
            out_dir=args.out_dir,
            captioner=captioner,
            chunk_s=args.chunk,
            frames_per_chunk=args.frames,
            analysis_height=args.height,
            with_audio=args.audio,
            transcriber=transcriber,
            skip_if_done=(not args.no_skip),
            language=args.language,
            skim=args.skim,
            skim_motion=args.skim_motion,
            skim_faces=args.skim_faces,
            hq_height=args.hq_height,
            audio_faces_only=args.audio_faces_only,
            audio_vad=args.audio_vad,
            audio_vad_threshold=args.audio_vad_threshold,
            dynamic_frames=args.dynamic_frames,
            frames_min=args.frames_min,
            frames_max=args.frames_max,
            motion_low=args.motion_low,
            motion_high=args.motion_high
        )
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    combined_csv = os.path.join(args.out_dir, "_combined.timeline.csv")
    combined.to_csv(combined_csv, index=False, encoding="utf-8-sig")

    logger.info("Done.")
    logger.info("Per-video CSVs: %s\\<video>.timeline.csv", args.out_dir)
    logger.info("Combined CSV:   %s", combined_csv)
    logger.info("Tip: sort/filter by anomaly_score_combined desc, and/or tags_visual contains 'person_present'.")


if __name__ == "__main__":
    main()
