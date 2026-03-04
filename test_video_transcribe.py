"""
Comprehensive test suite for video-transcribe.py

All external dependencies (ffmpeg, ffprobe, GPU models, filesystem I/O)
are mocked so tests run instantly without any hardware or media files.

Run:
    python -m pytest test_video_transcribe.py -v
"""

import os
import sys
import json
import math
import tempfile
import shutil
from datetime import datetime, timezone, timedelta
from unittest import mock
from unittest.mock import patch, MagicMock, PropertyMock, call

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Import the module under test.  The filename has a hyphen, so use importlib.
# ---------------------------------------------------------------------------
import importlib

_spec = importlib.util.spec_from_file_location(
    "video_transcribe",
    os.path.join(os.path.dirname(__file__), "video-transcribe.py"),
)
vt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vt)

import cv2  # needed for cv2 constants


# ========================== Utility Tests ==========================


class TestEnsureDir:
    def test_creates_directory(self, tmp_path):
        d = str(tmp_path / "sub" / "dir")
        assert not os.path.exists(d)
        vt.ensure_dir(d)
        assert os.path.isdir(d)

    def test_idempotent(self, tmp_path):
        d = str(tmp_path / "existing")
        os.makedirs(d)
        vt.ensure_dir(d)  # should not raise
        assert os.path.isdir(d)


class TestRunCmd:
    @patch("subprocess.run")
    def test_success(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="  hello  \n", stderr="")
        result = vt.run_cmd(["echo", "hello"])
        assert result == "hello"

    @patch("subprocess.run")
    def test_failure_raises(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="bad arg")
        with pytest.raises(RuntimeError, match="Command failed"):
            vt.run_cmd(["bad", "cmd"])


class TestListVideos:
    def test_finds_videos_recursively(self, tmp_path):
        sub = tmp_path / "cam1"
        sub.mkdir()
        (tmp_path / "clip1.mp4").write_text("")
        (sub / "clip2.mkv").write_text("")
        (sub / "clip3.mov").write_text("")
        (tmp_path / "readme.txt").write_text("")
        (tmp_path / "photo.jpg").write_text("")
        (sub / "clip4.avi").write_text("")

        vids = vt.list_videos(str(tmp_path))
        basenames = sorted(os.path.basename(v) for v in vids)
        assert basenames == ["clip1.mp4", "clip2.mkv", "clip3.mov", "clip4.avi"]

    def test_returns_sorted(self, tmp_path):
        (tmp_path / "z.mp4").write_text("")
        (tmp_path / "a.mp4").write_text("")
        (tmp_path / "m.mp4").write_text("")
        vids = vt.list_videos(str(tmp_path))
        paths = [os.path.basename(v) for v in vids]
        assert paths == sorted(paths)

    def test_empty_dir(self, tmp_path):
        assert vt.list_videos(str(tmp_path)) == []


class TestRobustZScores:
    def test_empty(self):
        assert vt.robust_z_scores([]) == []

    def test_constant_values(self):
        vals = [5.0, 5.0, 5.0, 5.0]
        z = vt.robust_z_scores(vals)
        assert all(v == pytest.approx(0.0) for v in z)

    def test_outlier_detection(self):
        # Use values with enough spread so MAD > 0
        vals = [1.0, 2.0, 3.0, 4.0, 100.0]
        z = vt.robust_z_scores(vals)
        # The outlier should have the largest absolute z-score
        assert abs(z[-1]) > abs(z[0])
        assert abs(z[-1]) > 2.0  # clearly an outlier

    def test_symmetric(self):
        vals = [0.0, 10.0]
        z = vt.robust_z_scores(vals)
        assert abs(z[0]) == pytest.approx(abs(z[1]))


# ========================== FFprobe Tests ==========================


class TestFfprobeDurationSeconds:
    @patch.object(vt, "run_cmd")
    def test_parses_float(self, mock_cmd):
        mock_cmd.return_value = "123.456"
        assert vt.ffprobe_duration_seconds("video.mp4") == pytest.approx(123.456)

    @patch.object(vt, "run_cmd")
    def test_integer_duration(self, mock_cmd):
        mock_cmd.return_value = "60"
        assert vt.ffprobe_duration_seconds("v.mp4") == pytest.approx(60.0)


class TestFfprobeCreationTimeLocal:
    @patch.object(vt, "run_cmd")
    def test_utc_creation_time(self, mock_cmd):
        data = {"format": {"tags": {"creation_time": "2024-01-15T10:30:00Z"}}}
        mock_cmd.return_value = json.dumps(data)
        dt = vt.ffprobe_creation_time_local("v.mp4")
        assert dt.tzinfo is not None
        utc_dt = dt.astimezone(timezone.utc)
        assert utc_dt.hour == 10
        assert utc_dt.minute == 30

    @patch.object(vt, "run_cmd")
    def test_offset_creation_time(self, mock_cmd):
        data = {"format": {"tags": {"creation_time": "2024-01-15T10:30:00+02:00"}}}
        mock_cmd.return_value = json.dumps(data)
        dt = vt.ffprobe_creation_time_local("v.mp4")
        assert dt.tzinfo is not None
        utc_dt = dt.astimezone(timezone.utc)
        assert utc_dt.hour == 8
        assert utc_dt.minute == 30

    @patch.object(vt, "run_cmd")
    def test_naive_creation_time(self, mock_cmd):
        data = {"format": {"tags": {"creation_time": "2024-01-15T10:30:00"}}}
        mock_cmd.return_value = json.dumps(data)
        dt = vt.ffprobe_creation_time_local("v.mp4")
        assert dt.hour == 10
        assert dt.minute == 30

    @patch.object(vt, "run_cmd")
    def test_space_separator(self, mock_cmd):
        data = {"format": {"tags": {"creation_time": "2024-01-15 10:30:00Z"}}}
        mock_cmd.return_value = json.dumps(data)
        dt = vt.ffprobe_creation_time_local("v.mp4")
        utc_dt = dt.astimezone(timezone.utc)
        assert utc_dt.hour == 10

    @patch.object(vt, "run_cmd")
    @patch("os.path.getmtime")
    def test_fallback_to_mtime(self, mock_mtime, mock_cmd):
        mock_cmd.side_effect = RuntimeError("no creation_time")
        mock_mtime.return_value = 1705312200.0
        dt = vt.ffprobe_creation_time_local("v.mp4")
        expected = datetime.fromtimestamp(1705312200.0).astimezone()
        assert dt == expected

    @patch.object(vt, "run_cmd")
    @patch("os.path.getmtime")
    def test_fallback_on_missing_tags(self, mock_mtime, mock_cmd):
        mock_cmd.return_value = json.dumps({"format": {}})
        mock_mtime.return_value = 1705312200.0
        dt = vt.ffprobe_creation_time_local("v.mp4")
        expected = datetime.fromtimestamp(1705312200.0).astimezone()
        assert dt == expected


# ========================== File Management Tests ==========================


class TestCopyToLocalWorkdir:
    @patch.object(vt, "ensure_dir")
    @patch("shutil.copy2")
    @patch("os.path.exists")
    def test_copies_when_not_present(self, mock_exists, mock_copy, mock_edir):
        mock_exists.return_value = False
        result = vt.copy_to_local_workdir("/nas/cam/clip.mp4")
        # Now uses hash-prefixed filenames
        assert result.endswith("clip.mp4")
        assert "_" in os.path.basename(result)  # has hash prefix
        mock_copy.assert_called_once()

    @patch.object(vt, "ensure_dir")
    @patch("shutil.copy2")
    @patch("os.path.exists")
    def test_skips_when_already_present(self, mock_exists, mock_copy, mock_edir):
        mock_exists.return_value = True
        result = vt.copy_to_local_workdir("/nas/cam/clip.mp4")
        assert result.endswith("clip.mp4")
        mock_copy.assert_not_called()

    @patch.object(vt, "ensure_dir")
    @patch("shutil.copy2")
    @patch("os.path.exists", return_value=False)
    def test_different_paths_get_different_hashes(self, mock_exists, mock_copy, mock_edir):
        r1 = vt.copy_to_local_workdir("/nas/cam1/clip.mp4")
        r2 = vt.copy_to_local_workdir("/nas/cam2/clip.mp4")
        # Same basename, different source → different local names
        assert os.path.basename(r1) != os.path.basename(r2)


class TestDownscaleToAnalysisCopy:
    @patch.object(vt, "run_cmd")
    @patch.object(vt, "ensure_dir")
    @patch("os.path.exists", return_value=False)
    def test_runs_ffmpeg(self, mock_exists, mock_edir, mock_cmd):
        vt.downscale_to_analysis_copy("src.mp4", "dst.mp4", height=480)
        mock_cmd.assert_called_once()
        args = mock_cmd.call_args[0][0]
        assert "ffmpeg" in args
        assert "scale=-2:480" in args

    @patch("os.path.exists", return_value=True)
    def test_skips_when_exists(self, mock_exists):
        vt.downscale_to_analysis_copy("src.mp4", "dst.mp4")


# ========================== Frame Extraction Tests ==========================


class TestExtractFramesForChunk:
    @patch.object(vt, "run_cmd")
    @patch.object(vt, "ensure_dir")
    def test_extracts_correct_number(self, mock_edir, mock_cmd, tmp_path):
        frame_dir = str(tmp_path / "frames")
        os.makedirs(frame_dir, exist_ok=True)

        # Simulate the batch ffmpeg call creating N frame files
        def create_frames(cmd):
            # The output pattern is the last argument
            pattern = cmd[-1]
            if "%02d" in pattern:
                # Batch mode: create all frame files
                for i in range(4):
                    fp = pattern.replace("%02d", f"{i:02d}")
                    with open(fp, "w") as f:
                        f.write("fake jpg")
            else:
                # Fallback per-frame mode
                with open(cmd[-1], "w") as f:
                    f.write("fake jpg")
            return ""

        mock_cmd.side_effect = create_frames

        paths = vt.extract_frames_for_chunk("video.mp4", 10.0, 20.0, 4, frame_dir)
        assert len(paths) == 4

    @patch.object(vt, "ensure_dir")
    def test_zero_frames(self, mock_edir, tmp_path):
        paths = vt.extract_frames_for_chunk("v.mp4", 0, 10, 0, str(tmp_path))
        assert paths == []


class TestReadFramesAtTimestamps:
    @patch("cv2.VideoCapture")
    def test_reads_multiple_frames(self, mock_cap_cls):
        frames = [
            np.zeros((180, 320, 3), dtype=np.uint8),
            np.full((180, 320, 3), 128, dtype=np.uint8),
        ]
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.side_effect = [(True, frames[0]), (True, frames[1])]
        mock_cap_cls.return_value = mock_cap

        results = vt.read_frames_at_timestamps("v.mp4", [1.0, 5.0])
        assert len(results) == 2
        assert results[0] is not None
        assert results[1] is not None
        # Should only open/close once
        mock_cap.release.assert_called_once()

    @patch("cv2.VideoCapture")
    def test_returns_none_for_unopened(self, mock_cap_cls):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_cap_cls.return_value = mock_cap

        results = vt.read_frames_at_timestamps("v.mp4", [1.0, 2.0])
        assert results == [None, None]


class TestReadFrameForSignals:
    @patch("cv2.VideoCapture")
    def test_reads_frame(self, mock_cap_cls):
        fake_frame = np.zeros((180, 320, 3), dtype=np.uint8)
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, fake_frame)
        mock_cap_cls.return_value = mock_cap

        result = vt.read_frame_for_signals("v.mp4", 5.0)
        assert result is not None
        assert result.shape == (180, 320, 3)
        mock_cap.set.assert_called_once_with(cv2.CAP_PROP_POS_MSEC, 5000.0)
        mock_cap.release.assert_called_once()

    @patch("cv2.VideoCapture")
    def test_returns_none_on_failure(self, mock_cap_cls):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_cap_cls.return_value = mock_cap

        result = vt.read_frame_for_signals("v.mp4", 5.0)
        assert result is None


class TestComputeChunkSignals:
    @patch.object(vt, "read_frames_at_timestamps")
    def test_returns_all_keys(self, mock_read):
        frame = np.full((180, 320, 3), 128, dtype=np.uint8)
        mock_read.return_value = [frame, frame, frame]

        signals = vt.compute_chunk_signals("v.mp4", 0.0, 10.0, sample_points=3)
        expected_keys = {"motion_score", "luma_mean", "luma_std", "ir_mode", "sat_mean"}
        assert set(signals.keys()) == expected_keys

    @patch.object(vt, "read_frames_at_timestamps")
    def test_low_motion_identical_frames(self, mock_read):
        frame = np.full((180, 320, 3), 128, dtype=np.uint8)
        mock_read.return_value = [frame, frame, frame]

        signals = vt.compute_chunk_signals("v.mp4", 0.0, 10.0, sample_points=3)
        assert signals["motion_score"] == pytest.approx(0.0, abs=0.01)

    @patch.object(vt, "read_frames_at_timestamps")
    def test_high_motion_different_frames(self, mock_read):
        mock_read.return_value = [
            np.zeros((180, 320, 3), dtype=np.uint8),
            np.full((180, 320, 3), 255, dtype=np.uint8),
            np.zeros((180, 320, 3), dtype=np.uint8),
        ]

        signals = vt.compute_chunk_signals("v.mp4", 0.0, 10.0, sample_points=3)
        assert signals["motion_score"] > 0.5

    @patch.object(vt, "read_frames_at_timestamps")
    def test_fallback_when_no_frames(self, mock_read):
        mock_read.return_value = [None, None, None]
        signals = vt.compute_chunk_signals("v.mp4", 0.0, 10.0, sample_points=3)
        assert signals["motion_score"] == pytest.approx(0.0)
        assert signals["ir_mode"] == "unknown"

    @patch.object(vt, "read_frames_at_timestamps")
    def test_ir_mode_detection(self, mock_read):
        gray_frame = np.full((180, 320, 3), 100, dtype=np.uint8)
        mock_read.return_value = [gray_frame, gray_frame, gray_frame]

        signals = vt.compute_chunk_signals("v.mp4", 0.0, 10.0, sample_points=3)
        assert signals["ir_mode"] == "ir"

    @patch.object(vt, "read_frames_at_timestamps")
    def test_day_mode_detection(self, mock_read):
        color_frame = np.zeros((180, 320, 3), dtype=np.uint8)
        color_frame[:, :, 2] = 255  # Red channel full
        mock_read.return_value = [color_frame, color_frame, color_frame]

        signals = vt.compute_chunk_signals("v.mp4", 0.0, 10.0, sample_points=3)
        assert signals["ir_mode"] == "day"


# ========================== Skim / Detection Tests ==========================


class TestSampleFramesForSkim:
    @patch.object(vt, "read_frames_at_timestamps")
    def test_returns_requested_count(self, mock_read):
        frame = np.zeros((180, 320, 3), dtype=np.uint8)
        mock_read.return_value = [frame, frame, frame]
        frames = vt.sample_frames_for_skim("v.mp4", 0.0, 10.0, sample_points=3)
        assert len(frames) == 3

    @patch.object(vt, "read_frames_at_timestamps")
    def test_handles_failures(self, mock_read):
        mock_read.return_value = [
            np.zeros((180, 320, 3), dtype=np.uint8),
            None,
            np.zeros((180, 320, 3), dtype=np.uint8),
        ]
        frames = vt.sample_frames_for_skim("v.mp4", 0.0, 10.0, sample_points=3)
        assert len(frames) == 2


class TestDetectFacesInFrames:
    def test_empty_list(self):
        assert vt.detect_faces_in_frames([]) == 0

    @patch.object(vt, "_get_haar_cascade")
    def test_detects_faces(self, mock_get_cascade):
        mock_cascade = MagicMock()
        mock_cascade.detectMultiScale.return_value = np.array([[10, 10, 50, 50], [100, 100, 50, 50]])
        mock_get_cascade.return_value = mock_cascade

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        count = vt.detect_faces_in_frames([frame])
        assert count == 2

    @patch.object(vt, "_get_haar_cascade")
    def test_no_faces(self, mock_get_cascade):
        mock_cascade = MagicMock()
        mock_cascade.detectMultiScale.return_value = ()
        mock_get_cascade.return_value = mock_cascade

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        count = vt.detect_faces_in_frames([frame])
        assert count == 0

    @patch.object(vt, "_get_haar_cascade")
    def test_returns_zero_when_no_cascade(self, mock_get_cascade):
        mock_get_cascade.return_value = None
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        count = vt.detect_faces_in_frames([frame])
        assert count == 0


class TestChooseFramesPerChunk:
    def test_static_mode(self):
        assert vt.choose_frames_per_chunk(6, 0.1, dynamic=False) == 6

    def test_dynamic_low_motion(self):
        result = vt.choose_frames_per_chunk(
            6, 0.005, dynamic=True, motion_low=0.01, motion_high=0.05,
            min_frames=2, max_frames=6
        )
        assert result == 2

    def test_dynamic_high_motion(self):
        result = vt.choose_frames_per_chunk(
            6, 0.10, dynamic=True, motion_low=0.01, motion_high=0.05,
            min_frames=2, max_frames=6
        )
        assert result == 6

    def test_dynamic_mid_motion(self):
        result = vt.choose_frames_per_chunk(
            6, 0.03, dynamic=True, motion_low=0.01, motion_high=0.05,
            min_frames=2, max_frames=6
        )
        assert 2 <= result <= 6


# ========================== Audio Tests ==========================


class TestAudioHasSpeech:
    @patch.object(vt, "run_cmd")
    def test_speech_above_threshold(self, mock_cmd):
        import wave
        import struct

        def fake_ffmpeg(cmd):
            wav_path = cmd[-1]
            os.makedirs(os.path.dirname(wav_path), exist_ok=True)
            with wave.open(wav_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                samples = [20000] * 16000
                wf.writeframes(struct.pack(f"{len(samples)}h", *samples))
            return ""

        mock_cmd.side_effect = fake_ffmpeg
        result = vt.audio_has_speech("v.mp4", 0.0, 1.0, rms_threshold=0.01)
        assert result is True

    @patch.object(vt, "run_cmd")
    def test_silence_below_threshold(self, mock_cmd):
        import wave
        import struct

        def fake_ffmpeg(cmd):
            wav_path = cmd[-1]
            os.makedirs(os.path.dirname(wav_path), exist_ok=True)
            with wave.open(wav_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                samples = [0] * 16000
                wf.writeframes(struct.pack(f"{len(samples)}h", *samples))
            return ""

        mock_cmd.side_effect = fake_ffmpeg
        result = vt.audio_has_speech("v.mp4", 0.0, 1.0, rms_threshold=0.01)
        assert result is False

    @patch.object(vt, "run_cmd")
    def test_returns_false_on_error(self, mock_cmd):
        """After fix: returns False on exception (previously returned True)."""
        mock_cmd.side_effect = RuntimeError("ffmpeg broken")
        result = vt.audio_has_speech("v.mp4", 0.0, 1.0)
        assert result is False


# ========================== PipelineConfig Tests ==========================


class TestPipelineConfig:
    def test_defaults(self):
        cfg = vt.PipelineConfig()
        assert cfg.chunk_s == 20
        assert cfg.frames_per_chunk == 6
        assert cfg.analysis_height == 720
        assert cfg.skim is False
        assert cfg.with_audio is False
        assert cfg.language == "hu"

    def test_custom_values(self):
        cfg = vt.PipelineConfig(chunk_s=30, skim=True, language="en")
        assert cfg.chunk_s == 30
        assert cfg.skim is True
        assert cfg.language == "en"


# ========================== AI Captioning Tests ==========================


class TestExtractJson:
    method = staticmethod(vt.VisualCaptionerQwen25VL._extract_json)

    def test_valid_json(self):
        text = 'Some preamble {"caption": "A person sits down.", "tags": "person_present, sat_down"} trailing'
        caption, tags = self.method(text)
        assert caption == "A person sits down."
        assert tags == "person_present, sat_down"

    def test_json_with_extra_keys(self):
        text = '{"caption": "Cat on chair", "tags": "animal", "extra": 123}'
        caption, tags = self.method(text)
        assert caption == "Cat on chair"
        assert tags == "animal"

    def test_malformed_json(self):
        text = '{"caption": broken json'
        caption, tags = self.method(text)
        assert caption == text.strip()
        assert tags == ""

    def test_no_json_at_all(self):
        text = "Just a plain text response with no JSON"
        caption, tags = self.method(text)
        assert caption == text.strip()
        assert tags == ""

    def test_empty_caption_falls_back(self):
        text = '{"caption": "", "tags": "some_tag"}'
        caption, _ = self.method(text)
        assert caption == text.strip()

    def test_multiline_json(self):
        text = '''Here is the result:
{
  "caption": "A dog walks across the room.",
  "tags": "animal, motion"
}
End of response.'''
        caption, tags = self.method(text)
        assert caption == "A dog walks across the room."
        assert tags == "animal, motion"


# ========================== Decomposed Helpers Tests ==========================


class TestComputeAnomalyScores:
    def test_adds_columns(self):
        import pandas as pd
        df = pd.DataFrame({
            "motion_score": [0.01, 0.02, 0.1],
            "tags_visual": ["person_present", "", "person_present, entered_room"],
        })
        result = vt.compute_anomaly_scores(df)
        assert "anomaly_z_motion" in result.columns
        assert "anomaly_score_motion" in result.columns
        assert "person_present_tag" in result.columns
        assert "anomaly_score_combined" in result.columns
        assert len(result) == 3

    def test_empty_df(self):
        import pandas as pd
        df = pd.DataFrame({"motion_score": [], "tags_visual": []})
        result = vt.compute_anomaly_scores(df)
        assert "anomaly_score_combined" in result.columns


class TestResumeSupport:
    def test_save_and_load_progress(self, tmp_path):
        progress_path = str(tmp_path / ".test.progress.jsonl")

        row1 = {"_chunk_idx": 0, "caption": "Hello"}
        row2 = {"_chunk_idx": 1, "caption": "World"}
        vt._save_chunk_row(progress_path, row1)
        vt._save_chunk_row(progress_path, row2)

        loaded = vt._load_progress(progress_path)
        assert len(loaded) == 2
        assert loaded[0]["_chunk_idx"] == 0
        assert loaded[1]["caption"] == "World"

    def test_load_nonexistent(self, tmp_path):
        rows = vt._load_progress(str(tmp_path / "does_not_exist.jsonl"))
        assert rows == []


# ========================== Integration: process_video ==========================


class TestProcessVideo:
    """Integration-level test with all externals mocked."""

    @patch.object(vt, "compute_chunk_signals")
    @patch.object(vt, "extract_frames_for_chunk")
    @patch.object(vt, "prepare_analysis_copies")
    @patch.object(vt, "ffprobe_duration_seconds")
    @patch.object(vt, "ffprobe_creation_time_local")
    @patch.object(vt, "ensure_dir")
    @patch("tempfile.mkdtemp")
    @patch("shutil.rmtree")
    def test_produces_correct_csv_schema(
        self, mock_rmtree, mock_mkdtemp, mock_edir,
        mock_creation, mock_dur, mock_prepare,
        mock_extract, mock_signals, tmp_path
    ):
        out_dir = str(tmp_path / "output")
        os.makedirs(out_dir, exist_ok=True)
        mock_mkdtemp.return_value = str(tmp_path / "tempframes")
        os.makedirs(str(tmp_path / "tempframes"), exist_ok=True)

        # Video is 45 seconds → 3 chunks (20, 20, 5)
        mock_dur.return_value = 45.0
        mock_creation.return_value = datetime(2024, 6, 15, 14, 0, 0)
        mock_prepare.return_value = ("analysis.mp4", "hq.mp4")

        mock_signals.return_value = {
            "motion_score": 0.03,
            "luma_mean": 0.5,
            "luma_std": 0.1,
            "ir_mode": "day",
            "sat_mean": 0.25,
        }
        mock_extract.return_value = ["frame_00.jpg", "frame_01.jpg"]

        mock_captioner = MagicMock()
        mock_captioner.caption_frames.return_value = vt.CaptionResult(
            caption="A person enters the room.", tags="person_present, entered_room"
        )

        df = vt.process_video(
            video_path="test_clip.mp4",
            out_dir=out_dir,
            captioner=mock_captioner,
            chunk_s=20,
            frames_per_chunk=3,
            skip_if_done=False,
        )

        assert len(df) == 3

        expected_cols = {
            "abs_start", "abs_end", "clip_file", "clip_start_media_time",
            "offset_start_s", "offset_end_s",
            "motion_score", "luma_mean", "luma_std", "sat_mean", "ir_mode",
            "skim_triggered", "faces_detected",
            "caption_visual", "tags_visual", "transcript_audio",
            "anomaly_z_motion", "anomaly_score_motion",
            "person_present_tag", "anomaly_score_combined",
        }
        assert expected_cols.issubset(set(df.columns))

        assert all(df["clip_file"] == "test_clip.mp4")
        assert df.iloc[0]["offset_start_s"] == pytest.approx(0.0)
        assert df.iloc[1]["offset_start_s"] == pytest.approx(20.0)
        assert df.iloc[2]["offset_start_s"] == pytest.approx(40.0)
        assert all(df["caption_visual"] == "A person enters the room.")

        csv_path = os.path.join(out_dir, "test_clip.timeline.csv")
        assert os.path.exists(csv_path)

    @patch.object(vt, "compute_chunk_signals")
    @patch.object(vt, "extract_frames_for_chunk")
    @patch.object(vt, "prepare_analysis_copies")
    @patch.object(vt, "ffprobe_duration_seconds")
    @patch.object(vt, "ffprobe_creation_time_local")
    @patch.object(vt, "ensure_dir")
    @patch("tempfile.mkdtemp")
    @patch("shutil.rmtree")
    def test_skip_when_csv_exists(
        self, mock_rmtree, mock_mkdtemp, mock_edir,
        mock_creation, mock_dur, mock_prepare,
        mock_extract, mock_signals, tmp_path
    ):
        out_dir = str(tmp_path / "output")
        os.makedirs(out_dir, exist_ok=True)

        import pandas as pd
        csv_path = os.path.join(out_dir, "test_clip.timeline.csv")
        pd.DataFrame({"col": [1]}).to_csv(csv_path, index=False)

        mock_captioner = MagicMock()

        df = vt.process_video(
            video_path="test_clip.mp4",
            out_dir=out_dir,
            captioner=mock_captioner,
            skip_if_done=True,
        )

        mock_dur.assert_not_called()
        mock_captioner.caption_frames.assert_not_called()


# ========================== CLI Argument Parsing ==========================


class TestCLIParsing:
    def _parse(self, args_list):
        import argparse
        ap = argparse.ArgumentParser()
        ap.add_argument("--in", dest="in_dir", required=True)
        ap.add_argument("--out", dest="out_dir", required=True)
        ap.add_argument("--chunk", type=int, default=40)
        ap.add_argument("--frames", type=int, default=3)
        ap.add_argument("--height", type=int, default=480)
        ap.add_argument("--hq-height", type=int, default=720)
        ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
        ap.add_argument("--audio", action="store_true")
        ap.add_argument("--whisper_model", type=str, default="small")
        ap.add_argument("--language", type=str, default="hu", choices=["hu", "en"])
        ap.add_argument("--skim", action="store_true")
        ap.add_argument("--skim-motion", type=float, default=0.02)
        ap.add_argument("--skim-faces", action="store_true")
        ap.add_argument("--audio-faces-only", action="store_true")
        ap.add_argument("--audio-vad", action="store_true")
        ap.add_argument("--audio-vad-threshold", type=float, default=0.01)
        ap.add_argument("--dynamic-frames", action="store_true")
        ap.add_argument("--frames-min", type=int, default=2)
        ap.add_argument("--frames-max", type=int, default=6)
        ap.add_argument("--motion-low", type=float, default=0.01)
        ap.add_argument("--motion-high", type=float, default=0.05)
        ap.add_argument("--no-skip", action="store_true")
        ap.add_argument("--dry-run", action="store_true")
        return ap.parse_args(args_list)

    def test_defaults(self):
        args = self._parse(["--in", "/nas/cam", "--out", "./out"])
        assert args.in_dir == "/nas/cam"
        assert args.out_dir == "./out"
        assert args.chunk == 40
        assert args.frames == 3
        assert args.height == 480
        assert args.audio is False
        assert args.skim is False
        assert args.language == "hu"
        assert args.dry_run is False

    def test_all_flags(self):
        args = self._parse([
            "--in", "/nas", "--out", "./o",
            "--chunk", "30", "--frames", "8",
            "--audio", "--skim", "--skim-faces",
            "--language", "en",
            "--dynamic-frames", "--no-skip",
            "--audio-vad", "--audio-faces-only",
            "--dry-run",
        ])
        assert args.chunk == 30
        assert args.frames == 8
        assert args.audio is True
        assert args.skim is True
        assert args.skim_faces is True
        assert args.language == "en"
        assert args.dynamic_frames is True
        assert args.no_skip is True
        assert args.audio_vad is True
        assert args.audio_faces_only is True
        assert args.dry_run is True
