"""Helpers for extracting visual context from local video files."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tools.vision_tools import vision_analyze_tool

logger = logging.getLogger(__name__)


_DEFAULT_MAX_FRAMES = 3
_DEFAULT_FPS = "1/3"
_DEFAULT_TIMEOUT_SECONDS = 25
_DEFAULT_MAX_CHARS_PER_FRAME = 320


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(float(raw))
        return value if value >= minimum else default
    except ValueError:
        return default


def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name, "").strip()
    return raw or default


def _clip_text(text: str, max_chars: int) -> str:
    compact = " ".join(str(text or "").split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3].rstrip() + "..."


def _extract_frames_with_ffmpeg(
    *,
    video_path: Path,
    output_dir: Path,
    max_frames: int,
    fps: str,
    timeout_seconds: int,
) -> Tuple[List[Path], Optional[str], Optional[str]]:
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        return [], "ffmpeg not found", "ffmpeg_missing"

    frame_pattern = output_dir / "frame_%02d.jpg"
    command = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"fps={fps}",
        "-frames:v",
        str(max_frames),
        str(frame_pattern),
    ]

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return [], f"ffmpeg timed out after {timeout_seconds}s", "ffmpeg_timeout"
    except Exception as exc:
        return [], f"ffmpeg invocation failed ({exc})", "ffmpeg_error"

    if completed.returncode != 0:
        details = (completed.stderr or completed.stdout or "").strip()
        details = details.splitlines()[-1] if details else "unknown ffmpeg error"
        return [], f"ffmpeg failed ({details})", "ffmpeg_failed"

    frames = sorted(output_dir.glob("frame_*.jpg"))
    if frames:
        return frames, None, None

    return [], "ffmpeg produced no frames", "no_frames"


async def summarize_video_visual_context(
    video_path: str,
    *,
    max_frames: Optional[int] = None,
    fps: Optional[str] = None,
    timeout_seconds: Optional[int] = None,
    max_chars_per_frame: Optional[int] = None,
) -> Dict[str, Any]:
    """Sample a few frames and summarize visible video context via vision models."""
    resolved_path = Path(os.path.expanduser(str(video_path or "").strip()))
    if not resolved_path.is_file():
        return {
            "success": False,
            "error": f"video file not found: {resolved_path}",
            "code": "file_not_found",
        }

    frame_limit = max_frames or _env_int("HERMES_VIDEO_VISION_MAX_FRAMES", _DEFAULT_MAX_FRAMES)
    frame_limit = max(1, frame_limit)

    sample_fps = fps or _env_str("HERMES_VIDEO_VISION_SAMPLE_FPS", _DEFAULT_FPS)
    ffmpeg_timeout = timeout_seconds or _env_int(
        "HERMES_VIDEO_VISION_TIMEOUT_SECONDS",
        _DEFAULT_TIMEOUT_SECONDS,
    )
    char_limit = max_chars_per_frame or _env_int(
        "HERMES_VIDEO_VISION_MAX_CHARS_PER_FRAME",
        _DEFAULT_MAX_CHARS_PER_FRAME,
        minimum=80,
    )

    tmp_dir = Path(tempfile.mkdtemp(prefix="hermes_video_frames_"))
    try:
        frames, frame_error, frame_error_code = _extract_frames_with_ffmpeg(
            video_path=resolved_path,
            output_dir=tmp_dir,
            max_frames=frame_limit,
            fps=sample_fps,
            timeout_seconds=ffmpeg_timeout,
        )
        if not frames:
            return {
                "success": False,
                "error": frame_error or "failed to extract video frames",
                "code": frame_error_code or "frame_extraction_failed",
            }

        frame_prompt = (
            "Describe this sampled video frame. Focus on scene, subjects, actions, "
            "visible text/UI, and notable changes. Avoid guessing details not visible."
        )

        frame_notes: List[str] = []
        for index, frame_path in enumerate(frames, start=1):
            try:
                result_json = await vision_analyze_tool(
                    image_url=str(frame_path),
                    user_prompt=frame_prompt,
                )
                result = json.loads(result_json)
                if result.get("success"):
                    analysis = _clip_text(result.get("analysis", ""), char_limit)
                    if analysis:
                        frame_notes.append(f"Frame {index}: {analysis}")
                    continue

                error = _clip_text(result.get("error", "analysis failed"), 120)
                frame_notes.append(f"Frame {index}: analysis unavailable ({error})")
            except Exception as exc:
                frame_notes.append(
                    f"Frame {index}: analysis unavailable ({_clip_text(str(exc), 120)})"
                )

        if not frame_notes:
            return {
                "success": False,
                "error": "vision analysis returned no usable frame descriptions",
                "code": "vision_empty",
            }

        summary = "\n".join(f"- {note}" for note in frame_notes)
        return {
            "success": True,
            "summary": summary,
            "sampled_frames": len(frames),
            "video_path": str(resolved_path),
        }
    finally:
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            logger.debug("Failed to clean temporary video frame directory: %s", tmp_dir)
