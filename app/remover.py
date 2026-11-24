from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import threading
from typing import Protocol

import cv2
import numpy as np


class BackgroundRemover(Protocol):
    """Produces an 8-bit alpha mask for an RGB frame."""

    def mask(self, rgb: np.ndarray) -> np.ndarray:
        ...

    def close(self) -> None:
        ...


@dataclass(frozen=True)
class MaskConfig:
    threshold: float = 0.50
    feather: float = 0.20
    temporal_smoothing: float = 0.65
    blur_size: int = 5
    edge_shrink: int = 0


@dataclass(frozen=True)
class GreenScreenConfig:
    dominance_threshold: float = 10.0
    feather: float = 34.0
    minimum_green: float = 30.0
    temporal_smoothing: float = 0.0
    blur_size: int = 3
    despill_strength: float = 0.85


def _box_blur(values: np.ndarray, size: int) -> np.ndarray:
    size = max(1, int(size))
    if size % 2 == 0:
        size += 1
    if size == 1:
        return values
    pad = size // 2
    padded = np.pad(values, ((pad, pad), (pad, pad)), mode="edge")
    integral = np.pad(padded, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    return (
        integral[size:, size:]
        - integral[:-size, size:]
        - integral[size:, :-size]
        + integral[:-size, :-size]
    ) / float(size * size)


def probability_to_alpha(
    probability: np.ndarray,
    config: MaskConfig,
    previous: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a model probability map into a stable, soft uint8 alpha mask."""
    probability = np.asarray(probability, dtype=np.float32)
    probability = np.clip(probability, 0.0, 1.0)

    feather = max(config.feather, 1e-3)
    low = np.clip(config.threshold - feather / 2.0, 0.0, 1.0)
    high = np.clip(config.threshold + feather / 2.0, low + 1e-3, 1.0)
    soft = np.clip((probability - low) / (high - low), 0.0, 1.0)
    soft = soft * soft * (3.0 - 2.0 * soft)

    shrink_size = max(0, int(config.edge_shrink))
    if shrink_size > 1:
        if shrink_size % 2 == 0:
            shrink_size += 1
        kernel = np.ones((shrink_size, shrink_size), dtype=np.uint8)
        soft = cv2.erode(soft, kernel, borderType=cv2.BORDER_REPLICATE)

    blur_size = max(1, int(config.blur_size))
    if blur_size % 2 == 0:
        blur_size += 1
    if blur_size > 1:
        soft = _box_blur(soft, blur_size)

    smoothing = np.clip(config.temporal_smoothing, 0.0, 0.95)
    if previous is not None and previous.shape == soft.shape:
        soft = smoothing * previous + (1.0 - smoothing) * soft

    soft = np.clip(soft, 0.0, 1.0)
    return np.rint(soft * 255.0).astype(np.uint8), soft


class MediaPipeSelfieRemover:
    """MediaPipe selfie segmentation with edge and temporal smoothing."""

    def __init__(self, config: MaskConfig | None = None) -> None:
        try:
            import mediapipe as mp
        except ImportError as exc:  # pragma: no cover - depends on runtime image
            raise RuntimeError(
                "MediaPipe or one of its runtime dependencies is missing; "
                "follow the installation commands in README.md"
            ) from exc

        self._config = config or MaskConfig()
        self._segmenter = mp.solutions.selfie_segmentation.SelfieSegmentation(
            model_selection=1
        )
        self._previous: np.ndarray | None = None

    def mask(self, rgb: np.ndarray) -> np.ndarray:
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("expected an HxWx3 RGB frame")

        result = self._segmenter.process(rgb)
        if result.segmentation_mask is None:
            return np.zeros(rgb.shape[:2], dtype=np.uint8)

        alpha, self._previous = probability_to_alpha(
            result.segmentation_mask, self._config, self._previous
        )
        return alpha

    def close(self) -> None:
        self._segmenter.close()


class GreenScreenRemover:
    """Deterministic chroma key with soft edges and green-spill suppression."""

    def __init__(self, config: GreenScreenConfig | None = None) -> None:
        self._config = config or GreenScreenConfig()
        self._previous: np.ndarray | None = None

    def mask(self, rgb: np.ndarray) -> np.ndarray:
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("expected an HxWx3 RGB frame")

        pixels = rgb.astype(np.float32)
        red, green, blue = pixels[..., 0], pixels[..., 1], pixels[..., 2]
        dominance = green - np.maximum(red, blue)
        feather = max(self._config.feather, 1.0)
        key = np.clip(
            (dominance - self._config.dominance_threshold) / feather,
            0.0,
            1.0,
        )
        brightness_gate = np.clip(
            (green - self._config.minimum_green) / 32.0,
            0.0,
            1.0,
        )
        key *= brightness_gate
        key = key * key * (3.0 - 2.0 * key)
        key = _box_blur(key, self._config.blur_size)
        alpha = np.clip(1.0 - key, 0.0, 1.0)

        smoothing = np.clip(self._config.temporal_smoothing, 0.0, 0.95)
        if self._previous is not None and self._previous.shape == alpha.shape:
            # Only smooth newly appearing foreground. When a moving subject leaves
            # a pixel, keeping the previous alpha turns the current (despilled)
            # green background grey and creates a visible motion trail.
            smoothed = smoothing * self._previous + (1.0 - smoothing) * alpha
            alpha = np.where(alpha < self._previous, alpha, smoothed)
        self._previous = alpha
        return np.rint(alpha * 255.0).astype(np.uint8)

    def refine_rgb(self, rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """Remove reflected green from semi-transparent subject edges."""
        pixels = rgb.astype(np.float32)
        red, green, blue = pixels[..., 0], pixels[..., 1], pixels[..., 2]
        spill = np.maximum(green - np.maximum(red, blue), 0.0)
        transparency = 1.0 - alpha.astype(np.float32) / 255.0
        edge_weight = np.clip(1.8 * transparency, 0.0, 1.0)
        chroma_weight = np.clip((spill + 8.0) / 32.0, 0.0, 1.0)
        despill_weight = np.maximum(
            edge_weight,
            chroma_weight * np.clip(self._config.despill_strength, 0.0, 1.0),
        )
        pixels[..., 1] = green - spill * despill_weight
        return np.clip(pixels, 0.0, 255.0).astype(np.uint8)

    def close(self) -> None:
        self._previous = None


class RVMRemover:
    """RVM MobileNetV3 video matting with per-stream recurrent state."""

    _sessions: dict[str, object] = {}
    _session_lock = threading.Lock()

    def __init__(
        self,
        edge_strength: float = 0.65,
        model_path: str | Path | None = None,
        require_cuda: bool = True,
    ) -> None:
        configured_path = model_path or os.getenv(
            "RVM_MODEL_PATH", "/app/models/rvm_mobilenetv3_fp16.onnx"
        )
        self._model_path = str(configured_path)
        self._edge_strength = max(0.0, min(1.0, edge_strength))
        self._session = self._get_session(self._model_path, require_cuda)
        self.backend = "RVM MobileNetV3 · CUDA"
        self._dtype = np.float16
        self._rec = [np.zeros((1, 1, 1, 1), dtype=self._dtype) for _ in range(4)]
        self._foreground: np.ndarray | None = None

    @classmethod
    def _get_session(cls, model_path: str, require_cuda: bool):
        cache_key = f"{model_path}:{require_cuda}"
        with cls._session_lock:
            if cache_key in cls._sessions:
                return cls._sessions[cache_key]

            path = Path(model_path)
            if not path.is_file():
                raise RuntimeError(f"RVM model not found: {path}")

            try:
                import onnxruntime as ort
            except ImportError as exc:
                raise RuntimeError("onnxruntime-gpu is not installed") from exc

            if hasattr(ort, "preload_dlls"):
                ort.preload_dlls(directory="")
            available = ort.get_available_providers()
            if require_cuda and "CUDAExecutionProvider" not in available:
                raise RuntimeError(
                    f"CUDAExecutionProvider unavailable; providers={available}"
                )
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if not require_cuda:
                providers = [p for p in providers if p in available]
            session = ort.InferenceSession(str(path), providers=providers)
            active = session.get_providers()
            if require_cuda and (
                not active or active[0] != "CUDAExecutionProvider"
            ):
                raise RuntimeError(f"RVM did not activate CUDA; providers={active}")
            cls._sessions[cache_key] = session
            return session

    @staticmethod
    def _downsample_ratio(height: int, width: int) -> float:
        override = os.getenv("RVM_DOWNSAMPLE_RATIO")
        if override:
            return max(0.05, min(1.0, float(override)))
        longest = max(height, width)
        if longest <= 512:
            return 1.0
        if longest <= 1280:
            return 0.6
        if longest <= 1920:
            return 0.4
        return 0.25

    def mask(self, rgb: np.ndarray) -> np.ndarray:
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("expected an HxWx3 RGB frame")

        src = np.ascontiguousarray(
            rgb.transpose(2, 0, 1)[np.newaxis].astype(self._dtype) / 255.0
        )
        ratio = np.asarray(
            [self._downsample_ratio(rgb.shape[0], rgb.shape[1])],
            dtype=np.float32,
        )
        fgr, pha, *self._rec = self._session.run(
            None,
            {
                "src": src,
                "r1i": self._rec[0],
                "r2i": self._rec[1],
                "r3i": self._rec[2],
                "r4i": self._rec[3],
                "downsample_ratio": ratio,
            },
        )
        self._foreground = np.clip(
            fgr[0].transpose(1, 2, 0).astype(np.float32) * 255.0,
            0.0,
            255.0,
        ).astype(np.uint8)

        alpha = np.clip(pha[0, 0].astype(np.float32), 0.0, 1.0)
        cutoff = 0.02 + 0.10 * self._edge_strength
        alpha = np.clip((alpha - cutoff) / (1.0 - cutoff), 0.0, 1.0)
        return np.rint(alpha * 255.0).astype(np.uint8)

    def refine_rgb(self, rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        if self._foreground is not None and self._foreground.shape == rgb.shape:
            return self._foreground
        return rgb

    def close(self) -> None:
        self._rec = [np.zeros((1, 1, 1, 1), dtype=self._dtype) for _ in range(4)]
        self._foreground = None


def pack_rgb_and_alpha(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Pack color and alpha side-by-side into one RGB frame.

    The left half is the original RGB image. The right half repeats the alpha
    value into R/G/B so it survives ordinary VP8/H.264 video transport.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("expected an HxWx3 RGB frame")
    if alpha.ndim != 2 or alpha.shape != rgb.shape[:2]:
        raise ValueError("alpha dimensions must match the RGB frame")

    alpha_rgb = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)
    return np.ascontiguousarray(np.concatenate((rgb, alpha_rgb), axis=1))

