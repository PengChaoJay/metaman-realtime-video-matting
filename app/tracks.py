from __future__ import annotations

import asyncio
import logging

from aiortc import MediaStreamTrack
from av import VideoFrame

from .remover import BackgroundRemover, pack_rgb_and_alpha

logger = logging.getLogger(__name__)


class BackgroundRemovalTrack(MediaStreamTrack):
    """Transforms an incoming WebRTC video track into packed RGB + alpha."""

    kind = "video"

    def __init__(self, source: MediaStreamTrack, remover: BackgroundRemover) -> None:
        super().__init__()
        self._source = source
        self._remover = remover

    async def recv(self) -> VideoFrame:
        frame = await self._source.recv()
        rgb = frame.to_ndarray(format="rgb24")

        # Model inference is CPU-heavy. Keeping it off the asyncio event loop
        # prevents signaling and WebRTC housekeeping from stalling.
        alpha = await asyncio.to_thread(self._remover.mask, rgb)
        refine_rgb = getattr(self._remover, "refine_rgb", None)
        if callable(refine_rgb):
            rgb = refine_rgb(rgb, alpha)
        packed = pack_rgb_and_alpha(rgb, alpha)

        output = VideoFrame.from_ndarray(packed, format="rgb24")
        output.pts = frame.pts
        output.time_base = frame.time_base
        return output

    def stop(self) -> None:
        try:
            self._remover.close()
        finally:
            self._source.stop()
            super().stop()
            logger.debug("background removal track stopped")

