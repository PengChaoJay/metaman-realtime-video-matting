import asyncio
import unittest
from fractions import Fraction

import numpy as np
from aiortc import RTCPeerConnection, VideoStreamTrack
from av import VideoFrame

from app.main import Offer, offer, shutdown


class SyntheticVideoTrack(VideoStreamTrack):
    async def recv(self) -> VideoFrame:
        pts, time_base = await self.next_timestamp()
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        image[8:56, 20:44] = (220, 180, 140)
        frame = VideoFrame.from_ndarray(image, format="rgb24")
        frame.pts = pts
        frame.time_base = time_base or Fraction(1, 90000)
        return frame


class WebRTCTests(unittest.IsolatedAsyncioTestCase):
    async def test_frame_round_trip_returns_packed_rgb_and_alpha(self) -> None:
        client = RTCPeerConnection()
        received: asyncio.Future = asyncio.get_running_loop().create_future()

        @client.on("track")
        def on_track(track) -> None:
            if not received.done():
                received.set_result(track)

        client.addTrack(SyntheticVideoTrack())
        await client.setLocalDescription(await client.createOffer())

        answer = await offer(
            Offer(
                sdp=client.localDescription.sdp,
                type=client.localDescription.type,
            )
        )
        await client.setRemoteDescription(
            type(client.localDescription)(sdp=answer["sdp"], type=answer["type"])
        )

        try:
            output_track = await asyncio.wait_for(received, timeout=10)
            frame = await asyncio.wait_for(output_track.recv(), timeout=20)
            self.assertEqual(frame.width, 128)
            self.assertEqual(frame.height, 64)
        finally:
            await client.close()
            await shutdown()

