from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

from aiortc import RTCPeerConnection, RTCSessionDescription
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .remover import (
    GreenScreenConfig,
    GreenScreenRemover,
    MaskConfig,
    MediaPipeSelfieRemover,
    RVMRemover,
)
from .frame_processor import jpeg_to_transparent_png
from .tracks import BackgroundRemovalTrack

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("background-streamer")

ROOT = Path(__file__).resolve().parents[1]
INDEX_HTML = ROOT / "static" / "index.html"
PEER_CONNECTIONS: set[RTCPeerConnection] = set()

app = FastAPI(title="Real-time Background Removal", version="0.1.0")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


class Offer(BaseModel):
    sdp: str
    type: str


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        logger.warning("invalid %s; using %s", name, default)
        return default


def create_remover(
    mode: str = "person",
    green_strength: float = 0.78,
    person_strength: float = 0.65,
):
    if mode == "green":
        strength = max(0.0, min(1.0, green_strength))
        return GreenScreenRemover(
            GreenScreenConfig(
                dominance_threshold=22.0 - 18.0 * strength,
                feather=44.0 - 14.0 * strength,
                minimum_green=45.0 - 20.0 * strength,
                despill_strength=0.55 + 0.45 * strength,
            )
        )
    edge_strength = max(0.0, min(1.0, person_strength))
    if os.getenv("PERSON_MODEL", "mediapipe").lower() == "rvm":
        try:
            return RVMRemover(
                edge_strength=edge_strength,
                require_cuda=os.getenv("RVM_REQUIRE_CUDA", "1") != "0",
            )
        except Exception:
            if os.getenv("RVM_ALLOW_FALLBACK", "1") == "0":
                raise
            logger.exception("RVM unavailable; falling back to MediaPipe")
    return MediaPipeSelfieRemover(
        MaskConfig(
            threshold=env_float("MASK_THRESHOLD", 0.50 + 0.20 * edge_strength),
            feather=env_float("MASK_FEATHER", 0.18 - 0.10 * edge_strength),
            temporal_smoothing=env_float(
                "MASK_TEMPORAL_SMOOTHING", 0.35 - 0.25 * edge_strength
            ),
            blur_size=int(env_float("MASK_BLUR_SIZE", 3)),
            edge_shrink=int(
                env_float("MASK_EDGE_SHRINK", 1 + 2 * round(3 * edge_strength))
            ),
        )
    )


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(INDEX_HTML)


@app.get("/healthz")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "person_model": "rvm-tfjs-webgl / mediapipe-server",
        "active_peer_connections": len(PEER_CONNECTIONS),
    }


@app.post("/offer")
async def offer(payload: Offer) -> dict[str, str]:
    if payload.type != "offer":
        raise HTTPException(status_code=400, detail="type must be 'offer'")

    pc = RTCPeerConnection()
    PEER_CONNECTIONS.add(pc)
    peer_name = f"pc-{id(pc)}"
    logger.info("%s created", peer_name)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange() -> None:
        logger.info("%s state=%s", peer_name, pc.connectionState)
        if pc.connectionState in {"failed", "closed", "disconnected"}:
            await pc.close()
            PEER_CONNECTIONS.discard(pc)

    @pc.on("track")
    def on_track(track: Any) -> None:
        logger.info("%s received %s track", peer_name, track.kind)
        if track.kind == "video":
            try:
                pc.addTrack(BackgroundRemovalTrack(track, create_remover()))
            except Exception:
                logger.exception("unable to initialize background remover")
                track.stop()

        @track.on("ended")
        async def on_ended() -> None:
            logger.info("%s input %s track ended", peer_name, track.kind)

    try:
        await pc.setRemoteDescription(
            RTCSessionDescription(sdp=payload.sdp, type=payload.type)
        )
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
    except Exception as exc:
        logger.exception("%s negotiation failed", peer_name)
        await pc.close()
        PEER_CONNECTIONS.discard(pc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    assert pc.localDescription is not None
    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}


@app.websocket("/ws/process")
async def process_video(websocket: WebSocket) -> None:
    """Process browser-produced JPEG frames over a reliable local TCP path."""
    await websocket.accept()
    mode = websocket.query_params.get("mode", "person")
    if mode not in {"person", "green"}:
        await websocket.close(code=1008, reason="unsupported removal mode")
        return
    try:
        green_strength = float(websocket.query_params.get("strength", "0.78"))
        person_strength = float(
            websocket.query_params.get("person_strength", "0.65")
        )
    except ValueError:
        await websocket.close(code=1008, reason="invalid removal strength")
        return
    remover = create_remover(mode, green_strength, person_strength)
    backend = getattr(
        remover,
        "backend",
        "Green Screen" if mode == "green" else "MediaPipe fallback",
    )
    await websocket.send_json({"type": "ready", "backend": backend})
    try:
        while True:
            jpeg = await websocket.receive_bytes()
            png = await asyncio.to_thread(jpeg_to_transparent_png, jpeg, remover)
            await websocket.send_bytes(png)
    except WebSocketDisconnect:
        logger.info("local video processor disconnected")
    except Exception:
        logger.exception("local video frame processing failed")
        await websocket.close(code=1011, reason="frame processing failed")
    finally:
        remover.close()


@app.on_event("shutdown")
async def shutdown() -> None:
    peers = list(PEER_CONNECTIONS)
    PEER_CONNECTIONS.clear()
    for pc in peers:
        await pc.close()

