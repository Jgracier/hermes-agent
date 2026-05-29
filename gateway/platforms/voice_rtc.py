"""
WebRTC voice session for the Hermes API server.

One VoiceRTCSession per connected client. Full-duplex audio:
  iPhone mic → VAD → STT → LLM run → TTS → iPhone speaker

Interrupt: VAD fires while TTS is playing → cancel current run → restart.

Dependencies: aiortc (pip install aiortc)
"""

import asyncio
import logging
import os
import struct
import tempfile
import time
import wave
from typing import Optional

logger = logging.getLogger(__name__)

# RMS silence thresholds — mirrors voice_mode.py defaults
_SILENCE_RMS = 200
_SPEECH_MIN_SECONDS = 0.4   # ignore clips shorter than this (noise)
_SILENCE_SECONDS = 1.2      # silence after speech ends the utterance


def _rms(pcm_bytes: bytes) -> float:
    """RMS of 16-bit little-endian PCM samples."""
    if not pcm_bytes:
        return 0.0
    samples = struct.unpack_from(f"<{len(pcm_bytes)//2}h", pcm_bytes)
    return (sum(s * s for s in samples) / len(samples)) ** 0.5


class VoiceRTCSession:
    """
    Manages one WebRTC voice session.

    Usage (from the aiohttp WebSocket handler):
        session = VoiceRTCSession(adapter, client_id, session_id)
        await session.run(websocket)
    """

    def __init__(self, adapter, client_id: str, session_id: Optional[str] = None):
        self._adapter = adapter          # APIServerAdapter — used to queue LLM runs
        self._client_id = client_id
        self._session_id = session_id or f"voice-{client_id}-{int(time.time())}"

        self._pc = None                  # RTCPeerConnection
        self._tts_track = None           # outbound audio track
        self._stop_event = asyncio.Event()

        # interrupt coordination
        self._active_run_id: Optional[str] = None
        self._tts_abort = asyncio.Event()

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    async def run(self, ws) -> None:
        """Drive signaling over *ws* (aiohttp WebSocketResponse)."""
        try:
            from aiortc import RTCPeerConnection, RTCSessionDescription
            from aiortc.sdp import candidate_from_sdp
        except ImportError:
            await ws.send_json({"error": "aiortc not installed on server"})
            return

        self._pc = RTCPeerConnection()
        sink = _AudioSink(self._on_utterance)
        self._pc.on("track", lambda track: self._on_track(track, sink))
        self._pc.on("connectionstatechange", lambda: logger.info(
            "[voice_rtc] %s connection: %s", self._client_id, self._pc.connectionState))

        # Add outbound audio track so iPhone gets an audio channel
        self._tts_track = _make_tts_track()
        self._pc.addTrack(self._tts_track)

        try:
            async for msg in ws:
                if self._stop_event.is_set():
                    break
                if msg.type != 0x1:  # WSMsgType.TEXT
                    continue
                try:
                    import json
                    data = json.loads(msg.data)
                except Exception:
                    continue

                kind = data.get("type")

                if kind == "offer":
                    offer_sdp = data["sdp"]
                    # Log direction lines from offer so we can see sendonly/recvonly/sendrecv
                    directions = [l for l in offer_sdp.splitlines() if l.startswith("a=") and any(d in l for d in ["sendonly","recvonly","sendrecv","inactive"])]
                    logger.info("[voice_rtc] %s offer directions: %s", self._client_id, directions)
                    await self._pc.setRemoteDescription(
                        RTCSessionDescription(sdp=offer_sdp, type="offer")
                    )
                    answer = await self._pc.createAnswer()
                    await self._pc.setLocalDescription(answer)
                    answer_sdp = self._pc.localDescription.sdp
                    answer_directions = [l for l in answer_sdp.splitlines() if l.startswith("a=") and any(d in l for d in ["sendonly","recvonly","sendrecv","inactive"])]
                    logger.info("[voice_rtc] %s answer directions: %s", self._client_id, answer_directions)
                    await ws.send_json({
                        "type": "answer",
                        "sdp": answer_sdp,
                    })

                elif kind == "candidate":
                    # iOS sends {"type":"candidate","candidate":{"candidate":"candidate:...","sdpMid":"0","sdpMLineIndex":0}}
                    c = data.get("candidate", {})
                    sdp_str = c.get("candidate", "") if isinstance(c, dict) else ""
                    if sdp_str:
                        try:
                            candidate = candidate_from_sdp(sdp_str.replace("candidate:", "", 1))
                            candidate.sdpMid = c.get("sdpMid")
                            candidate.sdpMLineIndex = c.get("sdpMLineIndex")
                            await self._pc.addIceCandidate(candidate)
                        except Exception as e:
                            logger.debug("[voice_rtc] ICE candidate error: %s", e)

                elif kind == "close":
                    break

        finally:
            self._stop_event.set()
            await self._pc.close()
            logger.info("[voice_rtc] %s session closed", self._client_id)

    # ------------------------------------------------------------------
    # Track / audio handling
    # ------------------------------------------------------------------

    def _on_track(self, track, sink) -> None:
        if track.kind == "audio":
            track.on("ended", lambda: logger.info("[voice_rtc] %s audio track ended", self._client_id))
            asyncio.ensure_future(sink.consume(track))

    async def _on_utterance(self, pcm_bytes: bytes, sample_rate: int) -> None:
        """Called by _AudioSink when a complete utterance is detected."""
        # interrupt any in-flight run + TTS
        await self._interrupt()

        # write PCM to temp WAV
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
        _write_wav(tmp_path, pcm_bytes, sample_rate)

        try:
            # STT
            transcript = await asyncio.get_event_loop().run_in_executor(
                None, _transcribe, tmp_path
            )
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        if not transcript:
            return

        logger.info("[voice_rtc] %s transcript: %s", self._client_id, transcript[:80])

        # LLM run → TTS → stream audio back
        async def _safe():
            try:
                await self._run_and_speak(transcript)
            except Exception:
                logger.exception("[voice_rtc] %s _run_and_speak unhandled error", self._client_id)
        asyncio.ensure_future(_safe())

    async def _interrupt(self) -> None:
        """Cancel any in-flight LLM run and abort TTS playback."""
        self._tts_abort.set()
        run_id = self._active_run_id
        if run_id:
            agent = self._adapter._active_run_agents.get(run_id)
            task = self._adapter._active_run_tasks.get(run_id)
            if agent:
                try:
                    agent.interrupt("Voice interrupt")
                except Exception:
                    pass
            if task and not task.done():
                task.cancel()
        self._active_run_id = None
        # brief yield so cancellation propagates
        await asyncio.sleep(0)
        self._tts_abort.clear()

    async def _run_and_speak(self, text: str) -> None:
        """Send *text* to LLM via /v1/chat/completions streaming — single request, no race."""
        import json, urllib.request, urllib.error

        port = os.getenv("API_SERVER_PORT", "8642")
        base = f"http://localhost:{port}"
        api_key = os.getenv("API_SERVER_KEY", "")
        logger.info("[voice_rtc] %s starting LLM run for: %s", self._client_id, text[:40])

        body = json.dumps({
            "model": "hermes-agent",
            "messages": [{"role": "user", "content": text}],
            "stream": True,
        }).encode()
        req = urllib.request.Request(
            f"{base}/v1/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "X-Hermes-Session-Id": self._session_id,
                "Accept": "text/event-stream",
            },
            method="POST",
        )

        abort = self._tts_abort

        def _stream_response() -> str:
            collected = ""
            try:
                resp = urllib.request.urlopen(req, timeout=120)
                logger.info("[voice_rtc] %s SSE stream opened", self._client_id)
                for raw_line in resp:
                    if abort.is_set():
                        break
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        event = json.loads(payload)
                        delta = (event.get("choices", [{}])[0]
                                 .get("delta", {})
                                 .get("content", ""))
                        if delta:
                            collected += delta
                    except Exception:
                        pass
            except Exception as e:
                logger.warning("[voice_rtc] LLM stream failed: %s", e)
            return collected

        full_text = await asyncio.get_event_loop().run_in_executor(None, _stream_response)
        self._active_run_id = None

        logger.info("[voice_rtc] %s LLM response (%d chars): %s", self._client_id, len(full_text), full_text[:60])
        if not full_text or self._tts_abort.is_set() or self._stop_event.is_set():
            if self._stop_event.is_set():
                logger.warning("[voice_rtc] %s session closed before TTS — discarding response", self._client_id)
            return

        await self._speak(full_text)

    async def _speak(self, text: str) -> None:
        """Convert *text* to audio and feed frames to the outbound track."""
        if not text.strip() or self._tts_abort.is_set():
            return

        logger.info("[voice_rtc] %s TTS generating (%d chars)", self._client_id, len(text))
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tmp_path = f.name

        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, _tts, text, tmp_path
            )
            if not result or self._tts_abort.is_set():
                logger.warning("[voice_rtc] %s TTS failed or aborted (result=%s)", self._client_id, result)
                return
            logger.info("[voice_rtc] %s TTS done, feeding to audio track", self._client_id)
            if self._tts_track:
                await self._tts_track.feed_audio_file(tmp_path, self._tts_abort)
                logger.info("[voice_rtc] %s audio feed complete", self._client_id)
            else:
                logger.warning("[voice_rtc] %s no tts_track to send audio to", self._client_id)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ------------------------------------------------------------------
# Audio sink — VAD + utterance detection
# ------------------------------------------------------------------

class _AudioSink:
    """Consumes an aiortc audio track, runs VAD, fires on_utterance callback."""

    def __init__(self, on_utterance):
        self._on_utterance = on_utterance
        self._buf = bytearray()
        self._speech_started = False
        self._silence_start: Optional[float] = None
        self._speech_start: Optional[float] = None
        self._sample_rate = 48000

    async def consume(self, track) -> None:
        while True:
            try:
                frame = await asyncio.wait_for(track.recv(), timeout=2.0)
            except asyncio.TimeoutError:
                continue
            except Exception:
                break

            self._sample_rate = frame.sample_rate
            # Convert frame to raw 16-bit PCM
            pcm = _frame_to_pcm16(frame)
            rms = _rms(pcm)
            now = time.monotonic()
            is_speech = rms > _SILENCE_RMS

            if is_speech:
                if not self._speech_started:
                    self._speech_started = True
                    self._speech_start = now
                    self._buf.clear()
                self._silence_start = None
                self._buf.extend(pcm)
            else:
                if self._speech_started:
                    self._buf.extend(pcm)
                    if self._silence_start is None:
                        self._silence_start = now
                    elif now - self._silence_start >= _SILENCE_SECONDS:
                        # end of utterance
                        duration = now - (self._speech_start or now)
                        if duration >= _SPEECH_MIN_SECONDS:
                            data = bytes(self._buf)
                            sr = self._sample_rate
                            asyncio.ensure_future(self._on_utterance(data, sr))
                        self._buf.clear()
                        self._speech_started = False
                        self._silence_start = None
                        self._speech_start = None


# ------------------------------------------------------------------
# Outbound TTS audio track
# ------------------------------------------------------------------

def _make_tts_track():
    """Factory that returns a proper MediaStreamTrack subclass instance."""
    from aiortc import MediaStreamTrack
    import av as _av

    class TTSAudioTrack(MediaStreamTrack):
        kind = "audio"

        def __init__(self):
            super().__init__()
            self._queue = asyncio.Queue()
            self._pts = 0
            self._sample_rate = 48000
            self._samples_per_frame = 960  # 20ms at 48kHz

        async def recv(self):
            from fractions import Fraction
            try:
                pcm = await asyncio.wait_for(self._queue.get(), timeout=0.02)
            except asyncio.TimeoutError:
                pcm = bytes(self._samples_per_frame * 2)  # silence

            samples = len(pcm) // 2
            frame = _av.AudioFrame(format="s16", layout="mono", samples=samples)
            frame.sample_rate = self._sample_rate
            frame.pts = self._pts
            frame.time_base = Fraction(1, self._sample_rate)
            frame.planes[0].update(pcm)
            self._pts += samples
            return frame

        async def feed_audio_file(self, path: str, abort_event: asyncio.Event) -> None:
            try:
                loop = asyncio.get_event_loop()
                def _decode():
                    chunks = []
                    container = _av.open(path)
                    resampler = _av.AudioResampler(format="s16", layout="mono", rate=self._sample_rate)
                    chunk_size = self._samples_per_frame * 2
                    buf = b""
                    for frame in container.decode(audio=0):
                        for resampled in resampler.resample(frame):
                            buf += bytes(resampled.planes[0])
                    # flush resampler
                    for resampled in resampler.resample(None):
                        buf += bytes(resampled.planes[0])
                    for i in range(0, len(buf), chunk_size):
                        chunks.append(buf[i:i + chunk_size])
                    return chunks

                chunks = await loop.run_in_executor(None, _decode)
                for chunk in chunks:
                    if abort_event.is_set():
                        break
                    await self._queue.put(chunk)
            except Exception as e:
                logger.warning("[voice_rtc] TTS feed error: %s", e)

    return TTSAudioTrack()


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _frame_to_pcm16(frame) -> bytes:
    """Convert an aiortc audio frame to raw 16-bit mono PCM bytes."""
    import av
    resampler = av.AudioResampler(format="s16", layout="mono", rate=frame.sample_rate)
    out = b""
    for resampled in resampler.resample(frame):
        out += bytes(resampled.planes[0])
    return out


def _write_wav(path: str, pcm: bytes, sample_rate: int) -> None:
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)


def _transcribe(wav_path: str) -> str:
    try:
        from tools.transcription_tools import transcribe_audio
        result = transcribe_audio(wav_path)
        return result.get("transcript", "").strip() if result.get("success") else ""
    except Exception as e:
        logger.warning("[voice_rtc] STT error: %s", e)
        return ""


def _tts(text: str, output_path: str) -> bool:
    try:
        from tools.tts_tool import text_to_speech_tool
        result = text_to_speech_tool(text, output_path=output_path)
        return bool(result and "success" in result)
    except Exception as e:
        logger.warning("[voice_rtc] TTS error: %s", e)
        return False
