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
        self._tts_track = _TTSAudioTrack()
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
                    await self._pc.setRemoteDescription(
                        RTCSessionDescription(sdp=data["sdp"], type="offer")
                    )
                    answer = await self._pc.createAnswer()
                    await self._pc.setLocalDescription(answer)
                    await ws.send_json({
                        "type": "answer",
                        "sdp": self._pc.localDescription.sdp,
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
        """Send *text* to the LLM via /v1/runs, stream events, speak response."""
        import json, urllib.request, urllib.error

        port = os.getenv("API_SERVER_PORT", "8642")
        base = f"http://localhost:{port}"
        api_key = os.getenv("API_SERVER_KEY", "")
        logger.info("[voice_rtc] %s starting LLM run for: %s", self._client_id, text[:40])
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-Hermes-Session-Id": self._session_id,
        }

        # Step 1 — POST /v1/runs, get run_id back immediately
        body = json.dumps({"input": text, "session_id": self._session_id}).encode()
        req = urllib.request.Request(f"{base}/v1/runs", data=body, headers=headers, method="POST")
        try:
            resp = await asyncio.get_event_loop().run_in_executor(
                None, lambda: urllib.request.urlopen(req, timeout=10)
            )
            run_data = json.loads(resp.read())
            run_id = run_data.get("id") or run_data.get("run_id")
        except Exception as e:
            logger.warning("[voice_rtc] LLM run create failed: %s", e)
            return

        if not run_id:
            logger.warning("[voice_rtc] No run_id in response: %s", run_data)
            return

        self._active_run_id = run_id
        logger.info("[voice_rtc] %s run created: %s", self._client_id, run_id)

        # Step 2 — GET /v1/runs/{run_id}/events, collect SSE text deltas.
        # Run entirely in executor — SSE iteration is blocking I/O and must
        # not touch the asyncio event loop directly.
        event_req = urllib.request.Request(
            f"{base}/v1/runs/{run_id}/events",
            headers={**headers, "Accept": "text/event-stream"},
        )
        abort = self._tts_abort

        def _consume_events() -> str:
            text = ""
            # Retry a few times — the run may not have started streaming yet
            for attempt in range(3):
                try:
                    resp = urllib.request.urlopen(event_req, timeout=120)
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
                                text += delta
                        except Exception:
                            pass
                    if text:
                        break
                    import time as _time
                    _time.sleep(0.5)
                except Exception as e:
                    logger.warning("[voice_rtc] LLM run events attempt %d failed: %s", attempt + 1, e)
                    import time as _time
                    _time.sleep(0.5)
            return text

        full_text = ""
        try:
            full_text = await asyncio.get_event_loop().run_in_executor(None, _consume_events)
        finally:
            self._active_run_id = None

        logger.info("[voice_rtc] %s LLM response (%d chars): %s", self._client_id, len(full_text), full_text[:60])
        if not full_text or self._tts_abort.is_set():
            return

        await self._speak(full_text)

    async def _speak(self, text: str) -> None:
        """Convert *text* to audio and feed frames to the outbound track."""
        if not text.strip() or self._tts_abort.is_set():
            return

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tmp_path = f.name

        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, _tts, text, tmp_path
            )
            if not result or self._tts_abort.is_set():
                return
            if self._tts_track:
                await self._tts_track.feed_audio_file(tmp_path, self._tts_abort)
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

class _TTSAudioTrack:
    """
    aiortc MediaStreamTrack that streams TTS audio to the client.
    Starts silent; feed_audio_file() decodes an mp3/wav and queues frames.
    """
    kind = "audio"

    def __init__(self):
        try:
            from aiortc import MediaStreamTrack as _MST
            # We can't easily subclass without aiortc internals, so we use
            # MediaPlayer as a source and swap audio on demand.
            # For now: use a queue-based custom track approach.
        except ImportError:
            pass
        self._queue: asyncio.Queue = asyncio.Queue()
        self._pts = 0
        self._sample_rate = 48000
        self._samples_per_frame = 960  # 20ms at 48kHz

    async def recv(self):
        """Called by aiortc to get the next audio frame."""
        import av
        from aiortc import MediaStreamTrack
        try:
            pcm = await asyncio.wait_for(self._queue.get(), timeout=0.02)
        except asyncio.TimeoutError:
            # silence frame
            pcm = bytes(self._samples_per_frame * 2)

        frame = av.AudioFrame(format="s16", layout="mono", samples=len(pcm) // 2)
        frame.sample_rate = self._sample_rate
        frame.pts = self._pts
        frame.time_base = f"1/{self._sample_rate}"
        frame.planes[0].update(pcm)
        self._pts += len(pcm) // 2
        return frame

    async def feed_audio_file(self, path: str, abort_event: asyncio.Event) -> None:
        """Decode audio file and queue PCM frames for transmission."""
        try:
            import av
            container = av.open(path)
            resampler = av.AudioResampler(format="s16", layout="mono", rate=self._sample_rate)
            for frame in container.decode(audio=0):
                if abort_event.is_set():
                    break
                for resampled in resampler.resample(frame):
                    pcm = bytes(resampled.planes[0])
                    # split into 20ms chunks
                    chunk_size = self._samples_per_frame * 2
                    for i in range(0, len(pcm), chunk_size):
                        if abort_event.is_set():
                            break
                        await self._queue.put(pcm[i:i + chunk_size])
                        await asyncio.sleep(0.018)  # pace to ~real-time
        except Exception as e:
            logger.warning("[voice_rtc] TTS feed error: %s", e)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _frame_to_pcm16(frame) -> bytes:
    """Convert an aiortc audio frame to raw 16-bit mono PCM bytes."""
    import av
    resampler = av.AudioResampler(format="s16", layout="mono", rate=frame.sample_rate)
    resampled = resampler.resample(frame)
    if resampled:
        return bytes(resampled[0].planes[0])
    return b""


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
