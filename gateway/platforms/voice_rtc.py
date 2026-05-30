"""
WebRTC voice session for the Hermes API server.

One VoiceRTCSession per connected client. Full-duplex audio:
  iPhone mic → VAD → STT → LLM streaming → sentence TTS pipeline → speaker

Latency design:
  - Sentence N TTS starts immediately when sentence N arrives from LLM stream
  - Sentence N+1 TTS runs concurrently while sentence N's audio plays
  - Audio fed to track queue in order — seamless FIFO playback, no gaps
  - Interrupt: VAD fires → abort flag → cancel in-flight run → restart

Dependencies: aiortc (pip install aiortc)
"""

import asyncio
import json
import logging
import os
import re
import struct
import tempfile
import time
import wave
from fractions import Fraction
from typing import List, Optional

logger = logging.getLogger(__name__)

# Reuse RMS threshold from voice_mode.py — single source of truth
try:
    from tools.voice_mode import SILENCE_RMS_THRESHOLD as _SILENCE_RMS
except ImportError:
    _SILENCE_RMS = 200

_SPEECH_MIN_SECONDS = 0.4  # ignore clips shorter than this (noise burst)
_SILENCE_SECONDS = 0.8     # shorter than voice_mode's 3.0s — chat needs fast turn-taking


def _rms(pcm_bytes: bytes) -> float:
    if not pcm_bytes:
        return 0.0
    samples = struct.unpack_from(f"<{len(pcm_bytes)//2}h", pcm_bytes)
    return (sum(s * s for s in samples) / len(samples)) ** 0.5


class VoiceRTCSession:
    """Manages one WebRTC voice session."""

    def __init__(self, adapter, client_id: str, session_id: Optional[str] = None):
        self._adapter = adapter
        self._client_id = client_id
        self._session_id = session_id or f"voice-{client_id}-{int(time.time())}"
        self._pc = None
        self._tts_track = None
        self._stop_event = asyncio.Event()
        self._tts_abort = asyncio.Event()
        self._tts_mute_until: float = 0.0  # wall-clock time until which VAD is muted
        self._history: List[dict] = []  # conversation history for context

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    async def run(self, ws) -> None:
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
            "[voice_rtc] %s state: %s", self._client_id, self._pc.connectionState))

        self._tts_track = _make_tts_track()
        self._pc.addTrack(self._tts_track)

        try:
            async for msg in ws:
                if self._stop_event.is_set():
                    break
                if msg.type != 0x1:  # TEXT
                    continue
                try:
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
                    await ws.send_json({"type": "answer", "sdp": self._pc.localDescription.sdp})

                elif kind == "candidate":
                    c = data.get("candidate", {})
                    sdp_str = c.get("candidate", "") if isinstance(c, dict) else ""
                    if sdp_str:
                        try:
                            candidate = candidate_from_sdp(sdp_str.replace("candidate:", "", 1))
                            candidate.sdpMid = c.get("sdpMid")
                            candidate.sdpMLineIndex = c.get("sdpMLineIndex")
                            await self._pc.addIceCandidate(candidate)
                        except Exception as e:
                            logger.debug("[voice_rtc] ICE candidate: %s", e)

                elif kind == "close":
                    break
        finally:
            self._stop_event.set()
            await self._pc.close()
            logger.info("[voice_rtc] %s session closed", self._client_id)

    # ------------------------------------------------------------------
    # Track / utterance handling
    # ------------------------------------------------------------------

    def _on_track(self, track, sink) -> None:
        if track.kind == "audio":
            asyncio.ensure_future(sink.consume(track))

    async def _on_utterance(self, pcm_bytes: bytes, sample_rate: int) -> None:
        # Suppress echo — ignore audio while TTS is still playing back
        if time.time() < self._tts_mute_until:
            return

        await self._interrupt()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
        _write_wav(tmp_path, pcm_bytes, sample_rate)

        try:
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

        # Filter whisper hallucinations (". . ." / very short noise)
        cleaned = transcript.strip().strip(".,!? \t")
        if len(cleaned) < 3 or all(c in "., " for c in cleaned):
            return

        logger.info("[voice_rtc] %s transcript: %s", self._client_id, transcript[:80])

        async def _safe():
            try:
                await self._run_and_speak(transcript)
            except Exception:
                logger.exception("[voice_rtc] %s run_and_speak error", self._client_id)

        asyncio.ensure_future(_safe())

    async def _interrupt(self) -> None:
        self._tts_abort.set()
        await asyncio.sleep(0)
        self._tts_abort.clear()

    # ------------------------------------------------------------------
    # LLM → sentence streaming → parallel TTS → ordered audio queue
    # ------------------------------------------------------------------

    async def _run_and_speak(self, text: str) -> None:
        """
        Pipeline:
          1. POST /v1/chat/completions streaming
          2. Detect sentence boundaries as deltas arrive
          3. Fire TTS task for each sentence immediately (concurrent)
          4. Feed audio to track queue in order (seamless playback)
        """
        port = os.getenv("API_SERVER_PORT", "8642")
        api_key = os.getenv("API_SERVER_KEY", "")
        url = f"http://localhost:{port}/v1/chat/completions"
        abort = self._tts_abort
        loop = asyncio.get_event_loop()
        logger.info("[voice_rtc] %s LLM start: %s", self._client_id, text[:50])

        # Maintain conversation history for multi-turn context
        self._history.append({"role": "user", "content": text})
        messages = list(self._history)

        body = json.dumps({
            "model": "hermes-agent",
            "messages": messages,
            "stream": True,
        }).encode()

        import urllib.request
        req = urllib.request.Request(url, data=body, headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-Hermes-Session-Id": self._session_id,
        }, method="POST")

        # asyncio.Queue: stream full response then TTS as one or two chunks.
        # Splitting into many sentences = many Edge TTS calls = jitter between each.
        # Instead: buffer 200 chars then flush, giving at most 2-3 TTS calls per response.
        chunk_q: asyncio.Queue = asyncio.Queue()
        _CHUNK_SIZE = 200  # chars — tune up for fewer calls, down for lower first-word latency

        def _stream_chunks() -> None:
            """Run in executor: parse SSE, emit text chunks of ~CHUNK_SIZE chars."""
            buf = ""
            try:
                resp = urllib.request.urlopen(req, timeout=60)
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
                        delta = (json.loads(payload)
                                 .get("choices", [{}])[0]
                                 .get("delta", {})
                                 .get("content", ""))
                        if delta:
                            buf += delta
                            # Flush on sentence boundary after enough chars — avoids mid-word cuts
                            if len(buf) >= _CHUNK_SIZE:
                                m = re.search(r'(?<=[.!?,;])\s+', buf)
                                if m:
                                    chunk = buf[:m.start()].strip()
                                    buf = buf[m.end():]
                                    if chunk:
                                        loop.call_soon_threadsafe(chunk_q.put_nowait, chunk)
                    except Exception:
                        pass
            except Exception as e:
                logger.warning("[voice_rtc] LLM stream failed: %s", e)
            finally:
                if buf.strip():
                    loop.call_soon_threadsafe(chunk_q.put_nowait, buf.strip())
                loop.call_soon_threadsafe(chunk_q.put_nowait, None)
                loop.call_soon_threadsafe(logger.info, "[voice_rtc] LLM stream complete")

        sentence_q = chunk_q  # alias so rest of code is unchanged

        def _tts_to_chunks(sentence: str) -> List[bytes]:
            """TTS one sentence → list of raw PCM chunks, ready to queue."""
            import av as _av
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                tmp = f.name
            try:
                if not _tts(sentence, tmp):
                    return []
                chunks = []
                container = _av.open(tmp)
                resampler = _av.AudioResampler(format="s16", layout="mono", rate=48000)
                buf = b""
                for frame in container.decode(audio=0):
                    for r in resampler.resample(frame):
                        buf += bytes(r.planes[0])
                for r in resampler.resample(None):
                    buf += bytes(r.planes[0])
                chunk_size = 960 * 2  # 20ms at 48kHz, 16-bit mono
                for i in range(0, len(buf), chunk_size):
                    chunks.append(buf[i:i + chunk_size])
                return chunks
            except Exception as e:
                logger.warning("[voice_rtc] TTS chunk error: %s", e)
                return []
            finally:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass

        # Start LLM streaming in background thread
        stream_future = loop.run_in_executor(None, _stream_chunks)

        # Collect sentences and fire TTS tasks immediately (concurrent)
        tts_tasks = []
        full_response_parts = []

        while not abort.is_set() and not self._stop_event.is_set():
            try:
                sentence = await asyncio.wait_for(sentence_q.get(), timeout=30.0)
            except asyncio.TimeoutError:
                break
            if sentence is None:
                break
            full_response_parts.append(sentence)
            # Fire TTS immediately — concurrent with remaining LLM generation
            task = loop.run_in_executor(None, _tts_to_chunks, sentence)
            tts_tasks.append(task)

        await stream_future

        if not tts_tasks or abort.is_set() or self._stop_event.is_set():
            return

        # Store assistant response in history
        full_response = " ".join(full_response_parts)
        self._history.append({"role": "assistant", "content": full_response})
        # Cap history at 20 turns to avoid token bloat
        if len(self._history) > 40:
            self._history = self._history[-40:]

        logger.info("[voice_rtc] %s response: %d sentences, queuing audio", self._client_id, len(tts_tasks))
        total_chunks = 0
        # Feed audio in order — each task finishes in parallel, we drain in sequence
        for task in tts_tasks:
            if abort.is_set() or self._stop_event.is_set():
                break
            chunks = await task
            if chunks and self._tts_track:
                for chunk in chunks:
                    if abort.is_set():
                        break
                    await self._tts_track.put_chunk(chunk)
                    total_chunks += 1

        # Mute VAD for estimated playback duration + 0.5s buffer for echo tail
        if total_chunks > 0:
            playback_seconds = total_chunks * 0.02  # 20ms per chunk
            self._tts_mute_until = time.time() + playback_seconds + 0.5


# ------------------------------------------------------------------
# Audio sink — VAD + utterance detection
# ------------------------------------------------------------------

class _AudioSink:
    """Receives aiortc audio frames, runs RMS-VAD, fires on_utterance."""

    def __init__(self, on_utterance):
        self._on_utterance = on_utterance
        self._buf = bytearray()
        self._speech_started = False
        self._silence_start: Optional[float] = None
        self._speech_start: Optional[float] = None
        self._sample_rate = 48000
        self._resampler = None  # created once on first frame

    async def consume(self, track) -> None:
        import av as _av
        while True:
            try:
                frame = await asyncio.wait_for(track.recv(), timeout=2.0)
            except asyncio.TimeoutError:
                continue
            except Exception:
                break

            self._sample_rate = frame.sample_rate

            # Persistent resampler — created once, not per frame
            if self._resampler is None:
                self._resampler = _av.AudioResampler(
                    format="s16", layout="mono", rate=frame.sample_rate
                )

            pcm = b""
            for r in self._resampler.resample(frame):
                pcm += bytes(r.planes[0])

            if not pcm:
                continue

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
            elif self._speech_started:
                self._buf.extend(pcm)
                if self._silence_start is None:
                    self._silence_start = now
                elif now - self._silence_start >= _SILENCE_SECONDS:
                    duration = now - (self._speech_start or now)
                    if duration >= _SPEECH_MIN_SECONDS:
                        asyncio.ensure_future(
                            self._on_utterance(bytes(self._buf), self._sample_rate)
                        )
                    self._buf.clear()
                    self._speech_started = False
                    self._silence_start = None
                    self._speech_start = None


# ------------------------------------------------------------------
# Outbound TTS audio track
# ------------------------------------------------------------------

def _make_tts_track():
    from aiortc import MediaStreamTrack
    import av as _av

    class TTSAudioTrack(MediaStreamTrack):
        kind = "audio"

        def __init__(self):
            super().__init__()
            self._queue: asyncio.Queue = asyncio.Queue()
            self._pts = 0
            self._sample_rate = 48000
            self._samples_per_frame = 960  # 20ms at 48kHz

        async def put_chunk(self, pcm: bytes) -> None:
            await self._queue.put(pcm)

        async def recv(self):
            # Clock-based pacing: sleep until this frame is due.
            # Without this, recv() is called thousands of times/sec,
            # flooding the Opus encoder and producing garbled audio.
            if not hasattr(self, "_start"):
                self._start = time.time()

            # Get audio or silence — never block
            try:
                pcm = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pcm = bytes(self._samples_per_frame * 2)

            # Ensure exactly one frame of samples, pad with silence if short
            frame_bytes = self._samples_per_frame * 2
            pcm = (pcm + bytes(frame_bytes))[:frame_bytes]

            # s16 interleaved, 1D array for mono — what aiortc's Opus encoder expects
            import numpy as np
            data = np.frombuffer(pcm, dtype=np.int16)
            frame = _av.AudioFrame.from_ndarray(data.reshape(1, -1), format="s16", layout="mono")
            frame.sample_rate = self._sample_rate
            frame.pts = self._pts
            frame.time_base = Fraction(1, self._sample_rate)
            self._pts += self._samples_per_frame

            # Sleep until this frame's wall-clock time
            due = self._start + (self._pts / self._sample_rate)
            wait = due - time.time()
            if wait > 0:
                await asyncio.sleep(wait)

            return frame

    return TTSAudioTrack()


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

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
