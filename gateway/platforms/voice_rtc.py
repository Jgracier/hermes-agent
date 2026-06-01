"""
WebRTC voice session for the Hermes API server.

One VoiceRTCSession per connected client. Full-duplex audio:
  iPhone mic → VAD → STT → LLM streaming → TTS → WebRTC Opus → speaker

Audio delivery:
  - TTS generates MP3, decoded to WAV with silence trimmed
  - MediaPlayer plays WAV via aiortc's background thread (no asyncio timing)
  - RTCRtpSender.replaceTrack() swaps tracks per response without renegotiation
  - Interrupt: replaceTrack(silent) immediately cuts outbound audio

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

_SPEECH_MIN_SECONDS = 0.4  # ignore clips shorter than this (noise burst)
_SILENCE_SECONDS = 0.8     # shorter than voice_mode's 3.0s — chat needs fast turn-taking
_CHUNK_SIZE = 200           # LLM chars buffered before TTS flush
_VAD_AGGRESSIVENESS = 2    # webrtcvad: 0=least aggressive, 3=most aggressive


class VoiceRTCSession:
    """Manages one WebRTC voice session."""

    def __init__(self, adapter, client_id: str, session_id: Optional[str] = None):
        self._adapter = adapter
        self._client_id = client_id
        self._session_id = session_id or f"voice-{client_id}-{int(time.time())}"
        self._pc = None
        self._sender = None
        self._silent_track = None
        self._stop_event = asyncio.Event()
        self._tts_abort = asyncio.Event()
        self._tts_play_until: float = 0.0  # monotonic time when current playback ends
        self._history: List[dict] = []

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

        # Silent track as initial placeholder; swapped to MediaPlayer per response
        self._silent_track = _make_silent_track()
        self._sender = self._pc.addTrack(self._silent_track)

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
    # Inbound audio handling
    # ------------------------------------------------------------------

    def _on_track(self, track, sink) -> None:
        if track.kind == "audio":
            asyncio.ensure_future(sink.consume(track))

    async def _on_utterance(self, pcm_bytes: bytes, sample_rate: int) -> None:
        # Echo suppression: ignore mic input while TTS is still playing
        if time.monotonic() < self._tts_play_until:
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

        # Filter Whisper hallucinations on silence/noise — reuse Hermes's
        # curated phrase list and repetition patterns ("Thank you", "you",
        # "Продолжение следует...", subtitle credits, etc.)
        if _is_hallucination(transcript):
            logger.debug("[voice_rtc] %s filtered hallucination: %s", self._client_id, transcript[:60])
            return

        logger.info("[voice_rtc] %s transcript: %s", self._client_id, transcript[:80])

        async def _safe():
            try:
                await self._run_and_speak(transcript)
            except Exception:
                logger.exception("[voice_rtc] %s run_and_speak error", self._client_id)

        asyncio.ensure_future(_safe())

    async def _interrupt(self) -> None:
        """Abort in-flight TTS and cut outbound audio immediately."""
        self._tts_abort.set()
        self._tts_play_until = 0.0
        if self._sender and self._silent_track:
            try:
                await self._sender.replaceTrack(self._silent_track)
            except Exception:
                pass
        await asyncio.sleep(0)
        self._tts_abort.clear()

    # ------------------------------------------------------------------
    # LLM → TTS → MediaPlayer pipeline
    # ------------------------------------------------------------------

    async def _run_and_speak(self, text: str) -> None:
        port = os.getenv("API_SERVER_PORT", "8642")
        api_key = os.getenv("API_SERVER_KEY", "")
        abort = self._tts_abort
        loop = asyncio.get_event_loop()
        logger.info("[voice_rtc] %s LLM: %s", self._client_id, text[:50])

        self._history.append({"role": "user", "content": text})
        messages = list(self._history)

        body = json.dumps({
            "model": "hermes-agent",
            "messages": messages,
            "stream": True,
        }).encode()

        import urllib.request
        req = urllib.request.Request(
            f"http://localhost:{port}/v1/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "X-Hermes-Session-Id": self._session_id,
            },
            method="POST",
        )

        chunk_q: asyncio.Queue = asyncio.Queue()

        def _stream_chunks() -> None:
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
                logger.warning("[voice_rtc] LLM failed: %s", e)
            finally:
                if buf.strip():
                    loop.call_soon_threadsafe(chunk_q.put_nowait, buf.strip())
                loop.call_soon_threadsafe(chunk_q.put_nowait, None)

        stream_future = loop.run_in_executor(None, _stream_chunks)

        # Collect text chunks and fire TTS concurrently per chunk
        tts_tasks = []
        full_parts = []

        while not abort.is_set() and not self._stop_event.is_set():
            try:
                chunk = await asyncio.wait_for(chunk_q.get(), timeout=30.0)
            except asyncio.TimeoutError:
                break
            if chunk is None:
                break
            full_parts.append(chunk)
            tts_tasks.append(loop.run_in_executor(None, _tts_to_wav, chunk))

        await stream_future

        if not tts_tasks or abort.is_set() or self._stop_event.is_set():
            return

        full_response = " ".join(full_parts)
        self._history.append({"role": "assistant", "content": full_response})
        if len(self._history) > 40:
            self._history = self._history[-40:]

        logger.info("[voice_rtc] %s playing %d audio chunk(s)", self._client_id, len(tts_tasks))

        from aiortc.contrib.media import MediaPlayer
        prev_player = None

        for task in tts_tasks:
            if abort.is_set() or self._stop_event.is_set():
                break

            wav_path, duration = await task
            if not wav_path:
                continue

            try:
                player = MediaPlayer(wav_path)
                if player.audio is None:
                    logger.warning("[voice_rtc] %s MediaPlayer.audio is None for %s", self._client_id, wav_path)
                    continue

                if not self._sender:
                    logger.warning("[voice_rtc] %s sender not ready, skipping playback", self._client_id)
                    continue

                logger.info("[voice_rtc] %s replaceTrack → player (%.2fs)", self._client_id, duration)

                # Stop previous player before swapping
                if prev_player is not None:
                    try:
                        prev_player.audio.stop()
                    except Exception:
                        pass

                await self._sender.replaceTrack(player.audio)
                prev_player = player

                logger.info("[voice_rtc] %s replaceTrack done, sleeping %.2fs", self._client_id, duration)

                # Track when this chunk ends for echo suppression
                self._tts_play_until = time.monotonic() + duration + 0.2

                # Wait for playback — duration is exact (from WAV sample count)
                await asyncio.sleep(duration + 0.15)

                logger.info("[voice_rtc] %s playback chunk done", self._client_id)

            except Exception as e:
                logger.warning("[voice_rtc] playback error: %s", e)
            finally:
                try:
                    os.unlink(wav_path)
                except OSError:
                    pass

        # Back to silence after all chunks play
        if not abort.is_set() and self._sender and self._silent_track:
            try:
                await self._sender.replaceTrack(self._silent_track)
            except Exception:
                pass
        if prev_player is not None:
            try:
                prev_player.audio.stop()
            except Exception:
                pass


# ------------------------------------------------------------------
# Audio sink — VAD + utterance detection
# ------------------------------------------------------------------

class _AudioSink:
    """Receives aiortc audio frames and detects utterances using the standard
    py-webrtcvad collector algorithm with hysteresis.

    Whisper runs at 16kHz natively, so we resample once to 16kHz and use the
    SAME stream for both VAD and the captured utterance — no dual-buffer drift.

    Hysteresis (the core fix for noise-triggering):
      - To START capturing: ≥ _TRIGGER_RATIO of frames in a sliding window
        must be speech. A single noise burst cannot trip it.
      - To END the utterance: ≥ _TRIGGER_RATIO of the window must be silence.
    """

    _RATE = 16000
    _FRAME_MS = 20
    _FRAME_SAMPLES = _RATE * _FRAME_MS // 1000   # 320 samples
    _FRAME_BYTES = _FRAME_SAMPLES * 2            # 640 bytes
    _WINDOW_FRAMES = 15                          # 300ms sliding window
    _TRIGGER_RATIO = 0.9                         # 90% agreement to flip state
    _MIN_UTTERANCE_FRAMES = int(_SPEECH_MIN_SECONDS * 1000 / _FRAME_MS)

    def __init__(self, on_utterance):
        from collections import deque
        self._on_utterance = on_utterance
        self._pending = bytearray()              # raw 16kHz bytes not yet framed
        self._utterance = bytearray()            # captured speech (16kHz)
        self._window = deque(maxlen=self._WINDOW_FRAMES)  # (frame, is_speech)
        self._triggered = False
        self._resampler = None
        try:
            import webrtcvad
            self._vad = webrtcvad.Vad(_VAD_AGGRESSIVENESS)
        except Exception:
            self._vad = None

    async def consume(self, track) -> None:
        import av as _av
        while True:
            try:
                frame = await asyncio.wait_for(track.recv(), timeout=2.0)
            except asyncio.TimeoutError:
                continue
            except Exception:
                break

            if self._resampler is None:
                self._resampler = _av.AudioResampler(
                    format="s16", layout="mono", rate=self._RATE
                )

            for r in self._resampler.resample(frame):
                self._pending += bytes(r.planes[0])

            # Process every complete 20ms frame
            while len(self._pending) >= self._FRAME_BYTES:
                chunk = bytes(self._pending[:self._FRAME_BYTES])
                del self._pending[:self._FRAME_BYTES]
                self._process_frame(chunk)

    def _process_frame(self, chunk: bytes) -> None:
        if self._vad is None:
            return  # no VAD available — silently drop (don't flood STT)
        try:
            speech = self._vad.is_speech(chunk, self._RATE)
        except Exception:
            return

        self._window.append((chunk, speech))

        if not self._triggered:
            # Enter speech only when the window is overwhelmingly speech
            n_speech = sum(1 for _, s in self._window if s)
            if n_speech >= self._TRIGGER_RATIO * self._WINDOW_FRAMES:
                self._triggered = True
                # Seed utterance with the buffered window (captures onset)
                for f, _ in self._window:
                    self._utterance += f
                self._window.clear()
        else:
            self._utterance += chunk
            # End utterance when the window is overwhelmingly silence
            n_silence = sum(1 for _, s in self._window if not s)
            if n_silence >= self._TRIGGER_RATIO * self._WINDOW_FRAMES:
                utterance = bytes(self._utterance)
                n_frames = len(utterance) // self._FRAME_BYTES
                self._triggered = False
                self._utterance.clear()
                self._window.clear()
                # Require a minimum length so a brief blip isn't transcribed
                if n_frames >= self._MIN_UTTERANCE_FRAMES:
                    asyncio.ensure_future(self._on_utterance(utterance, self._RATE))


# ------------------------------------------------------------------
# Silent placeholder track
# ------------------------------------------------------------------

def _make_silent_track():
    """Clock-paced silent track — holds the sender slot between responses."""
    from aiortc import MediaStreamTrack
    import av as _av

    class SilentTrack(MediaStreamTrack):
        kind = "audio"

        def __init__(self):
            super().__init__()
            self._pts = 0
            self._start = time.monotonic()

        async def recv(self):
            due = self._start + (self._pts / 48000)
            wait = due - time.monotonic()
            if wait > 0.001:
                await asyncio.sleep(wait)
            frame = _av.AudioFrame(format="s16", layout="mono", samples=960)
            frame.sample_rate = 48000
            frame.pts = self._pts
            frame.time_base = Fraction(1, 48000)
            frame.planes[0].update(bytes(960 * 2))
            self._pts += 960
            return frame

    return SilentTrack()


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _write_wav(path: str, pcm: bytes, sample_rate: int) -> None:
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)


def _trim_silence(pcm: bytes, sample_rate: int = 48000,
                  thresh: int = 80, max_silence_ms: int = 80) -> bytes:
    """Reduce Edge TTS silence padding from ~950ms to max_silence_ms."""
    if not pcm:
        return pcm
    n = len(pcm) // 2
    samples = list(struct.unpack_from(f"<{n}h", pcm))
    max_s = int(sample_rate * max_silence_ms / 1000)

    start = 0
    while start < n and abs(samples[start]) < thresh:
        start += 1
    start = max(0, start - max_s)

    end = n - 1
    while end > start and abs(samples[end]) < thresh:
        end -= 1
    end = min(n - 1, end + max_s)

    out, i = [], start
    while i <= end:
        if abs(samples[i]) < thresh:
            rs = i
            while i <= end and abs(samples[i]) < thresh:
                i += 1
            out.extend(samples[rs:rs + min(i - rs, max_s)])
        else:
            out.append(samples[i])
            i += 1

    if not out:
        return b""
    return struct.pack(f"<{len(out)}h", *out)


def _tts_to_wav(text: str):
    """TTS text → (wav_path, duration_seconds). Returns (None, 0) on failure."""
    import av as _av
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        mp3_path = f.name
    try:
        if not _tts(text, mp3_path):
            return None, 0.0

        # Decode MP3 → PCM, trim silence, save as WAV
        container = _av.open(mp3_path)
        resampler = _av.AudioResampler(format="s16", layout="mono", rate=48000)
        buf = b""
        for frame in container.decode(audio=0):
            for r in resampler.resample(frame):
                buf += bytes(r.planes[0])
        for r in resampler.resample(None):
            buf += bytes(r.planes[0])
        container.close()

        buf = _trim_silence(buf)
        if not buf:
            return None, 0.0

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = f.name
        _write_wav(wav_path, buf, 48000)

        duration = len(buf) / 2 / 48000  # exact: samples / sample_rate
        return wav_path, duration

    except Exception as e:
        logger.warning("[voice_rtc] TTS→WAV error: %s", e)
        return None, 0.0
    finally:
        try:
            os.unlink(mp3_path)
        except OSError:
            pass


def _transcribe(wav_path: str) -> str:
    try:
        from tools.transcription_tools import transcribe_audio
        result = transcribe_audio(wav_path)
        return result.get("transcript", "").strip() if result.get("success") else ""
    except Exception as e:
        logger.warning("[voice_rtc] STT error: %s", e)
        return ""


def _is_hallucination(transcript: str) -> bool:
    """True if transcript is a known Whisper silence hallucination.

    Delegates to Hermes's curated filter (tools.voice_mode). Falls back to a
    minimal length check only if that import is unavailable.
    """
    try:
        from tools.voice_mode import is_whisper_hallucination
        return is_whisper_hallucination(transcript)
    except ImportError:
        return len(transcript.strip().strip(".,!? \t")) < 3


def _tts(text: str, output_path: str) -> bool:
    try:
        from tools.tts_tool import text_to_speech_tool
        result = text_to_speech_tool(text, output_path=output_path)
        return bool(result and "success" in result)
    except Exception as e:
        logger.warning("[voice_rtc] TTS error: %s", e)
        return False
