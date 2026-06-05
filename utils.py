"""
Shared utilities: TTS engine, camera manager, and logging setup.
"""
import collections
import logging
import os
import queue
import re
import subprocess
import threading
import time
from logging.handlers import RotatingFileHandler

import cv2

import config


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging() -> logging.Logger:
    os.makedirs(os.path.dirname(config.LOG_FILE), exist_ok=True)
    logger = logging.getLogger("smart_glass")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = RotatingFileHandler(
        config.LOG_FILE,
        maxBytes=config.LOG_MAX_BYTES,
        backupCount=config.LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


logger = logging.getLogger("smart_glass.utils")


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

# Bangla Unicode block: U+0980–U+09FF
_BANGLA_RE = re.compile(r"[ঀ-৿]")


def detect_language(text: str) -> str:
    """Return 'bn' if >25% of non-space characters are Bangla, else 'en'."""
    stripped = text.replace(" ", "")
    if not stripped:
        return "en"
    bangla_count = len(_BANGLA_RE.findall(stripped))
    return "bn" if (bangla_count / len(stripped)) > 0.25 else "en"


# ---------------------------------------------------------------------------
# TTSEngine
# ---------------------------------------------------------------------------

class TTSEngine:
    """
    Non-blocking TTS via a background worker thread.
    Priority order: Piper TTS (if model present) → espeak-ng fallback.
    """

    def __init__(self, volume: int = config.DEFAULT_VOLUME):
        self._volume = volume
        self._queue: queue.Queue = queue.Queue(maxsize=config.TTS_QUEUE_MAXSIZE)
        self._current_proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._worker = threading.Thread(target=self._run, daemon=True, name="tts-worker")
        self._worker.start()
        self.set_volume(volume)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def speak(self, text: str, lang: str = "auto") -> None:
        """Enqueue text for speech. Non-blocking; drops oldest if full."""
        if not text or not text.strip():
            return
        if lang == "auto":
            lang = detect_language(text)
        try:
            self._queue.put_nowait((text.strip(), lang))
        except queue.Full:
            try:
                self._queue.get_nowait()   # drop oldest
            except queue.Empty:
                pass
            self._queue.put_nowait((text.strip(), lang))

    def stop_current(self) -> None:
        """Interrupt currently playing audio immediately."""
        with self._lock:
            if self._current_proc and self._current_proc.poll() is None:
                self._current_proc.terminate()
                self._current_proc = None
        # Drain any queued items so we don't replay stale speech
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def set_volume(self, pct: int) -> None:
        pct = max(0, min(100, pct))
        self._volume = pct
        try:
            subprocess.run(
                ["amixer", "sset", "Master", f"{pct}%"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except FileNotFoundError:
            pass  # amixer not available (dev machine)

    def drain(self, timeout: float = 5.0) -> None:
        """Block until the audio queue is empty (max timeout seconds)."""
        deadline = time.time() + timeout
        while not self._queue.empty() and time.time() < deadline:
            time.sleep(0.05)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self) -> None:
        while True:
            text, lang = self._queue.get()
            try:
                self._synthesize(text, lang)
            except Exception as exc:
                logger.warning("TTS error: %s", exc)
            finally:
                self._queue.task_done()

    def _synthesize(self, text: str, lang: str) -> None:
        """Choose Piper or espeak-ng and play audio."""
        if lang == "bn":
            self._espeak(text, config.ESPEAK_VOICE_BN)
            return

        # Try Piper for English
        if os.path.isfile(config.PIPER_EN_MODEL) and os.path.isfile(config.PIPER_BINARY):
            self._piper(text, config.PIPER_EN_MODEL)
        else:
            self._espeak(text, config.ESPEAK_VOICE_EN)

    def _piper(self, text: str, model_path: str) -> None:
        """Render via Piper and stream raw PCM to aplay."""
        try:
            piper_proc = subprocess.Popen(
                [config.PIPER_BINARY, "--model", model_path, "--output-raw"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            aplay_proc = subprocess.Popen(
                ["aplay", "-r", "22050", "-f", "S16_LE", "-c", "1", "-"],
                stdin=piper_proc.stdout,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            piper_proc.stdin.write(text.encode("utf-8"))
            piper_proc.stdin.close()

            with self._lock:
                self._current_proc = aplay_proc

            aplay_proc.wait()
            piper_proc.wait()
        except Exception as exc:
            logger.debug("Piper failed (%s), falling back to espeak-ng", exc)
            self._espeak(text, config.ESPEAK_VOICE_EN)
        finally:
            with self._lock:
                self._current_proc = None

    def _espeak(self, text: str, voice: str) -> None:
        try:
            proc = subprocess.Popen(
                [
                    "espeak-ng",
                    "-v", voice,
                    "-s", str(config.ESPEAK_SPEED),
                    "-a", str(int(self._volume * 2)),  # espeak amplitude 0-200
                    text,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            with self._lock:
                self._current_proc = proc
            proc.wait()
        except FileNotFoundError:
            logger.error("espeak-ng not found. Install: sudo apt install espeak-ng")
        finally:
            with self._lock:
                self._current_proc = None


# ---------------------------------------------------------------------------
# CameraManager
# ---------------------------------------------------------------------------

class CameraManager:
    """
    Grabs frames in a background thread into a small ring buffer.
    Callers get the freshest available frame without blocking the main loop.
    """

    def __init__(
        self,
        index: int = config.CAMERA_INDEX,
        width: int = config.CAMERA_WIDTH,
        height: int = config.CAMERA_HEIGHT,
        fps: int = config.CAMERA_FPS,
    ):
        self._index = index
        self._width = width
        self._height = height
        self._fps = fps
        self._cap: cv2.VideoCapture | None = None
        self._buffer: collections.deque = collections.deque(
            maxlen=config.FRAME_BUFFER_SIZE
        )
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._cap = cv2.VideoCapture(self._index)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera at index {self._index}. "
                "Check 'libcamera-hello --nopreview' and camera cable."
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap.set(cv2.CAP_PROP_FPS, self._fps)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # minimise latency

        self._running = True
        self._thread = threading.Thread(
            target=self._capture_loop, daemon=True, name="camera"
        )
        self._thread.start()
        # Wait until first frame arrives
        timeout = time.time() + 5.0
        while not self._buffer and time.time() < timeout:
            time.sleep(0.05)
        if not self._buffer:
            raise RuntimeError("Camera opened but no frames received within 5 s.")
        logger.info(
            "Camera started: %dx%d @ %d fps", self._width, self._height, self._fps
        )

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()

    def get_frame(self):
        """Return the latest frame (numpy array) or None if not ready."""
        with self._lock:
            return self._buffer[-1].copy() if self._buffer else None

    def _capture_loop(self) -> None:
        while self._running:
            ret, frame = self._cap.read()
            if ret:
                with self._lock:
                    self._buffer.append(frame)
            else:
                time.sleep(0.01)
