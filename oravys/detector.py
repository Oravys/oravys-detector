# Copyright (c) 2026 Oravys Inc. All rights reserved.
"""Main Detector class - unified interface for API and local analysis."""

from __future__ import annotations

import time
from pathlib import Path
from typing import BinaryIO

from oravys.result import AnalysisResult

API_BASE = "https://api.oravys.com/v1"


class Detector:
    """Voice deepfake detector with 3000+ forensic engines.

    Parameters
    ----------
    api_key : str, optional
        API key for cloud analysis (full engine suite).
        Get yours at https://app.oravys.com/api-keys
    mode : str
        "api" (default) uses the ORAVYS cloud with 3000+ engines.
        "local" runs basic DSP engines locally (no network, ~12 engines).
    base_url : str, optional
        Override the API endpoint (for enterprise on-prem deployments).

    Examples
    --------
    >>> det = Detector(api_key="sk-oravys-...")
    >>> result = det.analyze("call_recording.wav")
    >>> print(result.verdict)
    SYNTHETIC

    >>> det = Detector(mode="local")
    >>> result = det.analyze("voice.wav")
    >>> print(result.is_synthetic)
    False
    """

    def __init__(
        self,
        api_key: str | None = None,
        mode: str = "api",
        base_url: str | None = None,
    ):
        if mode not in ("api", "local"):
            raise ValueError(f"mode must be 'api' or 'local', got {mode!r}")

        self._mode = mode
        self._api_key = api_key
        self._base_url = (base_url or API_BASE).rstrip("/")

        if mode == "api" and not api_key:
            raise ValueError(
                "api_key is required for API mode. "
                "Get yours at https://app.oravys.com/api-keys "
                "or use mode='local' for offline analysis."
            )

    @property
    def mode(self) -> str:
        return self._mode

    def analyze(
        self,
        audio: str | Path | bytes | BinaryIO,
        *,
        tier: str = "professional",
        engines: list[str] | None = None,
        timeout: float = 120.0,
    ) -> AnalysisResult:
        """Analyze an audio file for synthetic voice detection.

        Parameters
        ----------
        audio : str, Path, bytes, or file-like
            Audio input. Accepts a file path, raw bytes, or a readable
            file object. Supports WAV, MP3, FLAC, M4A, OGG, WEBM.
        tier : str
            Analysis tier (free/explorer/professional/enterprise).
            Higher tiers unlock more engines.
        engines : list[str], optional
            Restrict to specific engine names. None = all available.
        timeout : float
            Request timeout in seconds (API mode only).

        Returns
        -------
        AnalysisResult
            Structured result with verdict, confidence, and per-engine scores.
        """
        audio_bytes = self._resolve_audio(audio)

        if self._mode == "api":
            return self._analyze_api(audio_bytes, tier, engines, timeout)
        return self._analyze_local(audio_bytes)

    def _resolve_audio(self, audio: str | Path | bytes | BinaryIO) -> bytes:
        if isinstance(audio, (str, Path)):
            p = Path(audio)
            if not p.exists():
                raise FileNotFoundError(f"Audio file not found: {p}")
            if p.stat().st_size > 100 * 1024 * 1024:
                raise ValueError("File exceeds 100 MB limit")
            return p.read_bytes()
        if isinstance(audio, bytes):
            return audio
        if hasattr(audio, "read"):
            return audio.read()
        raise TypeError(f"Unsupported audio type: {type(audio)}")

    def _analyze_api(
        self,
        audio_bytes: bytes,
        tier: str,
        engines: list[str] | None,
        timeout: float,
    ) -> AnalysisResult:
        import httpx

        headers = {"Authorization": f"Bearer {self._api_key}"}
        files = {"audio": ("audio.wav", audio_bytes, "audio/wav")}
        data = {"tier": tier}
        if engines:
            data["engines"] = ",".join(engines)

        t0 = time.monotonic()
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(
                f"{self._base_url}/analyze",
                headers=headers,
                files=files,
                data=data,
            )
            resp.raise_for_status()

        elapsed = time.monotonic() - t0
        body = resp.json()

        return AnalysisResult(
            is_synthetic=body.get("is_synthetic", False),
            confidence=body.get("confidence", 0.0),
            verdict=body.get("verdict", "UNKNOWN"),
            engines_fired=body.get("engines_fired", 0),
            engines_total=body.get("engines_total", 0),
            duration_seconds=elapsed,
            sample_rate=body.get("sample_rate", 0),
            raw=body,
        )

    def _analyze_local(self, audio_bytes: bytes) -> AnalysisResult:
        from oravys.engines.local_dsp import run_local_engines

        return run_local_engines(audio_bytes)

    def __repr__(self) -> str:
        return f"Detector(mode={self._mode!r})"
