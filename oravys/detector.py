# Copyright (c) 2026 Oravys Inc. All rights reserved.
"""Main Detector class - unified interface for API and experimental local analysis."""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import BinaryIO

from oravys.result import AnalysisResult

API_BASE = "https://oravys.com/api/v1"

_EXPERIMENTAL_WARNING_ISSUED = False


class Detector:
    """Voice deepfake detector with thousands of forensic engines.

    Parameters
    ----------
    api_key : str, optional
        API key for cloud analysis (full engine suite).
        Get yours at https://oravys.com/api-keys
    mode : str
        "api" (default, recommended) uses the ORAVYS cloud with 3000+ engines.
        "experimental" runs ~11 basic DSP engines locally with no network.
          This mode has a high false-negative rate and is provided only to
          let integrators probe the SDK without a key. Do not use in
          production.
        "local" is accepted as a deprecated alias for "experimental".
    base_url : str, optional
        Override the API endpoint (for enterprise on-prem deployments).

    Examples
    --------
    >>> det = Detector(api_key="oravys_live_...")
    >>> result = det.analyze("call_recording.wav")
    >>> print(result.verdict)
    SYNTHETIC
    """

    def __init__(
        self,
        api_key: str | None = None,
        mode: str = "api",
        base_url: str | None = None,
    ):
        # Backwards-compat: 'local' was the v0.1.1 name. Map to experimental
        # and emit a one-time DeprecationWarning so existing scripts keep
        # running while users get nudged.
        if mode == "local":
            warnings.warn(
                "mode='local' is deprecated, use mode='experimental' instead "
                "(or mode='api' with an api_key for real detection).",
                DeprecationWarning,
                stacklevel=2,
            )
            mode = "experimental"

        if mode not in ("api", "experimental"):
            raise ValueError(
                f"mode must be 'api' or 'experimental', got {mode!r}"
            )

        self._mode = mode
        self._api_key = api_key
        self._base_url = (base_url or API_BASE).rstrip("/")

        if mode == "api" and not api_key:
            raise ValueError(
                "api_key is required for API mode. "
                "Get yours at https://oravys.com/api-keys "
                "or use mode='experimental' for offline DSP probing "
                "(not suitable for real detection)."
            )

        if mode == "experimental":
            global _EXPERIMENTAL_WARNING_ISSUED
            if not _EXPERIMENTAL_WARNING_ISSUED:
                warnings.warn(
                    "ORAVYS experimental mode runs 11 simple DSP engines "
                    "locally. It has a high false-negative rate against "
                    "modern TTS / vocoder-based deepfakes (HiFi-GAN, "
                    "Tacotron, ElevenLabs, etc.) and is not suitable for "
                    "production use. Use mode='api' with an api_key for "
                    "the full 3000+ engine pipeline.",
                    UserWarning,
                    stacklevel=2,
                )
                _EXPERIMENTAL_WARNING_ISSUED = True

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

    @property
    def base_url(self) -> str:
        """Effective base URL used for API calls (resolved override or default)."""
        return self._base_url

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

        headers = {"X-API-Key": self._api_key or ""}
        files = {"audio": ("audio.wav", audio_bytes, "audio/wav")}
        data = {"tier": tier}
        if engines:
            data["engines"] = ",".join(engines)

        t0 = time.monotonic()
        with httpx.Client(timeout=timeout, follow_redirects=False) as client:
            resp = client.post(
                f"{self._base_url}/analyze",
                headers=headers,
                files=files,
                data=data,
            )
            if resp.status_code == 401:
                raise PermissionError(
                    "ORAVYS API returned 401 Unauthorized. Check that "
                    "X-API-Key is set and that the key has not been "
                    "revoked. Mint a new key at https://oravys.com/api-keys."
                )
            if resp.status_code == 429:
                raise RuntimeError(
                    "ORAVYS API returned 429 Too Many Requests. Daily "
                    "quota reached. Upgrade tier at https://oravys.com/pricing."
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
