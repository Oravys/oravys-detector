# Copyright (c) 2026 Oravys Inc. All rights reserved.
import io
import struct
import wave

import numpy as np
import pytest

from oravys import Detector, AnalysisResult
from oravys.result import EngineResult


def _make_wav(duration: float = 2.0, sr: int = 16000, freq: float = 440.0) -> bytes:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    samples = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    int_samples = (samples * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(int_samples.tobytes())
    return buf.getvalue()


def _make_noise_wav(duration: float = 2.0, sr: int = 16000) -> bytes:
    rng = np.random.default_rng(42)
    samples = rng.normal(0, 0.3, int(sr * duration)).astype(np.float32)
    int_samples = (np.clip(samples, -1, 1) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(int_samples.tobytes())
    return buf.getvalue()


class TestDetectorInit:
    def test_local_mode_no_key(self):
        det = Detector(mode="local")
        assert det.mode == "local"

    def test_api_mode_requires_key(self):
        with pytest.raises(ValueError, match="api_key is required"):
            Detector(mode="api")

    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="mode must be"):
            Detector(mode="invalid")

    def test_repr(self):
        det = Detector(mode="local")
        assert "local" in repr(det)


class TestLocalAnalysis:
    def test_analyze_sine_wave(self):
        wav = _make_wav(duration=3.0)
        det = Detector(mode="local")
        result = det.analyze(wav)

        assert isinstance(result, AnalysisResult)
        assert result.verdict in (
            "SYNTHETIC",
            "LIKELY_SYNTHETIC",
            "INCONCLUSIVE",
            "AUTHENTIC",
        )
        assert 0 <= result.confidence <= 1
        assert result.engines_fired > 0
        assert result.sample_rate == 16000

    def test_analyze_noise(self):
        wav = _make_noise_wav(duration=3.0)
        det = Detector(mode="local")
        result = det.analyze(wav)

        assert isinstance(result, AnalysisResult)
        assert result.engines_fired > 0

    def test_analyze_from_bytes(self):
        wav = _make_wav()
        det = Detector(mode="local")
        result = det.analyze(wav)
        assert result.engines_total == 11

    def test_analyze_from_path(self, tmp_path):
        p = tmp_path / "test.wav"
        p.write_bytes(_make_wav())
        det = Detector(mode="local")
        result = det.analyze(p)
        assert result.engines_fired > 0

    def test_analyze_from_str_path(self, tmp_path):
        p = tmp_path / "test.wav"
        p.write_bytes(_make_wav())
        det = Detector(mode="local")
        result = det.analyze(str(p))
        assert result.engines_fired > 0

    def test_file_not_found(self):
        det = Detector(mode="local")
        with pytest.raises(FileNotFoundError):
            det.analyze("/nonexistent/audio.wav")

    def test_engine_results_populated(self):
        wav = _make_wav(duration=3.0)
        det = Detector(mode="local")
        result = det.analyze(wav)

        assert len(result.engine_results) > 0
        for er in result.engine_results:
            assert isinstance(er, EngineResult)
            assert er.name
            assert er.domain


class TestAnalysisResult:
    def test_is_authentic(self):
        r = AnalysisResult(is_synthetic=False, confidence=0.1, verdict="AUTHENTIC")
        assert r.is_authentic is True

    def test_confidence_pct(self):
        r = AnalysisResult(is_synthetic=True, confidence=0.973, verdict="SYNTHETIC")
        assert r.confidence_pct == "97.3%"

    def test_summary(self):
        r = AnalysisResult(
            is_synthetic=True,
            confidence=0.85,
            verdict="SYNTHETIC",
            engines_fired=847,
            engines_total=3056,
        )
        s = r.summary()
        assert "SYNTHETIC" in s
        assert "847" in s

    def test_repr(self):
        r = AnalysisResult(
            is_synthetic=True, confidence=0.9, verdict="SYNTHETIC", engines_fired=10
        )
        assert "SYNTHETIC" in repr(r)
