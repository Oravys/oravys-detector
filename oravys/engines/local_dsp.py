# Copyright (c) 2026 Oravys Inc. All rights reserved.
"""Local DSP engines for offline analysis (no network required).

These are lightweight, CPU-only engines that run basic acoustic forensics.
For the full thousands of engines suite, use API mode.
"""

from __future__ import annotations

import io
import time
import wave

import numpy as np

from oravys.result import AnalysisResult, EngineResult


def _load_wav(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    import struct as _struct

    buf = io.BytesIO(audio_bytes)

    try:
        with wave.open(buf, "rb") as wf:
            sr = wf.getframerate()
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            raw = wf.readframes(wf.getnframes())

        if sampwidth == 2:
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sampwidth == 4:
            samples = (
                np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
            )
        else:
            raise ValueError(f"Unsupported sample width: {sampwidth}")
    except wave.Error:
        # IEEE float WAV (format tag 3) -- parse manually
        buf.seek(0)
        riff = buf.read(4)
        if riff != b"RIFF":
            raise ValueError("Not a WAV file")
        buf.read(4)  # file size
        if buf.read(4) != b"WAVE":
            raise ValueError("Not a WAV file")

        sr = 16000
        n_channels = 1
        sampwidth = 4
        raw = b""

        while True:
            chunk_id = buf.read(4)
            if len(chunk_id) < 4:
                break
            chunk_size = _struct.unpack("<I", buf.read(4))[0]
            if chunk_id == b"fmt ":
                fmt_data = buf.read(chunk_size)
                _fmt_tag = _struct.unpack("<H", fmt_data[0:2])[0]
                n_channels = _struct.unpack("<H", fmt_data[2:4])[0]
                sr = _struct.unpack("<I", fmt_data[4:8])[0]
                sampwidth = _struct.unpack("<H", fmt_data[14:16])[0] // 8
            elif chunk_id == b"data":
                raw = buf.read(chunk_size)
                break
            else:
                buf.read(chunk_size)

        if sampwidth == 4:
            samples = np.frombuffer(raw, dtype=np.float32)
        elif sampwidth == 8:
            samples = np.frombuffer(raw, dtype=np.float64).astype(np.float32)
        else:
            raise ValueError(f"Unsupported float WAV sample width: {sampwidth}")

    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)

    return samples, sr


def _engine_silence_ratio(samples: np.ndarray, sr: int) -> EngineResult:
    frame_len = int(0.025 * sr)
    hop = int(0.010 * sr)
    n_frames = max(1, (len(samples) - frame_len) // hop)
    silent = 0
    for i in range(n_frames):
        chunk = samples[i * hop : i * hop + frame_len]
        rms = np.sqrt(np.mean(chunk**2))
        if rms < 0.005:
            silent += 1
    ratio = silent / n_frames
    is_suspicious = ratio < 0.01 or ratio > 0.6
    return EngineResult(
        name="silence_ratio",
        domain="temporal",
        score=ratio,
        label="SUSPICIOUS" if is_suspicious else "NORMAL",
        detail=f"silence_ratio={ratio:.3f}",
    )


def _engine_zero_crossing_rate(samples: np.ndarray, sr: int) -> EngineResult:
    signs = np.sign(samples[:-1]) != np.sign(samples[1:])
    zcr = float(np.mean(signs))
    is_suspicious = zcr < 0.01 or zcr > 0.3
    return EngineResult(
        name="zero_crossing_rate",
        domain="temporal",
        score=zcr,
        label="SUSPICIOUS" if is_suspicious else "NORMAL",
        detail=f"zcr={zcr:.4f}",
    )


def _engine_spectral_flatness(samples: np.ndarray, sr: int) -> EngineResult:
    n_fft = 2048
    hop = 512
    flatness_vals = []
    for i in range(0, len(samples) - n_fft, hop):
        frame = samples[i : i + n_fft]
        spectrum = np.abs(np.fft.rfft(frame * np.hanning(n_fft)))
        spectrum = spectrum[1:]
        spectrum = np.maximum(spectrum, 1e-10)
        geo_mean = np.exp(np.mean(np.log(spectrum)))
        arith_mean = np.mean(spectrum)
        flatness_vals.append(geo_mean / arith_mean if arith_mean > 0 else 0)

    avg_flatness = float(np.mean(flatness_vals)) if flatness_vals else 0.0
    is_suspicious = avg_flatness > 0.5
    return EngineResult(
        name="spectral_flatness",
        domain="spectral",
        score=avg_flatness,
        label="SUSPICIOUS" if is_suspicious else "NORMAL",
        detail=f"flatness={avg_flatness:.4f}",
    )


def _engine_pitch_stability(samples: np.ndarray, sr: int) -> EngineResult:
    frame_len = int(0.030 * sr)
    hop = int(0.010 * sr)
    pitches = []
    for i in range(0, len(samples) - frame_len, hop):
        frame = samples[i : i + frame_len]
        if np.max(np.abs(frame)) < 0.01:
            continue
        corr = np.correlate(frame, frame, mode="full")
        corr = corr[len(corr) // 2 :]
        min_lag = int(sr / 500)
        max_lag = int(sr / 60)
        if max_lag >= len(corr):
            continue
        search = corr[min_lag:max_lag]
        if len(search) == 0:
            continue
        peak = np.argmax(search) + min_lag
        if corr[peak] > 0.3 * corr[0]:
            pitches.append(sr / peak)

    if len(pitches) < 5:
        return EngineResult(
            name="pitch_stability",
            domain="prosodic",
            score=0.0,
            label="INSUFFICIENT_DATA",
        )

    std = float(np.std(pitches))
    mean = float(np.mean(pitches))
    cv = std / mean if mean > 0 else 0
    is_suspicious = cv < 0.02
    return EngineResult(
        name="pitch_stability",
        domain="prosodic",
        score=cv,
        label="SUSPICIOUS" if is_suspicious else "NORMAL",
        detail=f"pitch_cv={cv:.4f}, mean_f0={mean:.1f}Hz",
    )


def _engine_harmonic_noise_ratio(samples: np.ndarray, sr: int) -> EngineResult:
    frame_len = int(0.030 * sr)
    hop = int(0.015 * sr)
    hnr_vals = []
    for i in range(0, len(samples) - frame_len, hop):
        frame = samples[i : i + frame_len]
        if np.max(np.abs(frame)) < 0.01:
            continue
        corr = np.correlate(frame, frame, mode="full")
        corr = corr[len(corr) // 2 :]
        if corr[0] < 1e-10:
            continue
        min_lag = int(sr / 500)
        max_lag = min(int(sr / 60), len(corr) - 1)
        if max_lag <= min_lag:
            continue
        peak = np.max(corr[min_lag:max_lag])
        noise = corr[0] - peak
        if noise > 0:
            hnr_vals.append(10 * np.log10(peak / noise))

    if not hnr_vals:
        return EngineResult(
            name="harmonic_noise_ratio",
            domain="spectral",
            score=0.0,
            label="INSUFFICIENT_DATA",
        )

    avg_hnr = float(np.mean(hnr_vals))
    is_suspicious = avg_hnr > 30
    return EngineResult(
        name="harmonic_noise_ratio",
        domain="spectral",
        score=avg_hnr,
        label="SUSPICIOUS" if is_suspicious else "NORMAL",
        detail=f"hnr={avg_hnr:.1f}dB",
    )


def _engine_spectral_bandwidth(samples: np.ndarray, sr: int) -> EngineResult:
    n_fft = 2048
    hop = 512
    bw_vals = []
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    for i in range(0, len(samples) - n_fft, hop):
        frame = samples[i : i + n_fft]
        mag = np.abs(np.fft.rfft(frame * np.hanning(n_fft)))
        mag_sum = np.sum(mag)
        if mag_sum < 1e-10:
            continue
        centroid = np.sum(freqs * mag) / mag_sum
        bw = np.sqrt(np.sum(mag * (freqs - centroid) ** 2) / mag_sum)
        bw_vals.append(bw)

    if not bw_vals:
        return EngineResult(
            name="spectral_bandwidth",
            domain="spectral",
            score=0.0,
            label="INSUFFICIENT_DATA",
        )

    avg_bw = float(np.mean(bw_vals))
    std_bw = float(np.std(bw_vals))
    cv = std_bw / avg_bw if avg_bw > 0 else 0
    is_suspicious = cv < 0.05
    return EngineResult(
        name="spectral_bandwidth",
        domain="spectral",
        score=cv,
        label="SUSPICIOUS" if is_suspicious else "NORMAL",
        detail=f"bw_cv={cv:.4f}, mean_bw={avg_bw:.0f}Hz",
    )


def _engine_energy_contour(samples: np.ndarray, sr: int) -> EngineResult:
    frame_len = int(0.025 * sr)
    hop = int(0.010 * sr)
    energies = []
    for i in range(0, len(samples) - frame_len, hop):
        chunk = (
            samples[i * hop : i * hop + frame_len]
            if i == 0
            else samples[i : i + frame_len]
        )
        energies.append(float(np.sum(chunk**2)))

    if len(energies) < 10:
        return EngineResult(
            name="energy_contour",
            domain="temporal",
            score=0.0,
            label="INSUFFICIENT_DATA",
        )

    e = np.array(energies)
    e_diff = np.diff(e)
    smoothness = float(np.std(e_diff) / (np.mean(np.abs(e_diff)) + 1e-10))
    is_suspicious = smoothness < 0.5
    return EngineResult(
        name="energy_contour",
        domain="temporal",
        score=smoothness,
        label="SUSPICIOUS" if is_suspicious else "NORMAL",
        detail=f"smoothness={smoothness:.3f}",
    )


def _engine_formant_transition(samples: np.ndarray, sr: int) -> EngineResult:
    n_fft = 1024
    hop = 256
    centroids = []
    for i in range(0, len(samples) - n_fft, hop):
        frame = samples[i : i + n_fft]
        mag = np.abs(np.fft.rfft(frame * np.hanning(n_fft)))
        freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
        mag_sum = np.sum(mag)
        if mag_sum < 1e-10:
            continue
        centroids.append(float(np.sum(freqs * mag) / mag_sum))

    if len(centroids) < 20:
        return EngineResult(
            name="formant_transition",
            domain="articulatory",
            score=0.0,
            label="INSUFFICIENT_DATA",
        )

    c = np.array(centroids)
    transitions = np.abs(np.diff(c))
    mean_trans = float(np.mean(transitions))
    std_trans = float(np.std(transitions))
    ratio = std_trans / mean_trans if mean_trans > 0 else 0
    is_suspicious = ratio < 0.3
    return EngineResult(
        name="formant_transition",
        domain="articulatory",
        score=ratio,
        label="SUSPICIOUS" if is_suspicious else "NORMAL",
        detail=f"transition_cv={ratio:.3f}",
    )


def _engine_micro_pause(samples: np.ndarray, sr: int) -> EngineResult:
    frame_len = int(0.010 * sr)
    hop = int(0.005 * sr)
    is_silent = []
    for i in range(0, len(samples) - frame_len, hop):
        chunk = samples[i : i + frame_len]
        rms = np.sqrt(np.mean(chunk**2))
        is_silent.append(rms < 0.008)

    if len(is_silent) < 50:
        return EngineResult(
            name="micro_pause",
            domain="prosodic",
            score=0.0,
            label="INSUFFICIENT_DATA",
        )

    pauses = []
    current = 0
    for s in is_silent:
        if s:
            current += 1
        else:
            if 2 <= current <= 20:
                pauses.append(current)
            current = 0

    duration_sec = len(samples) / sr
    pause_rate = len(pauses) / duration_sec if duration_sec > 0 else 0
    is_suspicious = pause_rate < 0.5
    return EngineResult(
        name="micro_pause",
        domain="prosodic",
        score=pause_rate,
        label="SUSPICIOUS" if is_suspicious else "NORMAL",
        detail=f"pauses/sec={pause_rate:.2f}, count={len(pauses)}",
    )


def _engine_breathiness(samples: np.ndarray, sr: int) -> EngineResult:
    n_fft = 2048
    hop = 512
    ratios = []
    for i in range(0, len(samples) - n_fft, hop):
        frame = samples[i : i + n_fft]
        mag = np.abs(np.fft.rfft(frame * np.hanning(n_fft)))
        freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
        voiced = mag[(freqs >= 80) & (freqs <= 1000)]
        breathy = mag[(freqs >= 2000) & (freqs <= 5000)]
        if np.sum(voiced) < 1e-10:
            continue
        ratios.append(float(np.sum(breathy) / np.sum(voiced)))

    if not ratios:
        return EngineResult(
            name="breathiness",
            domain="phonatory",
            score=0.0,
            label="INSUFFICIENT_DATA",
        )

    avg_ratio = float(np.mean(ratios))
    std_ratio = float(np.std(ratios))
    is_suspicious = std_ratio < 0.02
    return EngineResult(
        name="breathiness",
        domain="phonatory",
        score=std_ratio,
        label="SUSPICIOUS" if is_suspicious else "NORMAL",
        detail=f"breath_var={std_ratio:.4f}, mean={avg_ratio:.3f}",
    )


def _engine_tremor(samples: np.ndarray, sr: int) -> EngineResult:
    frame_len = int(0.050 * sr)
    hop = int(0.025 * sr)
    amplitudes = []
    for i in range(0, len(samples) - frame_len, hop):
        chunk = samples[i : i + frame_len]
        amplitudes.append(float(np.sqrt(np.mean(chunk**2))))

    if len(amplitudes) < 40:
        return EngineResult(
            name="tremor",
            domain="neuromotor",
            score=0.0,
            label="INSUFFICIENT_DATA",
        )

    a = np.array(amplitudes)
    a = a - np.mean(a)
    fft_amp = np.abs(np.fft.rfft(a))
    freqs = np.fft.rfftfreq(len(a), hop / sr)
    tremor_band = fft_amp[(freqs >= 4) & (freqs <= 12)]
    total_energy = np.sum(fft_amp**2)
    tremor_energy = np.sum(tremor_band**2)
    ratio = float(tremor_energy / total_energy) if total_energy > 0 else 0

    has_tremor = ratio > 0.05
    return EngineResult(
        name="tremor",
        domain="neuromotor",
        score=ratio,
        label="NORMAL" if has_tremor else "SUSPICIOUS",
        detail=f"tremor_ratio={ratio:.4f}",
    )


_LOCAL_ENGINES = [
    _engine_silence_ratio,
    _engine_zero_crossing_rate,
    _engine_spectral_flatness,
    _engine_pitch_stability,
    _engine_harmonic_noise_ratio,
    _engine_spectral_bandwidth,
    _engine_energy_contour,
    _engine_formant_transition,
    _engine_micro_pause,
    _engine_breathiness,
    _engine_tremor,
]


def run_local_engines(audio_bytes: bytes) -> AnalysisResult:
    t0 = time.time()
    samples, sr = _load_wav(audio_bytes)

    results = []
    for engine_fn in _LOCAL_ENGINES:
        try:
            results.append(engine_fn(samples, sr))
        except Exception:
            pass

    suspicious = sum(1 for r in results if r.label == "SUSPICIOUS")
    total_scored = sum(1 for r in results if r.label in ("SUSPICIOUS", "NORMAL"))

    if total_scored == 0:
        confidence = 0.0
        is_synthetic = False
    else:
        confidence = suspicious / total_scored
        is_synthetic = confidence > 0.5

    if confidence > 0.7:
        verdict = "SYNTHETIC"
    elif confidence > 0.5:
        verdict = "LIKELY_SYNTHETIC"
    elif confidence > 0.3:
        verdict = "INCONCLUSIVE"
    else:
        verdict = "AUTHENTIC"

    return AnalysisResult(
        is_synthetic=is_synthetic,
        confidence=confidence,
        verdict=verdict,
        engines_fired=len(results),
        engines_total=len(_LOCAL_ENGINES),
        duration_seconds=time.time() - t0,
        sample_rate=sr,
        engine_results=results,
    )
