# ORAVYS&trade; - Voice Deepfake Detection SDK

Detect synthetic and cloned voices in 5 lines of Python. The SDK calls the
hosted ORAVYS API (3000+ forensic engines) by default. An experimental local
DSP probe ships alongside for offline tinkering, but its 11 engines have a
high false-negative rate on modern TTS (HiFi-GAN, Tacotron, ElevenLabs) and
are not suitable for production.

> ORAVYS&trade; is a trademark of Oravys Inc. (Delaware, USA). All rights reserved. INPI 25 5212037.

## Install

```bash
pip install oravys
```

For the experimental local DSP path with extras:
```bash
pip install oravys[local]
```

## Quick Start (recommended)

```python
from oravys import Detector

det = Detector(api_key="oravys_live_...")
result = det.analyze("call_recording.wav")

print(result.verdict)        # "SYNTHETIC" or "AUTHENTIC"
print(result.confidence)     # 0.97
print(result.engines_fired)  # 847
```

The SDK sends `X-API-Key: <your-key>` to `https://oravys.com/api/v1/analyze`.
Get a key at [oravys.com/api-keys](https://oravys.com/api-keys) (free tier
available, paid tiers for production volume).

## Experimental local mode (offline, 11 DSP engines)

```python
from oravys import Detector

det = Detector(mode="experimental")
result = det.analyze("voice.wav")

print(result.is_synthetic)   # True/False
print(result.confidence_pct) # "27.3%"
```

A `UserWarning` is emitted on first use. The local engines miss most modern
synthesis. **Do not use experimental mode to make real decisions** about
voice authenticity. Use it to wire the SDK end-to-end without paying for an
API key, then switch to `mode="api"` once you have one.

`mode="local"` (the v0.1.1 name) is accepted as a deprecated alias and emits
a `DeprecationWarning`.

## CLI

```bash
# API analysis
oravys detect voice.wav --api-key oravys_live_... --json

# Experimental local probe
oravys detect voice.wav --mode experimental --verbose

# Version
oravys version
```

## Experimental local engines

The local mode runs these lightweight CPU-only DSP engines. They are
descriptive, not diagnostic.

| Engine | Domain | What it computes |
|--------|--------|------------------|
| silence_ratio | temporal | Ratio of low-RMS frames |
| zero_crossing_rate | temporal | Signal sign-change rate |
| spectral_flatness | spectral | Geometric/arithmetic mean ratio |
| pitch_stability | prosodic | F0 coefficient of variation |
| harmonic_noise_ratio | spectral | Periodicity vs noise floor |
| spectral_bandwidth | spectral | Spectral centroid variability |
| energy_contour | temporal | Frame-to-frame energy smoothness |
| formant_transition | articulatory | Spectral centroid transitions |
| micro_pause | prosodic | Hesitation events per second |
| breathiness | phonatory | 2-5kHz / 80-1kHz energy stability |
| tremor | neuromotor | 4-12Hz envelope modulation |

The hosted API mode runs 3000+ engines across 16 forensic domains, including
neural-vocoder artifact detection, codec round-trip forensics, and the
proprietary AuthenticityGate + ForensicArbitrator fusion stack. There is no
plan to ship that to PyPI.

## Authentication

API mode sends `X-API-Key: <your-key>` header. Keys are minted in your
ORAVYS dashboard and bound to a tier (free / pro / business / enterprise).
The API response includes a `X-Oravys-Attribution-Required` header so your
client can decide whether to render a "Powered by ORAVYS" badge automatically
(see [oravys.com/branding](https://oravys.com/branding) for ready-made
badges and the tier table).

Free / Pro / Business tiers require attribution. Enterprise tier is white-label.

## Changelog

### 0.1.2 (2026-05-11)

- Fix: `API_BASE` pointed to `https://api.oravys.com/v1` which is not a real
  host (DNS / Cloud Run mapping never existed). Switched to
  `https://oravys.com/api/v1` which is live.
- Fix: API mode auth header was `Authorization: Bearer ...`; the backend
  expects `X-API-Key: ...`. Calls now succeed.
- Change: default mode is now `api` (was `local`). Calling `Detector()`
  without `api_key` raises with instructions instead of silently running
  the underpowered local DSP.
- Change: `local` mode renamed to `experimental` and emits a `UserWarning`
  on first use describing the false-negative risk. `local` still works as
  a deprecated alias.
- Add: surfaced 401 / 429 error responses with actionable messages.
- Add: `Detector.base_url` property for visibility.

### 0.1.1 (2026-05-10)

- Initial release.

## License

Proprietary. Copyright (c) 2026 Oravys Inc. All rights reserved.
