# ORAVYS - Voice Deepfake Detection SDK

Detect synthetic and cloned voices in 5 lines of Python. 3000+ forensic engines.

## Install

```bash
pip install oravys
```

For local analysis with advanced DSP:
```bash
pip install oravys[local]
```

## Quick Start

### API Mode (3000+ engines)

```python
from oravys import Detector

det = Detector(api_key="sk-oravys-...")
result = det.analyze("call_recording.wav")

print(result.verdict)        # "SYNTHETIC" or "AUTHENTIC"
print(result.confidence)     # 0.97
print(result.engines_fired)  # 847
```

### Local Mode (offline, ~11 DSP engines)

```python
from oravys import Detector

det = Detector(mode="local")
result = det.analyze("voice.wav")

print(result.is_synthetic)   # True/False
print(result.confidence_pct) # "72.7%"
```

## CLI

```bash
# Local analysis
oravys detect voice.wav --verbose

# API analysis
oravys detect voice.wav --api-key sk-oravys-... --json

# Version
oravys version
```

## Local Engines

The local mode runs these lightweight DSP engines (CPU-only, no GPU needed):

| Engine | Domain | What it detects |
|--------|--------|-----------------|
| silence_ratio | temporal | Unnatural silence patterns |
| zero_crossing_rate | temporal | Synthetic signal characteristics |
| spectral_flatness | spectral | Noise-like spectrum (codec artifacts) |
| pitch_stability | prosodic | Unnaturally stable pitch (TTS signature) |
| harmonic_noise_ratio | spectral | Overly clean harmonics |
| spectral_bandwidth | spectral | Bandwidth consistency anomalies |
| energy_contour | temporal | Robotic energy patterns |
| formant_transition | articulatory | Missing natural articulation |
| micro_pause | prosodic | Absence of natural hesitations |
| breathiness | phonatory | Missing breath variation |
| tremor | neuromotor | Missing natural vocal tremor |

The full API mode runs 3000+ engines across 16 forensic domains.

## Get an API Key

Sign up at [app.oravys.com](https://app.oravys.com) to get your API key.

## License

Proprietary. Copyright (c) 2026 Oravys Inc. All rights reserved.
