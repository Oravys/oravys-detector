# ORAVYS Voice Deepfake Detection SDK
# Copyright (c) 2026 Oravys Inc. All rights reserved.
# https://oravys.com
"""
ORAVYS - Voice deepfake detection in 5 lines of Python.
========================================================

3000+ forensic engines. CPU-only. Works on any platform.

Quick start (API mode - recommended)::

    from oravys import Detector

    det = Detector(api_key="your-key")
    result = det.analyze("audio.wav")
    print(result.is_synthetic)    # True/False
    print(result.confidence)      # 0.97
    print(result.engines_fired)   # 847

Quick start (local mode - basic DSP engines)::

    from oravys import Detector

    det = Detector(mode="local")
    result = det.analyze("audio.wav")
    print(result.verdict)         # "SYNTHETIC" or "AUTHENTIC"

Created by Eliot Cohen Bacrie.
Powered by Oravys Inc. (https://oravys.com)
"""

import sys as _sys

from oravys.detector import Detector
from oravys.result import AnalysisResult

__version__ = "0.1.0"
__author__ = "Eliot Cohen Bacrie"
__license__ = "Proprietary"
__url__ = "https://oravys.com"

__all__ = ["Detector", "AnalysisResult"]

_NOTICE_ATTR = "_oravys_notice_shown"

if not getattr(_sys.modules[__name__], _NOTICE_ATTR, False):
    _sys.stderr.write(
        f"ORAVYS v{__version__} - Voice Deepfake Detection SDK"
        f" - https://oravys.com\n"
    )
    setattr(_sys.modules[__name__], _NOTICE_ATTR, True)
