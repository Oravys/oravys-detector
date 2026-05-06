# Copyright (c) 2026 Oravys Inc. All rights reserved.
"""Analysis result container."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EngineResult:
    name: str
    domain: str
    score: float
    label: str
    detail: str = ""


@dataclass
class AnalysisResult:
    is_synthetic: bool
    confidence: float
    verdict: str
    engines_fired: int = 0
    engines_total: int = 0
    duration_seconds: float = 0.0
    sample_rate: int = 0
    engine_results: list[EngineResult] = field(default_factory=list)
    raw: dict = field(default_factory=dict)

    @property
    def is_authentic(self) -> bool:
        return not self.is_synthetic

    @property
    def confidence_pct(self) -> str:
        return f"{self.confidence * 100:.1f}%"

    def summary(self) -> str:
        return (
            f"Verdict: {self.verdict} "
            f"(confidence={self.confidence_pct}, "
            f"engines={self.engines_fired}/{self.engines_total})"
        )

    def __repr__(self) -> str:
        return (
            f"AnalysisResult(verdict={self.verdict!r}, "
            f"confidence={self.confidence:.3f}, "
            f"engines_fired={self.engines_fired})"
        )
