from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Detection:
    """Result of the auto-detect stage."""
    mask: np.ndarray                     # uint8, 255 = "remove this"
    boxes: list[list[int]] = field(default_factory=list)
    score: float = 0.0                   # 0..1 confidence
    method: str = ""                     # e.g. "temporal-residual", "tile-repeat"
    kind: str = "static"                 # "static" | "moving"  (video only)
    notes: str = ""

    @property
    def found(self) -> bool:
        return bool(self.mask is not None and self.mask.any())

    def to_json(self) -> dict:
        return {
            "found": self.found,
            "score": round(float(self.score), 3),
            "method": self.method,
            "kind": self.kind,
            "boxes": self.boxes,
            "notes": self.notes,
            "coverage": round(float((self.mask > 127).mean()), 5) if self.found else 0.0,
        }


@dataclass
class JobState:
    id: str
    status: str = "queued"        # queued | running | done | error
    progress: float = 0.0
    stage: str = ""
    message: str = ""
    result_url: Optional[str] = None
    result_name: Optional[str] = None
    detection: Optional[dict] = None
