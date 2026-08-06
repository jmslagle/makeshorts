"""Loader for config/criteria.yaml — the tunable definition of "compelling".

This module deliberately contains no rubric prose. Every question, anchor, and
threshold lives in the YAML; this file only gives it a typed shape. If you find
yourself about to write criterion text here, it belongs in the YAML instead.

Both `prompt.py` (which renders the rubric into PROMPT.md) and `lint.py` (which
enforces it) read through this module, so the prompt and the gate can never
drift apart.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

DEFAULT_CRITERIA_PATH = Path("config/criteria.yaml")


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Duration(Strict):
    min: float
    target: float
    max: float

    @model_validator(mode="after")
    def _ordered(self) -> Duration:
        if not self.min <= self.target <= self.max:
            raise ValueError("duration must satisfy min <= target <= max")
        return self


class Gates(Strict):
    """Mechanical rules. The model never adjudicates these."""

    duration: Duration
    snap_to_word_boundaries: bool = True
    boundary_tolerance: float = 0.08
    start_on_sentence_start: bool = True
    end_on_sentence_end: bool = True
    max_internal_silence: float = 1.2
    pad_in: float = 0.12
    pad_out: float = 0.25
    hook_window: float = 3.0
    min_separation: float = 45.0
    max_clips: int = 8
    require_slide_region_when_visually_dependent: bool = True


class Criterion(Strict):
    """One editorial criterion, scored 1-5 with cited evidence.

    `anchors` keys are the integer scores that are actually described (1, 3, 5
    by convention); intermediate values are the gaps between them.
    """

    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    weight: float = Field(gt=0)
    question: str = Field(min_length=1)
    anchors: dict[int, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _anchor_range(self) -> Criterion:
        bad = [k for k in self.anchors if not 1 <= k <= 5]
        if bad:
            raise ValueError(f"criterion {self.id}: anchor scores out of range: {bad}")
        return self


class Scoring(Strict):
    method: Literal["weighted_mean"] = "weighted_mean"
    min_weighted_score: float = 3.8
    # criterion id -> minimum score. Any single failure kills the clip
    # regardless of the weighted mean.
    vetoes: dict[str, int] = Field(default_factory=dict)


class Diversity(Strict):
    max_per_theme: int = 2


class Criteria(Strict):
    version: str
    gates: Gates
    rubric: list[Criterion] = Field(min_length=1)
    scoring: Scoring = Field(default_factory=Scoring)
    diversity: Diversity = Field(default_factory=Diversity)

    # Set by load_criteria(); recorded into clips.json as CriteriaRef so an
    # edit list can always be read against the rubric that produced it.
    sha256: str = ""

    @property
    def ids(self) -> list[str]:
        return [c.id for c in self.rubric]

    def by_id(self, criterion_id: str) -> Criterion:
        for c in self.rubric:
            if c.id == criterion_id:
                return c
        raise KeyError(f"unknown criterion {criterion_id!r}")

    def weighted_score(self, scores: dict[str, int]) -> float:
        """The single source of truth for the arithmetic.

        lint.py recomputes every clip's weighted_score through this function
        rather than trusting the value written in clips.json.
        """
        total_weight = sum(c.weight for c in self.rubric)
        if total_weight == 0:
            raise ValueError("rubric weights sum to zero")
        acc = 0.0
        for c in self.rubric:
            if c.id not in scores:
                raise KeyError(f"missing score for criterion {c.id!r}")
            acc += c.weight * scores[c.id]
        return acc / total_weight

    def failed_vetoes(self, scores: dict[str, int]) -> list[tuple[str, int, int]]:
        """(criterion_id, actual, required) for each veto not met."""
        out = []
        for cid, required in self.scoring.vetoes.items():
            actual = scores.get(cid)
            if actual is not None and actual < required:
                out.append((cid, actual, required))
        return out

    @model_validator(mode="after")
    def _unique_ids_and_valid_vetoes(self) -> Criteria:
        ids = self.ids
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise ValueError(f"duplicate rubric criterion ids: {sorted(dupes)}")
        unknown = set(self.scoring.vetoes) - set(ids)
        if unknown:
            raise ValueError(
                f"scoring.vetoes references criteria not in the rubric: {sorted(unknown)}"
            )
        return self


def sha256_of(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_criteria(path: str | Path = DEFAULT_CRITERIA_PATH) -> Criteria:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"criteria file not found: {p}")
    data = yaml.safe_load(p.read_text()) or {}
    criteria = Criteria.model_validate(data)
    return criteria.model_copy(update={"sha256": sha256_of(p)})
