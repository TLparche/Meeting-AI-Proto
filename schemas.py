from __future__ import annotations

from enum import Enum
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AgendaStatus(str, Enum):
    PROPOSED = "PROPOSED"
    ACTIVE = "ACTIVE"
    CLOSING = "CLOSING"
    CLOSED = "CLOSED"


class AgendaActive(StrictModel):
    title: str = ""
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)


class AgendaCandidate(StrictModel):
    title: str
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)


class AgendaAnalysis(StrictModel):
    active: AgendaActive = Field(default_factory=AgendaActive)
    candidates: List[AgendaCandidate] = Field(default_factory=list)


class AgendaEvidenceLog(StrictModel):
    speaker: str = ""
    timestamp: str = ""
    quote: str = ""
    why: str = ""


class AgendaActionItem(StrictModel):
    item: str = ""
    owner: str = ""
    due: str = ""
    reasons: List[AgendaEvidenceLog] = Field(default_factory=list)


class AgendaDecisionResult(StrictModel):
    opinions: List[str] = Field(default_factory=list)
    conclusion: str = ""


class AgendaOutcome(StrictModel):
    agenda_title: str = ""
    key_utterances: List[str] = Field(default_factory=list)
    summary: str = ""
    agenda_keywords: List[str] = Field(default_factory=list)
    decision_results: List[AgendaDecisionResult] = Field(default_factory=list)
    action_items: List[AgendaActionItem] = Field(default_factory=list)


class EvidenceClaim(StrictModel):
    claim: str
    verifiability: float = Field(ge=0.0, le=1.0, default=0.0)
    note: str = ""


class EvidenceGate(StrictModel):
    claims: List[EvidenceClaim] = Field(default_factory=list)


class AnalysisOutput(StrictModel):
    agenda: AgendaAnalysis
    agenda_outcomes: List[AgendaOutcome] = Field(default_factory=list)
    evidence_gate: EvidenceGate = Field(default_factory=EvidenceGate)


class TranscriptUtterance(StrictModel):
    speaker: str
    text: str
    timestamp: Optional[str] = None


def validate_analysis_payload(payload: dict) -> AnalysisOutput:
    return AnalysisOutput.model_validate(payload)


def validation_error_text(err: ValidationError) -> str:
    return err.json(indent=2)
