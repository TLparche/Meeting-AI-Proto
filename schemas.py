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


class Band(str, Enum):
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"


class EvidenceStatus(str, Enum):
    VERIFIED = "VERIFIED"
    MIXED = "MIXED"
    UNVERIFIED = "UNVERIFIED"


class InterventionLevel(str, Enum):
    L0 = "L0"
    L1 = "L1"
    L2 = "L2"


class AgendaActive(StrictModel):
    title: str = ""
    status: Literal["ACTIVE", "CLOSING", "CLOSED"] = "ACTIVE"
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)


class AgendaCandidate(StrictModel):
    title: str
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)


class AgendaAnalysis(StrictModel):
    active: AgendaActive = Field(default_factory=AgendaActive)
    candidates: List[AgendaCandidate] = Field(default_factory=list)


class KCore(StrictModel):
    object: List[str] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)
    criteria: List[str] = Field(default_factory=list)


class KFacet(StrictModel):
    options: List[str] = Field(default_factory=list)
    evidence: List[str] = Field(default_factory=list)
    actions: List[str] = Field(default_factory=list)


class Keywords(StrictModel):
    k_core: KCore = Field(default_factory=KCore)
    k_facet: KFacet = Field(default_factory=KFacet)


class DriftScore(StrictModel):
    score: int = Field(ge=0, le=100, default=0)
    band: Band = Band.GREEN
    why: str = ""


class StagnationScore(StrictModel):
    score: int = Field(ge=0, le=100, default=0)
    why: str = ""


class FairTalkSpeaker(StrictModel):
    speaker: str
    p_intent: float = Field(ge=0.0, le=1.0, default=0.0)


class ParticipationScore(StrictModel):
    imbalance: int = Field(ge=0, le=100, default=0)
    fairtalk: List[FairTalkSpeaker] = Field(default_factory=list)


class DPSScore(StrictModel):
    score: int = Field(ge=0, le=100, default=0)
    why: str = ""


class Scores(StrictModel):
    drift: DriftScore = Field(default_factory=DriftScore)
    stagnation: StagnationScore = Field(default_factory=StagnationScore)
    participation: ParticipationScore = Field(default_factory=ParticipationScore)
    dps: DPSScore = Field(default_factory=DPSScore)


class EvidenceClaim(StrictModel):
    claim: str
    verifiability: float = Field(ge=0.0, le=1.0, default=0.0)
    note: str = ""


class EvidenceGate(StrictModel):
    status: EvidenceStatus = EvidenceStatus.UNVERIFIED
    claims: List[EvidenceClaim] = Field(default_factory=list)


class DecisionLock(StrictModel):
    triggered: bool = False
    reason: str = ""


class Intervention(StrictModel):
    level: InterventionLevel = InterventionLevel.L0
    banner_text: str = ""
    decision_lock: DecisionLock = Field(default_factory=DecisionLock)


class ResourceRecommendation(StrictModel):
    title: str
    url: str
    reason: str = ""


class OptionRecommendation(StrictModel):
    option: str
    pros: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)
    evidence_note: str = ""


class Recommendations(StrictModel):
    r1_resources: List[ResourceRecommendation] = Field(default_factory=list)
    r2_options: List[OptionRecommendation] = Field(default_factory=list)


class AnalysisOutput(StrictModel):
    agenda: AgendaAnalysis
    keywords: Keywords
    scores: Scores
    evidence_gate: EvidenceGate
    intervention: Intervention
    recommendations: Recommendations


class TranscriptUtterance(StrictModel):
    speaker: str
    text: str
    timestamp: Optional[str] = None


class AgendaItem(StrictModel):
    title: str
    status: AgendaStatus = AgendaStatus.PROPOSED


class ArtifactKind(str, Enum):
    MEETING_SUMMARY = "meeting_summary"
    DECISION_RESULTS = "decision_results"
    ACTION_ITEMS = "action_items"
    EVIDENCE_LOG = "evidence_log"


class ArtifactOutput(StrictModel):
    kind: ArtifactKind
    title: str
    markdown: str
    bullets: List[str] = Field(default_factory=list)


def validate_analysis_payload(payload: dict) -> AnalysisOutput:
    return AnalysisOutput.model_validate(payload)


def validate_artifact_payload(payload: dict) -> ArtifactOutput:
    return ArtifactOutput.model_validate(payload)


def validation_error_text(err: ValidationError) -> str:
    return err.json(indent=2)
