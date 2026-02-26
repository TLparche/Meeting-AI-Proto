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


class KeywordTaxonomy(StrictModel):
    K1_OBJECT: str = "무엇을 결정하는가(결정 대상)"
    K2_OPTION: str = "어떤 대안들이 있는가"
    K3_CONSTRAINT: str = "제한/조건(예산/시간/정책 등)"
    K4_CRITERION: str = "평가 기준(성능/비용/리스크 등)"
    K5_EVIDENCE: str = "근거(출처/데이터/사실 주장)"
    K6_ACTION: str = "누가/언제/무엇을 할지(담당/기한)"


class KCore(StrictModel):
    object: List[str] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)
    criteria: List[str] = Field(default_factory=list)


class KFacet(StrictModel):
    options: List[str] = Field(default_factory=list)
    evidence: List[str] = Field(default_factory=list)
    actions: List[str] = Field(default_factory=list)


class KeywordItem(StrictModel):
    keyword: str
    type: Literal["K1_OBJECT", "K2_OPTION", "K3_CONSTRAINT", "K4_CRITERION", "K5_EVIDENCE", "K6_ACTION"]
    score: float = Field(ge=0.0, le=1.0, default=0.0)
    first_seen: str = ""
    frequency: int = Field(ge=1, default=1)
    decision_value: float = Field(ge=0.0, le=1.0, default=0.0)
    evidence_boost: float = Field(ge=0.0, le=1.0, default=0.0)
    is_core: bool = False


class KeywordCandidate(StrictModel):
    keyword: str
    frequency: int = Field(ge=1, default=1)
    first_seen: str = ""


class KeywordClassification(StrictModel):
    keyword: str
    type: Literal["K1_OBJECT", "K2_OPTION", "K3_CONSTRAINT", "K4_CRITERION", "K5_EVIDENCE", "K6_ACTION"]


class KeywordScoring(StrictModel):
    keyword: str
    decision_value: float = Field(ge=0.0, le=1.0, default=0.0)
    evidence_boost: float = Field(ge=0.0, le=1.0, default=0.0)
    score: float = Field(ge=0.0, le=1.0, default=0.0)


class KeywordFinalSelection(StrictModel):
    k_core_required: List[str] = Field(default_factory=list)
    k_facet_target: str = "3~8"
    selected_core: List[str] = Field(default_factory=list)
    selected_facet: List[str] = Field(default_factory=list)
    diversity_boost_applied: bool = False


class KeywordPipeline(StrictModel):
    candidates: List[KeywordCandidate] = Field(default_factory=list)
    classification: List[KeywordClassification] = Field(default_factory=list)
    scoring: List[KeywordScoring] = Field(default_factory=list)
    final_selection: KeywordFinalSelection = Field(default_factory=KeywordFinalSelection)


class KeywordSummary(StrictModel):
    object_focus: str = ""
    core_count: int = Field(ge=0, default=0)
    facet_count: int = Field(ge=0, default=0)


class Keywords(StrictModel):
    taxonomy: KeywordTaxonomy = Field(default_factory=KeywordTaxonomy)
    k_core: KCore = Field(default_factory=KCore)
    k_facet: KFacet = Field(default_factory=KFacet)
    items: List[KeywordItem] = Field(default_factory=list)
    pipeline: KeywordPipeline = Field(default_factory=KeywordPipeline)
    summary: KeywordSummary = Field(default_factory=KeywordSummary)


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
    score: float = Field(ge=0.0, le=1.0, default=0.0)
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
    decision: str = ""
    opinions: List[str] = Field(default_factory=list)
    conclusion: str = ""


class AgendaOutcome(StrictModel):
    agenda_title: str = ""
    agenda_state: Literal["PROPOSED", "ACTIVE", "CLOSING", "CLOSED"] = "PROPOSED"
    flow_type: str = ""
    key_utterances: List[str] = Field(default_factory=list)
    summary: str = ""
    decision_results: List[AgendaDecisionResult] = Field(default_factory=list)
    action_items: List[AgendaActionItem] = Field(default_factory=list)


class AnalysisOutput(StrictModel):
    agenda: AgendaAnalysis
    agenda_outcomes: List[AgendaOutcome] = Field(default_factory=list)
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
