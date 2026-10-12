from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class GoalOption(BaseModel):
    id: str
    label: str


class GoalNextStep(BaseModel):
    stepId: str
    prompt: str
    control: str
    options: list[GoalOption] = Field(default_factory=list)
    allowFreeText: bool = True
    findings: list[dict[str, Any]] = Field(default_factory=list)
    milestones: list[dict[str, Any]] = Field(default_factory=list)


class GoalClarifyResult(BaseModel):
    title: str = ""
    goalType: str = ""
    description: str = ""
    category: str = ""
    audience: str = ""
    targetMetric: str = ""
    timeframe: str = ""
    successCriteria: str = ""
    availableHoursPerWeek: float | None = None
    constraints: list[str] = Field(default_factory=list)
    missingMaterial: list[str] = Field(default_factory=list)
    measurementType: str = ""
    completeness: int = 0
    needsSpaces: bool = False
    readyForBaseline: bool = False
    currentRole: str = ""
    startingLevel: str = ""
    geography: str = ""
    nextStep: GoalNextStep


class BaselineFinding(BaseModel):
    text: str
    sourceType: str = ""
    sourceId: str = ""
    category: str = "OUTPUT"


class GoalBaselineResult(BaseModel):
    summary: str
    findings: list[BaselineFinding] = Field(default_factory=list)


class GoalMilestone(BaseModel):
    title: str
    successCriteria: list[str] = Field(default_factory=list)
    dueLabel: str = ""
    dependencies: list[str] = Field(default_factory=list)


class GoalPlanResult(BaseModel):
    """Legacy thin plan shape — still accepted for non-expert single-pass paths."""

    measurementType: str = "MILESTONE"
    summary: str
    milestones: list[GoalMilestone] = Field(default_factory=list)
    baselineValue: float | None = None
    targetValue: float | None = None
    measurementDirection: str = ""


class SuccessMetric(BaseModel):
    label: str
    target: str


class SuccessDefinition(BaseModel):
    deadlineLabel: str = ""
    metrics: list[SuccessMetric] = Field(default_factory=list)


class PlanPhase(BaseModel):
    id: str = ""
    title: str
    windowLabel: str = ""
    outcome: str = ""


class DayTask(BaseModel):
    id: str = ""
    text: str
    done: bool = False
    why: str = ""
    definitionOfDone: str = ""
    note: str = ""


class PlanDay(BaseModel):
    dayIndex: int = 1
    dateLabel: str = ""
    title: str
    focus: str = ""
    hours: float | None = None
    tasks: list[DayTask] = Field(default_factory=list)
    definitionOfDone: str = ""


class PlanStrategy(BaseModel):
    id: str = ""
    title: str
    body: str
    measurableGoal: str = ""
    why: str = ""
    proofPoints: list[str] = Field(default_factory=list)
    effort: str = ""
    tags: list[str] = Field(default_factory=list)


class PriorityRecommendation(BaseModel):
    id: str = ""
    title: str
    reason: str
    priority: str = "HIGH"
    howSteps: list[str] = Field(default_factory=list)
    measurableGoal: str = ""
    milestoneTitle: str = ""
    suggestedDurationMinutes: int | None = None
    relatedStrategyId: str = ""


class ChecklistItem(BaseModel):
    id: str = ""
    text: str
    done: bool = False


class PlanChecklist(BaseModel):
    id: str = ""
    title: str
    items: list[ChecklistItem] = Field(default_factory=list)


class ScorecardRow(BaseModel):
    area: str
    criteria: str


class OpenQuestion(BaseModel):
    id: str = ""
    prompt: str
    options: list[GoalOption] = Field(default_factory=list)


class GoalStrategistResult(BaseModel):
    mission: str
    recommendation: str
    positioningHeadline: str = ""
    positioningBody: str = ""
    successDefinition: SuccessDefinition = Field(default_factory=SuccessDefinition)
    assumptions: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    mvpScope: list[str] = Field(default_factory=list)
    postpone: list[str] = Field(default_factory=list)
    audience: str = ""
    capacityNote: str = ""
    openQuestions: list[OpenQuestion] = Field(default_factory=list)


class GoalAdvisorResult(BaseModel):
    """Domain-adaptive growth / acquisition / operating strategies + priority recommendations."""

    positioningHeadline: str = ""
    positioningBody: str = ""
    strategies: list[PlanStrategy] = Field(default_factory=list)
    priorityRecommendations: list[PriorityRecommendation] = Field(default_factory=list)
    immediateNextStep: str = ""
    funnelTargets: list[str] = Field(default_factory=list)


class GoalExecutionResult(BaseModel):
    summary: str = ""
    measurementType: str = "MILESTONE"
    phases: list[PlanPhase] = Field(default_factory=list)
    days: list[PlanDay] = Field(default_factory=list)
    strategies: list[PlanStrategy] = Field(default_factory=list)
    priorityRecommendations: list[PriorityRecommendation] = Field(default_factory=list)
    checklists: list[PlanChecklist] = Field(default_factory=list)
    scorecard: list[ScorecardRow] = Field(default_factory=list)
    milestones: list[GoalMilestone] = Field(default_factory=list)
    baselineValue: float | None = None
    targetValue: float | None = None
    measurementDirection: str = ""


class GoalCriticResult(BaseModel):
    accepted: bool = False
    issues: list[str] = Field(default_factory=list)
    rewriteSummary: str = ""
    rewriteRecommendation: str = ""
    repairedDays: list[PlanDay] = Field(default_factory=list)
    repairedStrategies: list[PlanStrategy] = Field(default_factory=list)
    repairedPriorityRecommendations: list[PriorityRecommendation] = Field(default_factory=list)
    repairedChecklists: list[PlanChecklist] = Field(default_factory=list)
    repairedScorecard: list[ScorecardRow] = Field(default_factory=list)
    repairedMilestones: list[GoalMilestone] = Field(default_factory=list)


DOMAIN_CATEGORIES = (
    "CAREER",
    "EDUCATION",
    "BUSINESS",
    "TECHNOLOGY",
    "FINANCIAL",
    "HEALTH",
    "PRODUCTIVITY",
    "CREATIVE",
    "RELATIONSHIPS",
    "TRAVEL",
    "OTHER",
)


class LabeledAssumption(BaseModel):
    text: str
    reason: str = ""
    confirmed: bool = False


class GoalUnderstandingResult(BaseModel):
    primaryGoal: str = ""
    domain: str = "OTHER"
    subdomain: str = ""
    targetOutcome: str = ""
    intent: str = ""
    geography: str = ""
    startingLevel: str = "UNKNOWN"
    constraints: list[str] = Field(default_factory=list)
    prerequisites: list[str] = Field(default_factory=list)
    ambiguities: list[str] = Field(default_factory=list)
    researchNeeded: bool = True
    goalType: str = "other"
    assumptions: list[LabeledAssumption] = Field(default_factory=list)
    confidence: float = 0.5


class InterviewQuestion(BaseModel):
    stepId: str
    prompt: str
    control: str = "text"
    field: str = ""
    options: list[GoalOption] = Field(default_factory=list)
    leverage: str = ""


class GoalInterviewResult(BaseModel):
    questions: list[InterviewQuestion] = Field(default_factory=list)
    readyForSpaces: bool = False
    missingHighLeverage: list[str] = Field(default_factory=list)


class ResearchPlanResult(BaseModel):
    researchNeeded: bool = False
    reason: str = ""
    objective: str = ""
    searchQueries: list[str] = Field(default_factory=list)
    mode: str = "fast"
    location: str = ""
    reuseCached: bool = False
    maxResults: int = 8


class ExtractedRequirement(BaseModel):
    text: str
    kind: str = "FACT"
    sourceUrl: str = ""
    sourceTitle: str = ""
    confidence: float = 0.5
    geographicApplicability: str = ""
    freshness: str = "stable"


class EvidenceExtractionResult(BaseModel):
    requirements: list[ExtractedRequirement] = Field(default_factory=list)
    disagreements: list[str] = Field(default_factory=list)
    insufficientEvidence: bool = False
    notes: str = ""


class GoalGapItem(BaseModel):
    requirement: str
    status: str = "UNKNOWN"
    evidence: str = ""
    impact: str = "MEDIUM"
    sourceUrls: list[str] = Field(default_factory=list)


class GoalGapResult(BaseModel):
    satisfied: list[GoalGapItem] = Field(default_factory=list)
    partial: list[GoalGapItem] = Field(default_factory=list)
    missing: list[GoalGapItem] = Field(default_factory=list)
    unknown: list[GoalGapItem] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    highestImpact: list[str] = Field(default_factory=list)
    pathwaySummary: str = ""
    provisional: bool = False


class ValidationIssue(BaseModel):
    code: str
    severity: str = "error"
    message: str
    target: str = ""


class GoalValidationResult(BaseModel):
    accepted: bool = False
    provisional: bool = False
    issues: list[ValidationIssue] = Field(default_factory=list)
    repaired: bool = False

