from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
    )


class ExtractionMethod(str, Enum):
    FIRECRAWL = "firecrawl"
    REQUESTS_FALLBACK = "requests_fallback"
    NONE = "none"


class PipelineStatus(str, Enum):
    INITIALIZED = "initialized"
    SCRAPING = "scraping"
    ANALYZING = "analyzing"
    WRITING = "writing"
    RETRYING = "retrying"
    COMPLETED = "completed"
    FAILED = "failed"


class ErrorCategory(str, Enum):
    VALIDATION = "validation"
    NETWORK = "network"
    ACCESS = "access"
    EXTRACTION = "extraction"
    ANALYSIS = "analysis"
    REPORTING = "reporting"
    UNKNOWN = "unknown"


class PipelineError(StrictModel):
    category: ErrorCategory
    message: str = Field(min_length=1, max_length=2000)
    source_url: str | None = None
    recoverable: bool = True
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class PricePoint(StrictModel):
    label: str = Field(min_length=1, max_length=250)
    amount: str = Field(min_length=1, max_length=150)
    billing_period: str | None = Field(default=None, max_length=100)
    evidence: str = Field(min_length=1, max_length=500)


class CompetitorProfile(StrictModel):
    name: str = Field(min_length=1, max_length=200)
    url: HttpUrl
    title: str | None = Field(default=None, max_length=500)
    description: str | None = Field(default=None, max_length=2000)
    pricing: list[PricePoint] = Field(default_factory=list)
    features: list[str] = Field(default_factory=list, max_length=50)
    differentiators: list[str] = Field(default_factory=list, max_length=30)
    source_excerpt: str = Field(default="", max_length=8000)
    extraction_method: ExtractionMethod = ExtractionMethod.NONE
    extraction_confidence: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    http_status_code: int | None = Field(default=None, ge=100, le=599)

    @field_validator("features", "differentiators")
    @classmethod
    def deduplicate_and_limit(cls, values: list[str]) -> list[str]:
        result: list[str] = []
        observed: set[str] = set()

        for item in values:
            normalized = " ".join(item.split())
            key = normalized.lower()
            if normalized and key not in observed:
                observed.add(key)
                result.append(normalized)

        return result


class MarketGap(StrictModel):
    title: str = Field(min_length=1, max_length=250)
    finding: str = Field(min_length=1, max_length=1500)
    opportunity: str = Field(min_length=1, max_length=1500)
    evidence_urls: list[str] = Field(default_factory=list, max_length=20)
    priority: Literal["high", "medium", "low"]
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]


class CompetitiveAnalysis(StrictModel):
    market_summary: str = Field(min_length=1, max_length=3000)
    common_features: list[str] = Field(default_factory=list, max_length=50)
    unique_features_by_competitor: dict[str, list[str]] = Field(default_factory=dict)
    pricing_observations: list[str] = Field(default_factory=list, max_length=50)
    market_gaps: list[MarketGap] = Field(default_factory=list, max_length=20)
    recommended_positioning: str = Field(min_length=1, max_length=2000)
    analysis_confidence: Annotated[float, Field(ge=0.0, le=1.0)]


class StrategyRecommendation(StrictModel):
    title: str = Field(min_length=1, max_length=250)
    rationale: str = Field(min_length=1, max_length=1500)
    actions: list[str] = Field(min_length=1, max_length=10)
    expected_impact: Literal["high", "medium", "low"]
    time_horizon: Literal["0-30 days", "31-90 days", "90+ days"]


class StrategyReport(StrictModel):
    executive_summary: str = Field(min_length=1, max_length=3000)
    positioning_statement: str = Field(min_length=1, max_length=1000)
    recommendations: list[StrategyRecommendation] = Field(
        min_length=1,
        max_length=10,
    )
    risk_watchlist: list[str] = Field(default_factory=list, max_length=20)
    report_markdown: str = Field(min_length=1)


class GraphState(StrictModel):
    run_id: str = Field(min_length=8, max_length=100)
    target_domain: str | None = None
    competitors: list[str] = Field(default_factory=list, max_length=20)
    profiles: list[CompetitorProfile] = Field(default_factory=list)
    analysis: CompetitiveAnalysis | None = None
    strategy: StrategyReport | None = None
    errors: list[PipelineError] = Field(default_factory=list)
    retry_count: int = Field(default=0, ge=0)
    max_retries: int = Field(default=3, ge=1)
    status: PipelineStatus = PipelineStatus.INITIALIZED
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None

    @field_validator("competitors")
    @classmethod
    def normalize_competitor_urls(cls, values: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()

        for candidate in values:
            url = candidate.strip()
            if not url:
                continue

            if not url.startswith(("http://", "https://")):
                url = f"https://{url}"

            canonical = url.rstrip("/")
            if canonical not in seen:
                seen.add(canonical)
                normalized.append(canonical)

        if not normalized:
            raise ValueError("At least one valid competitor URL is required.")

        return normalized

    def latest_recoverable_error(self) -> PipelineError | None:
        for error in reversed(self.errors):
            if error.recoverable:
                return error
        return None
