from __future__ import annotations

import logging
import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup, Tag
from firecrawl import FirecrawlApp
from langchain_core.runnables import RunnableConfig
from pydantic import ValidationError
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import Settings
from models import (
    CompetitorProfile,
    CompetitiveAnalysis,
    ErrorCategory,
    ExtractionMethod,
    GraphState,
    MarketGap,
    PipelineError,
    PipelineStatus,
    PricePoint,
    StrategyRecommendation,
    StrategyReport,
)


LOGGER = logging.getLogger(__name__)

PRICE_PATTERN = re.compile(
    r"(?<!\w)"
    r"(?:[$€£]\s?\d{1,5}(?:[,\d]{0,10})?(?:\.\d{1,2})?"
    r"|\d{1,5}(?:\.\d{1,2})?\s?(?:USD|EUR|GBP))"
    r"(?:\s?(?:/|per\s+)(?:month|mo|year|yr|user|seat))?",
    re.IGNORECASE,
)

FEATURE_HINTS = (
    "analytics",
    "api",
    "automation",
    "collaboration",
    "compliance",
    "crm",
    "dashboard",
    "data",
    "enterprise",
    "integration",
    "intelligence",
    "monitoring",
    "reporting",
    "security",
    "workflow",
    "ai",
)

BLOCK_MARKERS = (
    "access denied",
    "captcha",
    "cloudflare",
    "temporarily unavailable",
    "unusual traffic",
    "verify you are human",
)


@dataclass(frozen=True)
class ScrapeResult:
    markdown: str
    title: str | None
    description: str | None
    status_code: int | None
    method: ExtractionMethod


class ScraperAgent:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.session = self._build_session()
        self.firecrawl = (
            FirecrawlApp(api_key=settings.firecrawl_api_key)
            if settings.firecrawl_enabled
            else None
        )

    def _build_session(self) -> requests.Session:
        retry_strategy = Retry(
            total=self.settings.max_retries,
            connect=self.settings.max_retries,
            read=self.settings.max_retries,
            backoff_factor=0.8,
            status_forcelist=(408, 429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
            raise_on_status=False,
        )

        adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=20, pool_maxsize=20)
        session = requests.Session()
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        session.headers.update(
            {
                "User-Agent": self.settings.user_agent,
                "Accept": (
                    "text/html,application/xhtml+xml,application/xml;q=0.9,"
                    "text/plain;q=0.8,*/*;q=0.5"
                ),
                "Accept-Language": "en-US,en;q=0.9",
                "Cache-Control": "no-cache",
            }
        )
        return session

    def scrape_competitor(self, url: str) -> CompetitorProfile:
        LOGGER.info("Scraper agent processing %s", url)

        primary_error: Exception | None = None
        result: ScrapeResult | None = None

        if self.firecrawl is not None:
            try:
                result = self._scrape_with_firecrawl(url)
                LOGGER.info("Firecrawl extraction succeeded for %s", url)
            except Exception as exc:
                primary_error = exc
                LOGGER.warning(
                    "Firecrawl extraction failed for %s: %s. Falling back to HTTP extraction.",
                    url,
                    exc,
                )

        if result is None:
            result = self._scrape_with_requests(url)

        profile = self._build_profile(url, result)

        if primary_error is not None and not profile.source_excerpt:
            raise RuntimeError(
                f"Both Firecrawl and fallback extraction failed for {url}: {primary_error}"
            )

        return profile

    def _scrape_with_firecrawl(self, url: str) -> ScrapeResult:
        if self.firecrawl is None:
            raise RuntimeError("Firecrawl is not configured.")

        response: Any = self.firecrawl.scrape_url(
            url,
            formats=["markdown"],
            only_main_content=True,
            timeout=self.settings.request_timeout_seconds * 1000,
        )

        markdown = self._extract_firecrawl_markdown(response)
        metadata = self._extract_firecrawl_metadata(response)

        if len(markdown.strip()) < 50:
            raise ValueError("Firecrawl returned insufficient extractable content.")

        return ScrapeResult(
            markdown=markdown,
            title=metadata.get("title"),
            description=metadata.get("description"),
            status_code=200,
            method=ExtractionMethod.FIRECRAWL,
        )

    @staticmethod
    def _extract_firecrawl_markdown(response: Any) -> str:
        if isinstance(response, dict):
            data = response.get("data", response)
            if isinstance(data, dict):
                return str(data.get("markdown") or data.get("content") or "")
            return str(response.get("markdown") or "")

        data = getattr(response, "data", response)
        if isinstance(data, dict):
            return str(data.get("markdown") or data.get("content") or "")

        return str(getattr(data, "markdown", "") or getattr(response, "markdown", ""))

    @staticmethod
    def _extract_firecrawl_metadata(response: Any) -> dict[str, str]:
        if isinstance(response, dict):
            data = response.get("data", response)
            if isinstance(data, dict):
                metadata = data.get("metadata", {})
                return metadata if isinstance(metadata, dict) else {}
            return {}

        data = getattr(response, "data", response)
        metadata = getattr(data, "metadata", {})
        if isinstance(metadata, dict):
            return metadata
        return {}

    def _scrape_with_requests(self, url: str) -> ScrapeResult:
        try:
            response = self.session.get(
                url,
                timeout=(
                    min(10, self.settings.request_timeout_seconds),
                    self.settings.request_timeout_seconds,
                ),
                allow_redirects=True,
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"Network request failed: {exc}") from exc

        body_sample = response.text[:5000].lower()
        if response.status_code in {401, 403, 429} or any(
            marker in body_sample for marker in BLOCK_MARKERS
        ):
            raise PermissionError(
                f"Target blocked automated access with HTTP {response.status_code}."
            )

        if response.status_code >= 400:
            raise RuntimeError(
                f"Target returned HTTP {response.status_code} for {response.url}."
            )

        content_type = response.headers.get("Content-Type", "").lower()
        if "html" not in content_type and "text" not in content_type:
            raise ValueError(f"Unsupported response content type: {content_type}.")

        soup = BeautifulSoup(response.text, "html.parser")
        self._remove_noise(soup)

        title = self._get_meta_content(soup, "og:title") or self._safe_text(soup.title)
        description = (
            self._get_meta_content(soup, "description")
            or self._get_meta_content(soup, "og:description")
        )
        markdown = self._html_to_clean_text(soup)

        if len(markdown.strip()) < 50:
            raise ValueError("Fallback extraction returned insufficient readable content.")

        return ScrapeResult(
            markdown=markdown,
            title=title,
            description=description,
            status_code=response.status_code,
            method=ExtractionMethod.REQUESTS_FALLBACK,
        )

    @staticmethod
    def _remove_noise(soup: BeautifulSoup) -> None:
        for element in soup(
            [
                "script",
                "style",
                "noscript",
                "svg",
                "iframe",
                "nav",
                "footer",
                "header",
                "form",
                "aside",
                "button",
            ]
        ):
            element.decompose()

    @staticmethod
    def _safe_text(tag: Tag | None) -> str | None:
        if tag is None:
            return None
        value = " ".join(tag.get_text(" ", strip=True).split())
        return value or None

    @staticmethod
    def _get_meta_content(soup: BeautifulSoup, property_name: str) -> str | None:
        tag = soup.find("meta", attrs={"property": property_name})
        if tag is None:
            tag = soup.find("meta", attrs={"name": property_name})
        if tag is None:
            return None
        content = tag.get("content")
        return str(content).strip() if content else None

    @staticmethod
    def _html_to_clean_text(soup: BeautifulSoup) -> str:
        candidate = soup.find("main") or soup.find("article") or soup.body or soup
        text = candidate.get_text("\n", strip=True)

        lines: list[str] = []
        observed: set[str] = set()
        for line in text.splitlines():
            normalized = " ".join(line.split())
            key = normalized.lower()
            if len(normalized) >= 2 and key not in observed:
                lines.append(normalized)
                observed.add(key)

        return "\n".join(lines[:1200])

    def _build_profile(self, url: str, result: ScrapeResult) -> CompetitorProfile:
        content = result.markdown
        prices = self._extract_pricing(content)
        features = self._extract_features(content)
        differentiators = self._extract_differentiators(content)

        confidence = self._calculate_confidence(
            content=content,
            prices=prices,
            features=features,
            title=result.title,
        )

        return CompetitorProfile(
            name=self._derive_name(url, result.title),
            url=url,
            title=result.title,
            description=result.description,
            pricing=prices,
            features=features,
            differentiators=differentiators,
            source_excerpt=content[:8000],
            extraction_method=result.method,
            extraction_confidence=confidence,
            http_status_code=result.status_code,
        )

    @staticmethod
    def _derive_name(url: str, title: str | None) -> str:
        if title:
            cleaned = re.split(r"\s[|\-–—]\s", title, maxsplit=1)[0].strip()
            if cleaned:
                return cleaned[:200]

        hostname = urlparse(url).netloc.lower().removeprefix("www.")
        return hostname.split(".")[0].replace("-", " ").replace("_", " ").title()

    @staticmethod
    def _extract_pricing(content: str) -> list[PricePoint]:
        results: list[PricePoint] = []
        observed: set[str] = set()

        lines = content.splitlines()
        for index, line in enumerate(lines):
            for match in PRICE_PATTERN.finditer(line):
                amount = match.group(0).strip()
                context_lines = lines[max(0, index - 1): min(len(lines), index + 2)]
                evidence = " ".join(context_lines)
                label = ScraperAgent._infer_price_label(line, evidence)
                billing_period = ScraperAgent._infer_billing_period(amount, evidence)
                key = f"{label.lower()}|{amount.lower()}"

                if key not in observed:
                    observed.add(key)
                    results.append(
                        PricePoint(
                            label=label,
                            amount=amount,
                            billing_period=billing_period,
                            evidence=evidence[:500],
                        )
                    )

                if len(results) >= 12:
                    return results

        return results

    @staticmethod
    def _infer_price_label(line: str, evidence: str) -> str:
        match = re.search(
            r"\b(free|starter|basic|pro|professional|business|team|growth|enterprise|premium)\b",
            f"{line} {evidence}",
            re.IGNORECASE,
        )
        return match.group(1).title() if match else "Listed price"

    @staticmethod
    def _infer_billing_period(amount: str, evidence: str) -> str | None:
        combined = f"{amount} {evidence}".lower()
        if re.search(r"\b(month|mo)\b|/", combined):
            return "monthly"
        if re.search(r"\b(year|yr|annual)\b", combined):
            return "annual"
        if re.search(r"\b(user|seat)\b", combined):
            return "per user"
        return None

    @staticmethod
    def _extract_features(content: str) -> list[str]:
        features: list[str] = []
        observed: set[str] = set()

        for raw_line in content.splitlines():
            line = " ".join(raw_line.split())
            lower = line.lower()

            if len(line) < 4 or len(line) > 180:
                continue

            is_feature_like = (
                any(hint in lower for hint in FEATURE_HINTS)
                or line.startswith(("-", "*", "•"))
                or bool(re.match(r"^[A-Z][A-Za-z0-9 &/+\-]{2,80}$", line))
            )

            marketing_noise = (
                "cookie" in lower
                or "sign in" in lower
                or "log in" in lower
                or "copyright" in lower
            )

            if is_feature_like and not marketing_noise:
                cleaned = line.lstrip("-*• ").strip()
                key = cleaned.lower()
                if key not in observed:
                    observed.add(key)
                    features.append(cleaned)

            if len(features) >= 25:
                break

        return features

    @staticmethod
    def _extract_differentiators(content: str) -> list[str]:
        differentiators: list[str] = []
        cue_words = (
            "only",
            "unique",
            "first",
            "built for",
            "purpose-built",
            "specialized",
            "leading",
            "trusted by",
        )

        for raw_line in content.splitlines():
            line = " ".join(raw_line.split())
            lower = line.lower()
            if len(line) < 15 or len(line) > 250:
                continue
            if any(cue in lower for cue in cue_words):
                differentiators.append(line)

            if len(differentiators) >= 10:
                break

        return differentiators

    @staticmethod
    def _calculate_confidence(
        content: str,
        prices: list[PricePoint],
        features: list[str],
        title: str | None,
    ) -> float:
        score = 0.20
        score += min(len(content) / 5000, 0.35)
        score += min(len(features) / 20, 0.25)
        score += min(len(prices) / 5, 0.15)
        score += 0.05 if title else 0.0
        return round(min(score, 0.98), 2)


class GapAnalystAgent:
    def analyze(self, profiles: list[CompetitorProfile]) -> CompetitiveAnalysis:
        if not profiles:
            raise ValueError("Cannot analyze an empty set of competitor profiles.")

        feature_sets = {
            profile.name: {self._normalize_feature(feature) for feature in profile.features}
            for profile in profiles
        }

        feature_counts: Counter[str] = Counter()
        for feature_set in feature_sets.values():
            feature_counts.update(feature_set)

        common_features = [
            feature
            for feature, count in feature_counts.most_common()
            if count >= 2 and feature
        ][:12]

        unique_features_by_competitor: dict[str, list[str]] = {}
        for competitor_name, feature_set in feature_sets.items():
            unique = [
                feature
                for feature in sorted(feature_set)
                if feature_counts[feature] == 1
            ][:8]
            if unique:
                unique_features_by_competitor[competitor_name] = unique

        pricing_observations = self._build_pricing_observations(profiles)
        market_gaps = self._identify_market_gaps(
            profiles=profiles,
            common_features=common_features,
            pricing_observations=pricing_observations,
        )

        average_confidence = sum(
            profile.extraction_confidence for profile in profiles
        ) / len(profiles)

        market_summary = (
            f"CompetitivePulse analyzed {len(profiles)} publicly accessible competitor "
            f"profile{'s' if len(profiles) != 1 else ''}. The observed market emphasizes "
            f"{', '.join(common_features[:5]) if common_features else 'a fragmented set of capabilities'}. "
            f"Extraction quality averaged {average_confidence:.0%}; findings should be validated "
            "against current product pages before making high-stakes commercial decisions."
        )

        positioning = self._build_positioning(common_features, market_gaps)

        return CompetitiveAnalysis(
            market_summary=market_summary,
            common_features=common_features,
            unique_features_by_competitor=unique_features_by_competitor,
            pricing_observations=pricing_observations,
            market_gaps=market_gaps,
            recommended_positioning=positioning,
            analysis_confidence=round(average_confidence, 2),
        )

    @staticmethod
    def _normalize_feature(feature: str) -> str:
        normalized = feature.lower()
        normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()

        aliases = {
            "artificial intelligence": "ai",
            "ai powered": "ai",
            "analytics dashboard": "analytics",
            "reporting analytics": "analytics",
            "integrations": "integration",
            "workflows": "workflow",
            "automated workflows": "automation",
        }
        return aliases.get(normalized, normalized)

    @staticmethod
    def _build_pricing_observations(
        profiles: list[CompetitorProfile],
    ) -> list[str]:
        observations: list[str] = []
        profiles_with_pricing = [profile for profile in profiles if profile.pricing]

        if not profiles_with_pricing:
            return [
                "No reliably parseable public price points were found in the retrieved content. "
                "Treat pricing transparency as an explicit research gap."
            ]

        observations.append(
            f"{len(profiles_with_pricing)} of {len(profiles)} analyzed competitors displayed "
            "at least one parseable public price point."
        )

        for profile in profiles_with_pricing[:8]:
            preview = ", ".join(
                f"{price.label}: {price.amount}"
                for price in profile.pricing[:3]
            )
            observations.append(f"{profile.name}: {preview}.")

        return observations

    @staticmethod
    def _identify_market_gaps(
        profiles: list[CompetitorProfile],
        common_features: list[str],
        pricing_observations: list[str],
    ) -> list[MarketGap]:
        source_urls = [str(profile.url) for profile in profiles]
        gaps: list[MarketGap] = []

        profiles_with_pricing = sum(1 for profile in profiles if profile.pricing)
        if profiles_with_pricing < len(profiles):
            gaps.append(
                MarketGap(
                    title="Transparent, buyer-friendly pricing",
                    finding=(
                        f"Only {profiles_with_pricing} of {len(profiles)} profiles exposed "
                        "parseable pricing in the retrieved public content."
                    ),
                    opportunity=(
                        "Differentiate with clear packaging, a public starting price or pricing "
                        "estimator, and an explicit explanation of what changes across tiers."
                    ),
                    evidence_urls=source_urls,
                    priority="high",
                    confidence=0.75,
                )
            )

        if "security" not in common_features and "compliance" not in common_features:
            gaps.append(
                MarketGap(
                    title="Trust and governance as a product surface",
                    finding=(
                        "Security and compliance did not emerge as consistently visible product "
                        "messages across the extracted competitive feature set."
                    ),
                    opportunity=(
                        "Lead with clear governance controls, auditability, deployment boundaries, "
                        "and enterprise-ready security documentation."
                    ),
                    evidence_urls=source_urls,
                    priority="medium",
                    confidence=0.62,
                )
            )

        if "integration" not in common_features:
            gaps.append(
                MarketGap(
                    title="Faster ecosystem adoption",
                    finding=(
                        "Integration capability was not a dominant recurring theme in the extracted "
                        "content."
                    ),
                    opportunity=(
                        "Prioritize a focused integration catalog, documented APIs, and guided "
                        "onboarding for the systems buyers already use."
                    ),
                    evidence_urls=source_urls,
                    priority="medium",
                    confidence=0.58,
                )
            )

        if not gaps:
            gaps.append(
                MarketGap(
                    title="Execution-focused differentiation",
                    finding=(
                        "The extracted competitors show substantial overlap in baseline feature "
                        "messaging."
                    ),
                    opportunity=(
                        "Differentiate through measurable time-to-value, opinionated workflows, "
                        "superior implementation support, and outcome-based proof."
                    ),
                    evidence_urls=source_urls,
                    priority="high",
                    confidence=0.66,
                )
            )

        return gaps[:5]

    @staticmethod
    def _build_positioning(
        common_features: list[str],
        market_gaps: list[MarketGap],
    ) -> str:
        baseline = ", ".join(common_features[:3]) or "baseline category capabilities"
        primary_gap = market_gaps[0].title.lower()

        return (
            f"Position the offering as the fastest path to measurable business outcomes: "
            f"match expected capabilities such as {baseline}, while making {primary_gap} "
            "a concrete, evidence-backed reason to choose and remain with the product."
        )


class StrategyWriterAgent:
    def write(
        self,
        profiles: list[CompetitorProfile],
        analysis: CompetitiveAnalysis,
    ) -> StrategyReport:
        if not profiles:
            raise ValueError("Strategy generation requires at least one profile.")

        recommendations = self._recommendations_from_analysis(analysis)
        competitor_names = ", ".join(profile.name for profile in profiles)

        executive_summary = (
            f"The analysis of {competitor_names} indicates a market where buyers can likely "
            f"expect overlapping baseline functionality, but where visibility into proof, "
            f"packaging, and differentiation varies. The recommended strategy is to pair "
            f"{analysis.recommended_positioning.lower()} with a disciplined evidence system: "
            "validated competitive claims, buyer-facing comparison assets, and measurable "
            "implementation outcomes."
        )

        risk_watchlist = [
            "Public pages can change frequently; re-run the analysis before sales campaigns, pricing decisions, or board-level reporting.",
            "Automated extraction may miss JavaScript-rendered content, gated pages, regional pricing, and personalized offers.",
            "Do not treat inferred feature overlap as proof of functional equivalence without product validation.",
        ]

        report_markdown = self._render_markdown(
            profiles=profiles,
            analysis=analysis,
            executive_summary=executive_summary,
            recommendations=recommendations,
            risk_watchlist=risk_watchlist,
        )

        return StrategyReport(
            executive_summary=executive_summary,
            positioning_statement=analysis.recommended_positioning,
            recommendations=recommendations,
            risk_watchlist=risk_watchlist,
            report_markdown=report_markdown,
        )

    @staticmethod
    def _recommendations_from_analysis(
        analysis: CompetitiveAnalysis,
    ) -> list[StrategyRecommendation]:
        recommendations: list[StrategyRecommendation] = []

        for gap in analysis.market_gaps[:3]:
            time_horizon: str
            if gap.priority == "high":
                time_horizon = "0-30 days"
            elif gap.priority == "medium":
                time_horizon = "31-90 days"
            else:
                time_horizon = "90+ days"

            recommendations.append(
                StrategyRecommendation(
                    title=f"Address: {gap.title}",
                    rationale=gap.opportunity,
                    actions=[
                        f"Create a buyer-facing proof asset that directly addresses {gap.title.lower()}.",
                        "Validate all claims with product, security, and customer-facing stakeholders.",
                        "Instrument adoption and conversion metrics to evaluate whether the differentiated message performs.",
                    ],
                    expected_impact=gap.priority,
                    time_horizon=time_horizon,
                )
            )

        recommendations.append(
            StrategyRecommendation(
                title="Operationalize competitive intelligence",
                rationale=(
                    "Competitive intelligence creates durable value when changes are captured, "
                    "validated, and routed into sales enablement, product planning, and messaging."
                ),
                actions=[
                    "Schedule recurring scans of approved public competitor URLs.",
                    "Store validated profiles and changes in a durable database or data warehouse.",
                    "Create review workflows for legal, product marketing, and sales enablement owners.",
                ],
                expected_impact="high",
                time_horizon="31-90 days",
            )
        )

        return recommendations[:4]

    @staticmethod
    def _render_markdown(
        profiles: list[CompetitorProfile],
        analysis: CompetitiveAnalysis,
        executive_summary: str,
        recommendations: list[StrategyRecommendation],
        risk_watchlist: list[str],
    ) -> str:
        lines: list[str] = [
            "# CompetitivePulse Intelligence Report",
            "",
            "## Executive Summary",
            "",
            executive_summary,
            "",
            "## Recommended Positioning",
            "",
            analysis.recommended_positioning,
            "",
            "## Competitive Profiles",
            "",
        ]

        for profile in profiles:
            lines.extend(
                [
                    f"### {profile.name}",
                    "",
                    f"- **URL:** {profile.url}",
                    f"- **Extraction method:** {profile.extraction_method.value}",
                    f"- **Extraction confidence:** {profile.extraction_confidence:.0%}",
                ]
            )

            if profile.description:
                lines.append(f"- **Description:** {profile.description}")

            if profile.pricing:
                pricing_preview = "; ".join(
                    f"{item.label} — {item.amount}"
                    for item in profile.pricing[:5]
                )
                lines.append(f"- **Observed pricing:** {pricing_preview}")

            if profile.features:
                lines.append(
                    f"- **Extracted features:** {', '.join(profile.features[:10])}"
                )

            if profile.differentiators:
                lines.append(
                    f"- **Differentiator signals:** {' | '.join(profile.differentiators[:3])}"
                )

            lines.append("")

        lines.extend(
            [
                "## Market Findings",
                "",
                analysis.market_summary,
                "",
                "### Commonly Observed Features",
                "",
            ]
        )

        if analysis.common_features:
            lines.extend(f"- {feature}" for feature in analysis.common_features)
        else:
            lines.append("- No recurring features were confidently identified.")

        lines.extend(["", "### Pricing Observations", ""])
        lines.extend(f"- {observation}" for observation in analysis.pricing_observations)

        lines.extend(["", "### Market Gaps", ""])
        for gap in analysis.market_gaps:
            lines.extend(
                [
                    f"#### {gap.title} ({gap.priority.title()} priority)",
                    "",
                    f"**Finding:** {gap.finding}",
                    "",
                    f"**Opportunity:** {gap.opportunity}",
                    "",
                    f"**Confidence:** {gap.confidence:.0%}",
                    "",
                ]
            )

        lines.extend(["## Strategic Recommendations", ""])
        for index, recommendation in enumerate(recommendations, start=1):
            lines.extend(
                [
                    f"### {index}. {recommendation.title}",
                    "",
                    f"**Impact:** {recommendation.expected_impact.title()}  ",
                    f"**Time horizon:** {recommendation.time_horizon}",
                    "",
                    recommendation.rationale,
                    "",
                    "**Actions**",
                    "",
                ]
            )
            lines.extend(f"- {action}" for action in recommendation.actions)
            lines.append("")

        lines.extend(["## Risk Watchlist", ""])
        lines.extend(f"- {risk}" for risk in risk_watchlist)
        lines.extend(
            [
                "",
                "## Methodology Note",
                "",
                "This report is generated from publicly accessible web content. "
                "Extraction is automated and heuristic-assisted; validate material claims, "
                "pricing, availability, and feature parity before external use.",
                "",
            ]
        )

        return "\n".join(lines)


def scraper_node(state: GraphState, config: RunnableConfig, settings: Settings) -> dict[str, Any]:
    del config
    started = time.perf_counter()
    agent = ScraperAgent(settings)
    profiles: list[CompetitorProfile] = []
    errors: list[PipelineError] = []

    for url in state.competitors:
        try:
            profiles.append(agent.scrape_competitor(url))
        except (requests.RequestException, PermissionError, ValueError, RuntimeError, ValidationError) as exc:
            category = _classify_error(exc)
            errors.append(
                PipelineError(
                    category=category,
                    message=str(exc),
                    source_url=url,
                    recoverable=True,
                )
            )
            LOGGER.exception("Failed to scrape %s", url)

    elapsed = time.perf_counter() - started
    LOGGER.info(
        "Scraper node complete | profiles=%s | errors=%s | latency=%.2fs",
        len(profiles),
        len(errors),
        elapsed,
    )

    return {
        "profiles": profiles,
        "errors": [*state.errors, *errors],
        "status": PipelineStatus.SCRAPING,
    }


def recovery_node(state: GraphState, config: RunnableConfig, settings: Settings) -> dict[str, Any]:
    del config, settings
    retry_count = state.retry_count + 1
    recoverable_error = state.latest_recoverable_error()

    LOGGER.warning(
        "Recovery node invoked | retry=%s/%s | error=%s",
        retry_count,
        state.max_retries,
        recoverable_error.message if recoverable_error else "unknown",
    )

    return {
        "retry_count": retry_count,
        "status": PipelineStatus.RETRYING,
    }


def analyst_node(state: GraphState, config: RunnableConfig, settings: Settings) -> dict[str, Any]:
    del config, settings
    try:
        analysis = GapAnalystAgent().analyze(state.profiles)
        return {
            "analysis": analysis,
            "status": PipelineStatus.ANALYZING,
        }
    except (ValueError, ValidationError) as exc:
        error = PipelineError(
            category=ErrorCategory.ANALYSIS,
            message=str(exc),
            recoverable=False,
        )
        LOGGER.exception("Analysis node failed")
        return {
            "errors": [*state.errors, error],
            "status": PipelineStatus.FAILED,
        }


def writer_node(state: GraphState, config: RunnableConfig, settings: Settings) -> dict[str, Any]:
    del config, settings

    if state.analysis is None:
        error = PipelineError(
            category=ErrorCategory.REPORTING,
            message="Writer node received no competitive analysis.",
            recoverable=False,
        )
        return {
            "errors": [*state.errors, error],
            "status": PipelineStatus.FAILED,
        }

    try:
        strategy = StrategyWriterAgent().write(state.profiles, state.analysis)
        return {
            "strategy": strategy,
            "status": PipelineStatus.WRITING,
        }
    except (ValueError, ValidationError) as exc:
        error = PipelineError(
            category=ErrorCategory.REPORTING,
            message=str(exc),
            recoverable=False,
        )
        LOGGER.exception("Writer node failed")
        return {
            "errors": [*state.errors, error],
            "status": PipelineStatus.FAILED,
        }


def complete_node(state: GraphState, config: RunnableConfig, settings: Settings) -> dict[str, Any]:
    del config, settings
    return {
        "status": PipelineStatus.COMPLETED,
    }


def _classify_error(exc: Exception) -> ErrorCategory:
    if isinstance(exc, PermissionError):
        return ErrorCategory.ACCESS
    if isinstance(exc, requests.RequestException):
        return ErrorCategory.NETWORK
    if isinstance(exc, ValidationError):
        return ErrorCategory.VALIDATION
    if isinstance(exc, ValueError):
        return ErrorCategory.EXTRACTION
    if isinstance(exc, RuntimeError):
        return ErrorCategory.EXTRACTION
    return ErrorCategory.UNKNOWN
