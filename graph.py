from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from agents import (
    analyst_node,
    complete_node,
    recovery_node,
    scraper_node,
    writer_node,
)
from config import Settings
from models import GraphState, PipelineStatus


def route_after_scrape(
    state: GraphState,
) -> Literal["recover", "analyze", "fail"]:
    """
    Route after scraping:
    - Analyze immediately when at least one profile is available.
    - Retry only if no profiles were retrieved and retries remain.
    - Stop otherwise, preserving structured errors for the caller.
    """
    if state.profiles:
        return "analyze"

    if state.latest_recoverable_error() is not None and state.retry_count < state.max_retries:
        return "recover"

    return "fail"


def route_after_recovery(
    state: GraphState,
) -> Literal["scrape", "fail"]:
    if state.retry_count < state.max_retries:
        return "scrape"
    return "fail"


def route_after_analysis(
    state: GraphState,
) -> Literal["write", "fail"]:
    if state.analysis is not None and state.status != PipelineStatus.FAILED:
        return "write"
    return "fail"


def finalize_failure(state: GraphState) -> dict[str, object]:
    return {
        "status": PipelineStatus.FAILED,
        "completed_at": datetime.now(UTC),
    }


def finalize_success(state: GraphState) -> dict[str, object]:
    if state.strategy is None:
        return finalize_failure(state)

    return {
        "status": PipelineStatus.COMPLETED,
        "completed_at": datetime.now(UTC),
    }


def build_competitive_pulse_graph(settings: Settings):
    """
    Create a compiled LangGraph workflow.

    The graph retries only complete scraping failures. Partial retrieval proceeds
    to analysis so the user still receives useful intelligence plus transparent
    errors for failed sources.
    """
    workflow = StateGraph(GraphState)

    def configured_scraper(
        state: GraphState,
        config: RunnableConfig,
    ) -> dict[str, object]:
        return scraper_node(state, config, settings)

    def configured_recovery(
        state: GraphState,
        config: RunnableConfig,
    ) -> dict[str, object]:
        return recovery_node(state, config, settings)

    def configured_analyst(
        state: GraphState,
        config: RunnableConfig,
    ) -> dict[str, object]:
        return analyst_node(state, config, settings)

    def configured_writer(
        state: GraphState,
        config: RunnableConfig,
    ) -> dict[str, object]:
        return writer_node(state, config, settings)

    def configured_complete(
        state: GraphState,
        config: RunnableConfig,
    ) -> dict[str, object]:
        return complete_node(state, config, settings)

    workflow.add_node("scrape", configured_scraper)
    workflow.add_node("recover", configured_recovery)
    workflow.add_node("analyze", configured_analyst)
    workflow.add_node("write", configured_writer)
    workflow.add_node("complete", configured_complete)
    workflow.add_node("fail", finalize_failure)

    workflow.add_edge(START, "scrape")

    workflow.add_conditional_edges(
        "scrape",
        route_after_scrape,
        {
            "recover": "recover",
            "analyze": "analyze",
            "fail": "fail",
        },
    )

    workflow.add_conditional_edges(
        "recover",
        route_after_recovery,
        {
            "scrape": "scrape",
            "fail": "fail",
        },
    )

    workflow.add_conditional_edges(
        "analyze",
        route_after_analysis,
        {
            "write": "write",
            "fail": "fail",
        },
    )

    workflow.add_edge("write", "complete")
    workflow.add_edge("complete", END)
    workflow.add_edge("fail", END)

    return workflow.compile()
