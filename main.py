from __future__ import annotations

import argparse
import logging
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

from langchain_core.runnables import RunnableConfig
from pydantic import ValidationError

from config import configure_logging, configure_observability, load_settings
from graph import build_competitive_pulse_graph
from models import GraphState, PipelineStatus


LOGGER = logging.getLogger(__name__)

DEFAULT_COMPETITORS = [
    "https://www.notion.so/product",
    "https://www.atlassian.com/software/jira",
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="CompetitivePulse",
        description=(
            "Autonomous multi-agent competitive intelligence and lead enrichment engine."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument(
        "--domain",
        type=str,
        help=(
            "Single competitor domain or URL. Example: competitor.com or "
            "https://competitor.com/pricing"
        ),
    )
    input_group.add_argument(
        "--competitor",
        dest="competitors",
        action="append",
        help=(
            "Competitor domain or URL. Repeat the flag for multiple competitors. "
            "Example: --competitor example.com --competitor example.org"
        ),
    )

    parser.add_argument(
        "--target-domain",
        type=str,
        default=None,
        help="Optional domain for the company conducting the analysis.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output Markdown report path.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=None,
        help="Override the configured maximum complete-scrape retry count.",
    )
    parser.add_argument(
        "--print-report",
        action="store_true",
        help="Print the generated Markdown report to stdout.",
    )

    return parser.parse_args()


def resolve_competitors(arguments: argparse.Namespace) -> list[str]:
    if arguments.domain:
        return [arguments.domain]

    if arguments.competitors:
        return arguments.competitors

    return DEFAULT_COMPETITORS


def build_output_path(
    reports_directory: Path,
    explicit_output: Path | None,
) -> Path:
    if explicit_output is not None:
        path = explicit_output
        if path.suffix.lower() != ".md":
            path = path.with_suffix(".md")
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    reports_directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return reports_directory / f"competitivepulse-report-{timestamp}.md"


def write_report(report_path: Path, markdown: str) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(markdown, encoding="utf-8")


def main() -> int:
    arguments = parse_arguments()

    try:
        settings = load_settings()
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    configure_logging(settings)
    configure_observability(settings)

    competitors = resolve_competitors(arguments)
    if len(competitors) > settings.max_competitors:
        print(
            f"Input error: received {len(competitors)} competitors but the configured "
            f"limit is {settings.max_competitors}.",
            file=sys.stderr,
        )
        return 2

    max_retries = arguments.max_retries or settings.max_retries
    if max_retries < 1:
        print("--max-retries must be at least 1.", file=sys.stderr)
        return 2

    try:
        initial_state = GraphState(
            run_id=str(uuid.uuid4()),
            target_domain=arguments.target_domain,
            competitors=competitors,
            max_retries=max_retries,
        )
    except ValidationError as exc:
        print(f"Input validation error:\n{exc}", file=sys.stderr)
        return 2

    LOGGER.info(
        "CompetitivePulse run started | run_id=%s | competitors=%s | firecrawl=%s | langsmith=%s",
        initial_state.run_id,
        len(initial_state.competitors),
        settings.firecrawl_enabled,
        settings.langsmith_enabled,
    )

    graph = build_competitive_pulse_graph(settings)
    runnable_config: RunnableConfig = {
        "run_name": f"competitivepulse-{initial_state.run_id}",
        "tags": [
            "competitivepulse",
            settings.environment,
            "multi-agent",
            "competitive-intelligence",
        ],
        "metadata": {
            "run_id": initial_state.run_id,
            "target_domain": initial_state.target_domain or "not-specified",
            "competitor_count": len(initial_state.competitors),
            "firecrawl_enabled": settings.firecrawl_enabled,
        },
    }

    try:
        result = graph.invoke(initial_state, config=runnable_config)
        final_state = GraphState.model_validate(result)
    except Exception as exc:
        LOGGER.exception("Unrecoverable graph execution failure.")
        print(f"Pipeline failure: {exc}", file=sys.stderr)
        return 1

    if final_state.status != PipelineStatus.COMPLETED or final_state.strategy is None:
        print("\nCompetitivePulse did not complete successfully.", file=sys.stderr)
        print(f"Run ID: {final_state.run_id}", file=sys.stderr)

        if final_state.errors:
            print("\nStructured errors:", file=sys.stderr)
            for error in final_state.errors:
                source = f" | source={error.source_url}" if error.source_url else ""
                print(
                    f"- [{error.category.value}] {error.message}{source}",
                    file=sys.stderr,
                )
        return 1

    report_path = build_output_path(settings.reports_directory, arguments.output)
    write_report(report_path, final_state.strategy.report_markdown)

    LOGGER.info(
        "CompetitivePulse run completed | run_id=%s | profiles=%s | errors=%s | report=%s",
        final_state.run_id,
        len(final_state.profiles),
        len(final_state.errors),
        report_path,
    )

    print("\nCompetitivePulse completed successfully.")
    print(f"Run ID: {final_state.run_id}")
    print(f"Profiles analyzed: {len(final_state.profiles)}")
    print(f"Non-fatal source errors: {len(final_state.errors)}")
    print(f"Report written: {report_path}")

    if settings.langsmith_enabled:
        print(f"LangSmith project: {settings.langsmith_project}")

    if arguments.print_report:
        print("\n" + "=" * 80 + "\n")
        print(final_state.strategy.report_markdown)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
