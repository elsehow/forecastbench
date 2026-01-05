#!/usr/bin/env python3
"""Run a small benchmark test with multiple models.

This script demonstrates the full ForecastBench v2 pipeline:
1. Fetch questions from available sources
2. Create a small question set for testing
3. Run forecasts with multiple cheap/fast models
4. Display results

Usage:
    # From v2 directory:
    uv run python scripts/run_benchmark.py

    # With options:
    uv run python scripts/run_benchmark.py --num-questions 20 --dry-run
"""

import argparse
import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

from forecastbench.forecasters.llm import LLMForecaster
from forecastbench.models import Forecast, Question, QuestionType
from forecastbench.sampling import QuestionSampler, SamplingConfig
from forecastbench.sources import get_all_sources
from forecastbench.storage.sqlite import SQLiteStorage

# Cheap/fast models for testing (uses less API credits)
TEST_MODELS = [
    ("anthropic/claude-3-haiku-20240307", "ANTHROPIC_API_KEY"),
    ("openai/gpt-4o-mini", "OPENAI_API_KEY"),
    ("gemini/gemini-2.0-flash", "GOOGLE_API_KEY"),
    ("mistral/mistral-small-latest", "MISTRAL_API_KEY"),
]


def get_available_models() -> list[str]:
    """Get models that have API keys configured."""
    import os

    available = []
    for model, env_var in TEST_MODELS:
        if os.environ.get(env_var):
            available.append(model)
    return available


async def fetch_questions(storage: SQLiteStorage, sources: list[str] | None = None) -> int:
    """Fetch questions from all available sources."""
    print("\n" + "=" * 60)
    print("STEP 1: Fetching questions from sources")
    print("=" * 60)

    all_sources = get_all_sources()
    if sources:
        all_sources = {k: v for k, v in all_sources.items() if k in sources}

    total_questions = 0
    for name, source_cls in all_sources.items():
        print(f"\n  Fetching from {name}...")
        try:
            source = source_cls()
            questions = await source.fetch_questions()
            if questions:
                await storage.save_questions(questions)
                print(f"    -> {len(questions)} questions")
                total_questions += len(questions)
            else:
                print(f"    -> No questions (may need API key)")
        except Exception as e:
            print(f"    -> Error: {e}")

    print(f"\n  Total: {total_questions} questions fetched")
    return total_questions


async def create_question_set(
    storage: SQLiteStorage, num_questions: int, name: str | None = None
) -> tuple[int | None, list[Question]]:
    """Create a question set by sampling from available questions.

    Returns:
        Tuple of (question_set_id, list of sampled questions)
    """
    print("\n" + "=" * 60)
    print("STEP 2: Creating question set")
    print("=" * 60)

    # Get available questions
    questions = await storage.get_questions(resolved=False)
    binary_questions = [q for q in questions if q.question_type == QuestionType.BINARY]

    print(f"\n  Available questions: {len(questions)} total, {len(binary_questions)} binary")

    if len(binary_questions) < num_questions:
        print(f"  Warning: Only {len(binary_questions)} binary questions available")
        num_questions = min(num_questions, len(binary_questions))

    if num_questions == 0:
        print("  Error: No questions available to sample!")
        return None, []

    # Sample questions
    config = SamplingConfig(
        num_questions=num_questions,
        market_fraction=0.5,  # 50% market, 50% data
        max_resolution_days=365,
    )
    sampler = QuestionSampler(config)
    result = sampler.sample_stratified(binary_questions)
    sampled = result.questions

    print(f"  Sampled {len(sampled)} questions")
    if result.warnings:
        for warning in result.warnings:
            print(f"  Warning: {warning}")

    # Create question set
    set_name = name or f"test-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    freeze_date = datetime.now()
    forecast_due_date = freeze_date + timedelta(days=1)  # Due tomorrow for testing
    resolution_dates = [
        freeze_date + timedelta(days=d) for d in [7, 14, 30]  # Shorter horizons for testing
    ]

    question_set_id = await storage.create_question_set(
        name=set_name,
        questions=sampled,
        freeze_date=freeze_date.date(),
        forecast_due_date=forecast_due_date.date(),
        resolution_dates=[d.date() for d in resolution_dates],
    )

    print(f"  Created question set #{question_set_id}: '{set_name}'")
    print(f"  Freeze date: {freeze_date.date()}")
    print(f"  Forecast due: {forecast_due_date.date()}")
    print(f"  Resolution dates: {[d.date() for d in resolution_dates]}")

    return question_set_id, sampled


async def get_questions_for_set(storage: SQLiteStorage, question_set_id: int) -> list[Question]:
    """Get questions for a question set."""
    # Get question IDs from the set
    items = await storage.get_question_set_items(question_set_id)

    # Fetch each question
    questions = []
    for item in items:
        q = await storage.get_question(item["source"], item["question_id"])
        if q:
            questions.append(q)

    return questions


async def run_forecasts(
    storage: SQLiteStorage,
    question_set_id: int,
    questions: list[Question],
    models: list[str],
    dry_run: bool = False,
) -> dict[str, int]:
    """Run forecasts for all models on the question set."""
    print("\n" + "=" * 60)
    print("STEP 3: Running forecasts")
    print("=" * 60)

    print(f"\n  Questions to forecast: {len(questions)}")
    print(f"  Models to run: {len(models)}")

    if dry_run:
        print("\n  [DRY RUN - not actually calling APIs]")
        return {model: 0 for model in models}

    results = {}
    for model in models:
        print(f"\n  Running {model}...")
        forecaster = LLMForecaster(model=model)

        success_count = 0
        error_count = 0

        for i, question in enumerate(questions):
            try:
                forecast = await forecaster.forecast(question)
                # Create new forecast with question_set_id since Forecast is immutable
                forecast_with_set = Forecast(
                    question_id=forecast.question_id,
                    source=forecast.source,
                    forecaster=forecast.forecaster,
                    probability=forecast.probability,
                    point_estimate=forecast.point_estimate,
                    quantile_values=forecast.quantile_values,
                    reasoning=forecast.reasoning,
                    question_set_id=question_set_id,
                )
                await storage.save_forecast(forecast_with_set)

                if forecast.reasoning and forecast.reasoning.startswith("Error:"):
                    error_count += 1
                else:
                    success_count += 1

                # Progress indicator
                if (i + 1) % 10 == 0:
                    print(f"    Progress: {i + 1}/{len(questions)}")

            except Exception as e:
                print(f"    Error on question {question.id}: {e}")
                error_count += 1

        results[model] = success_count
        print(f"    -> {success_count} successful, {error_count} errors")

    return results


async def show_summary(storage: SQLiteStorage, question_set_id: int):
    """Show a summary of forecasts made."""
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    forecasts = await storage.get_forecasts(question_set_id=question_set_id)

    # Group by model
    by_model: dict[str, list] = {}
    for f in forecasts:
        if f.forecaster not in by_model:
            by_model[f.forecaster] = []
        by_model[f.forecaster].append(f)

    print(f"\n  Question set #{question_set_id}")
    print(f"  Total forecasts: {len(forecasts)}")
    print("\n  By model:")
    for model, model_forecasts in sorted(by_model.items()):
        probs = [f.probability for f in model_forecasts if f.probability is not None]
        if probs:
            avg_prob = sum(probs) / len(probs)
            print(f"    {model}: {len(model_forecasts)} forecasts, avg probability: {avg_prob:.2%}")
        else:
            print(f"    {model}: {len(model_forecasts)} forecasts")

    # Show a few example forecasts
    if forecasts:
        print("\n  Sample forecasts:")
        for forecast in forecasts[:3]:
            q_id = forecast.question_id
            print(f"    - Q: {q_id[:50]}..." if len(q_id) > 50 else f"    - Q: {q_id}")
            print(f"      Model: {forecast.forecaster}")
            if forecast.probability is not None:
                print(f"      Prob: {forecast.probability:.2%}")
            if forecast.reasoning:
                reasoning_preview = forecast.reasoning[:100].replace("\n", " ")
                print(f"      Reasoning: {reasoning_preview}...")
            print()


async def main():
    parser = argparse.ArgumentParser(description="Run a small ForecastBench benchmark test")
    parser.add_argument(
        "--num-questions",
        type=int,
        default=10,
        help="Number of questions to sample (default: 10)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't actually call LLM APIs",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        help="Specific sources to fetch from (default: all available)",
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip fetching questions (use existing DB)",
    )
    parser.add_argument(
        "--question-set",
        type=int,
        help="Use existing question set instead of creating new one",
    )
    parser.add_argument(
        "--db",
        type=str,
        default="benchmark_test.db",
        help="Database file to use (default: benchmark_test.db)",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("ForecastBench v2 - Benchmark Test")
    print("=" * 60)

    # Check available models
    available_models = get_available_models()
    print(f"\nAvailable models (with API keys): {len(available_models)}")
    for model in available_models:
        print(f"  - {model}")

    if not available_models:
        print("\nError: No API keys found! Set at least one of:")
        print("  ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY, MISTRAL_API_KEY")
        print("\nYou can add these to v2/.env")
        return 1

    # Initialize storage (auto-initializes on first use)
    # Use absolute path relative to v2 directory
    v2_dir = Path(__file__).parent.parent
    db_path = v2_dir / args.db
    storage = SQLiteStorage(db_path)
    print(f"\nDatabase: {db_path}")

    # Step 1: Fetch questions
    if not args.skip_fetch:
        await fetch_questions(storage, args.sources)
    else:
        print("\n[Skipping question fetch - using existing DB]")

    # Step 2: Create or use question set
    questions: list[Question] = []
    if args.question_set:
        question_set_id = args.question_set
        print(f"\n[Using existing question set #{question_set_id}]")
        questions = await get_questions_for_set(storage, question_set_id)
        print(f"  Found {len(questions)} questions in set")
    else:
        question_set_id, questions = await create_question_set(storage, args.num_questions)
        if question_set_id is None:
            return 1

    # Step 3: Run forecasts
    await run_forecasts(storage, question_set_id, questions, available_models, dry_run=args.dry_run)

    # Step 4: Show summary
    if not args.dry_run:
        await show_summary(storage, question_set_id)

    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print(f"""
  Your forecasts are saved in {args.db}

  To see the leaderboard (after questions resolve):
    uv run forecastbench leaderboard --question-set {question_set_id}

  To resolve questions and compute scores:
    uv run forecastbench resolve --question-set {question_set_id}

  To export forecasts:
    uv run forecastbench forecasts --question-set {question_set_id} --format json
""")

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
