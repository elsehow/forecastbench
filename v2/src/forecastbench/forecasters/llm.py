"""LLM-based forecaster using LiteLLM with structured outputs."""

import logging
from datetime import datetime, timezone

import litellm
from pydantic import BaseModel, Field

from forecastbench.config import settings
from forecastbench.forecasters.base import Forecaster
from forecastbench.models import Forecast, Question, QuestionType

logger = logging.getLogger(__name__)

# Prompts for different question types

BINARY_PROMPT = """You are an expert superforecaster, familiar with the work of Tetlock and others. \
Make a prediction of the probability that the question will be resolved as true. \
You MUST give a probability estimate between 0 and 1 UNDER ALL CIRCUMSTANCES. \
If for some reason you can't answer, pick the base rate, but return a number between 0 and 1.

Question:
{question}

Question Background:
{background}

Today's Date: {today_date}

Resolution Date: {resolution_date}

Provide your probability estimate."""


CONTINUOUS_PROMPT = """You are an expert superforecaster, familiar with the work of Tetlock and others. \
Make a prediction for the numeric value that this question will resolve to. \
You MUST give a point estimate UNDER ALL CIRCUMSTANCES.

Question:
{question}

Question Background:
{background}
{value_range_context}

Today's Date: {today_date}

Resolution Date: {resolution_date}

Provide your point estimate and reasoning."""


QUANTILE_PROMPT = """You are an expert superforecaster, familiar with the work of Tetlock and others. \
Make a prediction for the distribution of possible values for this question. \
Provide your estimates for each of the requested quantiles.

Question:
{question}

Question Background:
{background}
{value_range_context}

Today's Date: {today_date}

Resolution Date: {resolution_date}

Quantiles to predict: {quantiles}

For each quantile, provide the value X such that there is that probability the true value is less than X."""


# Structured output schemas for each question type


class BinaryForecastResponse(BaseModel):
    """Structured output for binary forecast responses."""

    probability: float = Field(
        ge=0.0,
        le=1.0,
        description="Probability estimate between 0 and 1",
    )
    reasoning: str = Field(
        description="Brief reasoning for the forecast",
    )


class ContinuousForecastResponse(BaseModel):
    """Structured output for continuous forecast responses."""

    point_estimate: float = Field(
        description="Point estimate for the numeric value",
    )
    confidence_low: float | None = Field(
        default=None,
        description="Lower bound of 80% confidence interval (optional)",
    )
    confidence_high: float | None = Field(
        default=None,
        description="Upper bound of 80% confidence interval (optional)",
    )
    reasoning: str = Field(
        description="Brief reasoning for the forecast",
    )


class QuantileForecastResponse(BaseModel):
    """Structured output for quantile forecast responses."""

    quantile_values: list[float] = Field(
        description="Predicted values for each quantile, in the same order as requested",
    )
    reasoning: str = Field(
        description="Brief reasoning for the forecast",
    )


class LLMForecaster(Forecaster):
    """Forecaster that uses LLMs via LiteLLM with structured outputs.

    Supports binary, continuous, and quantile question types.
    """

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.0,
    ):
        """Initialize the LLM forecaster.

        Args:
            model: LiteLLM model identifier (e.g., "openai/gpt-4o", "anthropic/claude-3-opus").
                  If not provided, uses settings.default_model.
            temperature: Sampling temperature (0.0 for deterministic).
        """
        self.model = model or settings.default_model
        self.temperature = temperature

    @property
    def name(self) -> str:
        return self.model

    def _build_prompt(self, question: Question) -> tuple[str, type[BaseModel]]:
        """Build the appropriate prompt and response schema for a question.

        Returns:
            Tuple of (prompt_string, response_schema_class)
        """
        today_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        resolution_date = question.resolution_date or "Not specified"
        background = question.background or "No additional background provided."

        if question.question_type == QuestionType.BINARY:
            prompt = BINARY_PROMPT.format(
                question=question.text,
                background=background,
                today_date=today_date,
                resolution_date=resolution_date,
            )
            return prompt, BinaryForecastResponse

        elif question.question_type == QuestionType.CONTINUOUS:
            value_range_context = ""
            if question.value_range:
                low, high = question.value_range
                value_range_context = f"\nExpected range: {low} to {high}"

            prompt = CONTINUOUS_PROMPT.format(
                question=question.text,
                background=background,
                value_range_context=value_range_context,
                today_date=today_date,
                resolution_date=resolution_date,
            )
            return prompt, ContinuousForecastResponse

        elif question.question_type == QuestionType.QUANTILE:
            value_range_context = ""
            if question.value_range:
                low, high = question.value_range
                value_range_context = f"\nExpected range: {low} to {high}"

            quantiles = question.quantiles or [0.1, 0.25, 0.5, 0.75, 0.9]
            quantiles_str = ", ".join(f"{q:.0%}" for q in quantiles)

            prompt = QUANTILE_PROMPT.format(
                question=question.text,
                background=background,
                value_range_context=value_range_context,
                today_date=today_date,
                resolution_date=resolution_date,
                quantiles=quantiles_str,
            )
            return prompt, QuantileForecastResponse

        else:
            # Default to binary
            prompt = BINARY_PROMPT.format(
                question=question.text,
                background=background,
                today_date=today_date,
                resolution_date=resolution_date,
            )
            return prompt, BinaryForecastResponse

    async def forecast(self, question: Question) -> Forecast:
        """Generate a forecast for a question using the LLM with structured output.

        Automatically selects the appropriate prompt and response format based
        on the question type (binary, continuous, or quantile).

        Args:
            question: The question to forecast.

        Returns:
            A Forecast with the appropriate prediction fields populated.
        """
        prompt, response_schema = self._build_prompt(question)

        try:
            response = await litellm.acompletion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                response_format=response_schema,
            )

            content = response.choices[0].message.content
            parsed = response_schema.model_validate_json(content)

            # Build forecast based on question type
            if isinstance(parsed, BinaryForecastResponse):
                return Forecast(
                    question_id=question.id,
                    source=question.source,
                    forecaster=self.name,
                    probability=parsed.probability,
                    reasoning=parsed.reasoning,
                )
            elif isinstance(parsed, ContinuousForecastResponse):
                return Forecast(
                    question_id=question.id,
                    source=question.source,
                    forecaster=self.name,
                    point_estimate=parsed.point_estimate,
                    reasoning=parsed.reasoning,
                )
            elif isinstance(parsed, QuantileForecastResponse):
                return Forecast(
                    question_id=question.id,
                    source=question.source,
                    forecaster=self.name,
                    quantile_values=parsed.quantile_values,
                    reasoning=parsed.reasoning,
                )
            else:
                raise ValueError(f"Unknown response type: {type(parsed)}")

        except Exception as e:
            logger.error(f"Error forecasting question {question.id}: {e}")
            # Return a default forecast on error
            return Forecast(
                question_id=question.id,
                source=question.source,
                forecaster=self.name,
                probability=0.5 if question.question_type == QuestionType.BINARY else None,
                reasoning=f"Error: {e}",
            )
