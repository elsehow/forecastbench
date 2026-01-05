"""LLM-based forecaster using LiteLLM with structured outputs."""

import logging
from datetime import datetime, timezone

import litellm
from pydantic import BaseModel, Field

from forecastbench.forecasters.base import Forecaster
from forecastbench.models import Forecast, Question

logger = logging.getLogger(__name__)

# Prompt adapted from Halawi et al. (2024) "Approaching Human-Level Forecasting with Language Models"
FORECAST_PROMPT = """You are an expert superforecaster, familiar with the work of Tetlock and others. \
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


class ForecastResponse(BaseModel):
    """Structured output for forecast responses."""

    probability: float = Field(
        ge=0.0,
        le=1.0,
        description="Probability estimate between 0 and 1",
    )
    reasoning: str = Field(
        description="Brief reasoning for the forecast",
    )


class LLMForecaster(Forecaster):
    """Forecaster that uses LLMs via LiteLLM with structured outputs."""

    def __init__(
        self,
        model: str = "openai/gpt-4o-mini",
        temperature: float = 0.0,
    ):
        """Initialize the LLM forecaster.

        Args:
            model: LiteLLM model identifier (e.g., "openai/gpt-4o", "anthropic/claude-3-opus").
            temperature: Sampling temperature (0.0 for deterministic).
        """
        self.model = model
        self.temperature = temperature

    @property
    def name(self) -> str:
        return self.model

    def _build_prompt(self, question: Question) -> str:
        """Build the forecast prompt for a question."""
        return FORECAST_PROMPT.format(
            question=question.text,
            background=question.background or "No additional background provided.",
            today_date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            resolution_date=question.resolution_date or "Not specified",
        )

    async def forecast(self, question: Question) -> Forecast:
        """Generate a forecast for a question using the LLM with structured output.

        Args:
            question: The question to forecast.

        Returns:
            A Forecast with the predicted probability.
        """
        prompt = self._build_prompt(question)

        try:
            response = await litellm.acompletion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                response_format=ForecastResponse,
            )

            # Parse the structured response
            content = response.choices[0].message.content
            forecast_response = ForecastResponse.model_validate_json(content)

            return Forecast(
                question_id=question.id,
                source=question.source,
                forecaster=self.name,
                probability=forecast_response.probability,
                reasoning=forecast_response.reasoning,
            )

        except Exception as e:
            logger.error(f"Error forecasting question {question.id}: {e}")
            # Return a default forecast on error
            return Forecast(
                question_id=question.id,
                source=question.source,
                forecaster=self.name,
                probability=0.5,
                reasoning=f"Error: {e}",
            )
