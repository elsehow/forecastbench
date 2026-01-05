"""Tests for forecasters."""

import pytest

from forecastbench.forecasters.llm import ForecastResponse, LLMForecaster
from forecastbench.models import Question


class TestForecastResponse:
    """Tests for the structured forecast response model."""

    def test_valid_probability(self):
        """Create response with valid probability."""
        resp = ForecastResponse(probability=0.75, reasoning="Test reasoning")
        assert resp.probability == 0.75

    def test_probability_bounds(self):
        """Probability must be between 0 and 1."""
        with pytest.raises(ValueError):
            ForecastResponse(probability=1.5, reasoning="Too high")
        with pytest.raises(ValueError):
            ForecastResponse(probability=-0.1, reasoning="Too low")

    def test_parse_json(self):
        """Parse JSON response."""
        json_str = '{"probability": 0.65, "reasoning": "Based on analysis..."}'
        resp = ForecastResponse.model_validate_json(json_str)
        assert resp.probability == 0.65
        assert "analysis" in resp.reasoning


class TestLLMForecaster:
    """Tests for LLM forecaster."""

    @pytest.fixture
    def sample_question(self) -> Question:
        return Question(
            id="test-q",
            source="test",
            text="Will the S&P 500 close above 5000 by end of January 2025?",
            background="The S&P 500 is currently trading around 4900.",
        )

    def test_forecaster_name(self):
        """Forecaster name matches model."""
        forecaster = LLMForecaster(model="openai/gpt-4o-mini")
        assert forecaster.name == "openai/gpt-4o-mini"

    def test_build_prompt(self, sample_question: Question):
        """Verify prompt is built correctly."""
        forecaster = LLMForecaster()
        prompt = forecaster._build_prompt(sample_question)

        assert "S&P 500" in prompt
        assert "5000" in prompt
        assert "superforecaster" in prompt

    @pytest.mark.skip(reason="Requires API key - run manually with OPENAI_API_KEY set")
    async def test_forecast_with_real_api(self, sample_question: Question):
        """Integration test with real LLM API."""
        forecaster = LLMForecaster(model="openai/gpt-4o-mini")
        forecast = await forecaster.forecast(sample_question)

        assert 0 <= forecast.probability <= 1
        assert forecast.question_id == sample_question.id
        assert forecast.forecaster == "openai/gpt-4o-mini"
        assert forecast.reasoning is not None
