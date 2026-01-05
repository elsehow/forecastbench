"""ForecastBench v2 - A dynamic benchmark for LLM forecasting accuracy."""

from forecastbench.models import (
    Forecast,
    ForecastScore,
    Question,
    QuestionType,
    Resolution,
    SourceType,
)

__all__ = [
    "Question",
    "QuestionType",
    "SourceType",
    "Forecast",
    "Resolution",
    "ForecastScore",
]
__version__ = "2.0.0"
