"""Forecasters for ForecastBench."""

from forecastbench.forecasters.base import Forecaster
from forecastbench.forecasters.llm import LLMForecaster

__all__ = ["Forecaster", "LLMForecaster"]
