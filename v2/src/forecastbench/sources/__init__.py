"""Question sources for ForecastBench."""

from forecastbench.sources.base import QuestionSource, registry

# Import all sources to trigger registration
from forecastbench.sources.fred import FREDSource
from forecastbench.sources.infer import INFERSource
from forecastbench.sources.manifold import ManifoldSource
from forecastbench.sources.metaculus import MetaculusSource
from forecastbench.sources.polymarket import PolymarketSource
from forecastbench.sources.yfinance import YahooFinanceSource

__all__ = [
    "QuestionSource",
    "registry",
    "FREDSource",
    "INFERSource",
    "ManifoldSource",
    "MetaculusSource",
    "PolymarketSource",
    "YahooFinanceSource",
]
