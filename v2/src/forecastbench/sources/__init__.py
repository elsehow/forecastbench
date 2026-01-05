"""Question sources for ForecastBench."""

from forecastbench.sources.base import QuestionSource, registry

# Import all sources to trigger registration
from forecastbench.sources.fred import FREDSource
from forecastbench.sources.infer import INFERSource
from forecastbench.sources.manifold import ManifoldSource
from forecastbench.sources.metaculus import MetaculusSource
from forecastbench.sources.polymarket import PolymarketSource
from forecastbench.sources.yfinance import YahooFinanceSource


def get_all_sources() -> dict[str, type[QuestionSource]]:
    """Get all registered source classes.

    Returns:
        Dict mapping source name to source class.
    """
    return {name: registry.get(name) for name in registry.list()}


__all__ = [
    "QuestionSource",
    "registry",
    "get_all_sources",
    "FREDSource",
    "INFERSource",
    "ManifoldSource",
    "MetaculusSource",
    "PolymarketSource",
    "YahooFinanceSource",
]
