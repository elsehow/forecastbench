"""Storage backends for ForecastBench."""

from forecastbench.storage.base import Storage
from forecastbench.storage.sqlite import SQLiteStorage

__all__ = ["Storage", "SQLiteStorage"]
