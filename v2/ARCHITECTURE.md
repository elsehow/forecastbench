# ForecastBench v2 Architecture

## Goals

1. **Fewer lines of code**: ~3,000 LOC (down from ~17,500)
2. **More extensible**: Plugin-based sources, config-driven models
3. **Better tested**: Integration tests with real data
4. **Better data collection**: SQLite + queryable historical data

## Current Pain Points

| Area | Current State | Impact |
|------|--------------|--------|
| Question sources | 3,859 LOC, 95% copy-paste boilerplate | Hard to add new sources |
| LLM integration | 1,025 LOC across 6 provider-specific functions | Breaks with each new model |
| Testing | 0% coverage | Risky deployments |
| Data storage | JSONL files in GCS, no queryability | Can't analyze historical data |
| Configuration | Scattered across 5+ files, manual dicts | Error-prone, no validation |

## Key Design Decisions

### 1. LiteLLM for All LLM Calls

Single unified interface replaces 6 provider-specific implementations:

```python
import litellm

async def get_forecast(model: str, prompt: str) -> str:
    response = await litellm.acompletion(
        model=model,  # "openai/gpt-4o", "anthropic/claude-3-opus", etc.
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content
```

### 2. Abstract Question Source Pattern

Each source implements a simple interface:

```python
class QuestionSource(ABC):
    name: str

    @abstractmethod
    async def fetch_questions(self) -> list[Question]: ...

    @abstractmethod
    async def fetch_resolution(self, question_id: str) -> float | None: ...
```

New sources = one file, auto-registered via decorator.

### 3. SQLite for Local Storage

Queryable database replaces JSONL files:

- Questions, forecasts, resolutions as tables
- Full SQL query support
- Easy export to Parquet/HuggingFace
- GCS sync for production

### 4. Pydantic for All Data Models

Type-safe, validated data throughout:

```python
class Question(BaseModel):
    id: str
    source: str
    text: str
    background: str | None
    resolution_date: date
    category: str | None
    resolved: bool = False
```

## Directory Structure

```
v2/
├── pyproject.toml
├── config/
│   └── settings.yaml
├── src/forecastbench/
│   ├── __init__.py
│   ├── config.py           # Pydantic settings
│   ├── models.py           # Data models (Question, Forecast, Resolution)
│   ├── sources/            # Question sources
│   │   ├── __init__.py     # Registry
│   │   ├── base.py         # Abstract base
│   │   ├── manifold.py
│   │   ├── metaculus.py
│   │   └── ...
│   ├── forecasters/        # LLM forecasting
│   │   ├── __init__.py
│   │   ├── base.py
│   │   └── llm.py          # LiteLLM implementation
│   ├── scoring.py          # Brier scores, stats
│   ├── storage/            # Data layer
│   │   ├── __init__.py
│   │   ├── base.py
│   │   └── sqlite.py
│   └── cli.py              # CLI interface
└── tests/
    ├── conftest.py
    ├── test_sources.py     # Integration tests with real APIs
    ├── test_forecasters.py
    └── test_storage.py
```

## Migration Plan

### Phase 1: Core Infrastructure (Current)
- Data models (Pydantic)
- Abstract source interface
- First concrete source (Manifold)
- LiteLLM forecaster
- SQLite storage
- Integration tests

### Phase 2: More Sources
- Metaculus
- Polymarket
- FRED, ACLED, etc.

### Phase 3: Pipeline & CLI
- Orchestration logic
- CLI commands
- Scheduling

### Phase 4: Production
- GCS sync
- Cloud Run deployment
- Website integration

## Testing Philosophy

- Prefer integration tests over unit tests
- Test with real API data where possible
- Use pytest fixtures for common setup
- Keep tests simple and readable
