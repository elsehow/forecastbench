"""Integration tests for question sources using real API data."""

import pytest

from forecastbench.models import QuestionType
from forecastbench.sources import registry
from forecastbench.sources.manifold import ManifoldSource


class TestManifoldSource:
    """Integration tests for Manifold Markets source."""

    @pytest.fixture
    def source(self):
        return ManifoldSource()

    async def test_fetch_questions_returns_questions(self, source: ManifoldSource):
        """Fetch real questions from Manifold API."""
        questions = await source.fetch_questions()

        assert len(questions) > 0, "Should fetch at least one question"

        # Check the first question has required fields
        q = questions[0]
        assert q.id, "Question should have an ID"
        assert q.source == "manifold"
        assert q.text, "Question should have text"
        assert q.question_type == QuestionType.BINARY

    async def test_fetch_questions_have_valid_data(self, source: ManifoldSource):
        """Verify fetched questions have valid data."""
        questions = await source.fetch_questions()

        for q in questions[:5]:  # Check first 5
            assert q.id is not None
            assert len(q.text) > 0
            assert q.url is None or q.url.startswith("http")

    async def test_fetch_resolution_for_question(self, source: ManifoldSource):
        """Fetch resolution for a real question."""
        questions = await source.fetch_questions()
        assert len(questions) > 0

        # Get resolution for the first question
        resolution = await source.fetch_resolution(questions[0].id)

        # Open markets should have a current probability
        if resolution is not None:
            assert 0 <= resolution.value <= 1
            assert resolution.source == "manifold"
            assert resolution.question_id == questions[0].id

    async def test_source_is_registered(self):
        """Verify Manifold source is in the registry."""
        assert "manifold" in registry
        assert registry.get("manifold") is ManifoldSource


class TestSourceRegistry:
    """Tests for the source registry."""

    def test_list_sources(self):
        """List all registered sources."""
        sources = registry.list()
        assert "manifold" in sources

    def test_get_unknown_source_raises(self):
        """Getting unknown source raises KeyError."""
        with pytest.raises(KeyError):
            registry.get("nonexistent_source")
