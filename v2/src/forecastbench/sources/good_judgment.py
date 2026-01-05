"""Good Judgment Open source.

Good Judgment Open is a public forecasting platform with questions on
geopolitics, economics, science, and more.
"""

import logging
from datetime import date, datetime, timezone

import httpx

from forecastbench.models import Question, QuestionType, Resolution, SourceType
from forecastbench.sources.base import QuestionSource, registry

logger = logging.getLogger(__name__)

# Good Judgment Open uses a REST API
# Note: API details may need to be updated based on actual API documentation
BASE_URL = "https://www.gjopen.com/api/v1"


@registry.register
class GoodJudgmentSource(QuestionSource):
    """Fetch questions from Good Judgment Open.

    Good Judgment Open is a public forecasting platform run by Good Judgment Inc.,
    founded by Philip Tetlock (author of Superforecasting).
    """

    name = "good_judgment"

    def __init__(
        self,
        api_key: str | None = None,
        http_client: httpx.AsyncClient | None = None,
    ):
        self._api_key = api_key
        self._client = http_client

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            headers = {}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"
            self._client = httpx.AsyncClient(timeout=30.0, headers=headers)
        return self._client

    async def _get_questions_page(
        self,
        page: int = 1,
        per_page: int = 50,
        status: str = "open",
    ) -> dict:
        """Get a page of questions from Good Judgment Open."""
        client = await self._get_client()
        params = {
            "page": page,
            "per_page": per_page,
            "status": status,
        }

        response = await client.get(f"{BASE_URL}/questions", params=params)
        response.raise_for_status()
        return response.json()

    async def _get_question(self, question_id: str) -> dict:
        """Get a specific question by ID."""
        client = await self._get_client()
        response = await client.get(f"{BASE_URL}/questions/{question_id}")
        response.raise_for_status()
        return response.json()

    def _parse_question(self, data: dict) -> Question:
        """Convert Good Judgment question data to a Question."""
        # Parse timestamps
        created_str = data.get("created_at") or data.get("publish_time")
        created_at = datetime.now(timezone.utc)
        if created_str:
            try:
                created_at = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
            except ValueError:
                pass

        close_str = data.get("close_time") or data.get("resolution_date")
        resolution_date = None
        if close_str:
            try:
                resolution_date = datetime.fromisoformat(
                    close_str.replace("Z", "+00:00")
                ).date()
            except ValueError:
                pass

        # Determine resolution status
        resolved = data.get("status") in ("resolved", "closed")
        resolution_value = None
        if resolved:
            resolution = data.get("resolution")
            if resolution == "yes":
                resolution_value = 1.0
            elif resolution == "no":
                resolution_value = 0.0
            elif isinstance(resolution, (int, float)):
                resolution_value = float(resolution)

        # Get current forecast as base rate
        base_rate = None
        crowd_forecast = data.get("crowd_forecast") or data.get("community_prediction")
        if crowd_forecast is not None:
            try:
                base_rate = float(crowd_forecast)
            except (ValueError, TypeError):
                pass

        return Question(
            id=str(data["id"]),
            source=self.name,
            source_type=SourceType.MARKET,
            text=data.get("title") or data.get("question"),
            background=data.get("description") or data.get("background"),
            url=data.get("url") or f"https://www.gjopen.com/questions/{data['id']}",
            question_type=QuestionType.BINARY,
            created_at=created_at,
            resolution_date=resolution_date,
            category=data.get("category") or data.get("topic"),
            resolved=resolved,
            resolution_value=resolution_value,
            base_rate=base_rate,
        )

    async def fetch_questions(self) -> list[Question]:
        """Fetch open questions from Good Judgment Open."""
        questions = []
        page = 1

        while True:
            try:
                data = await self._get_questions_page(page=page)
            except httpx.HTTPError as e:
                logger.warning(f"Failed to fetch page {page}: {e}")
                break

            page_questions = data.get("questions") or data.get("data") or []
            if not page_questions:
                break

            for q_data in page_questions:
                try:
                    questions.append(self._parse_question(q_data))
                except Exception as e:
                    logger.warning(f"Failed to parse question {q_data.get('id')}: {e}")

            # Check for more pages
            total_pages = data.get("total_pages") or data.get("pages") or 1
            if page >= total_pages:
                break
            page += 1

        logger.info(f"Fetched {len(questions)} questions from Good Judgment Open")
        return questions

    async def fetch_resolution(self, question_id: str) -> Resolution | None:
        """Fetch resolution for a specific question."""
        try:
            data = await self._get_question(question_id)
        except httpx.HTTPError as e:
            logger.warning(f"Failed to fetch question {question_id}: {e}")
            return None

        status = data.get("status")
        if status in ("resolved", "closed"):
            resolution = data.get("resolution")
            if resolution == "yes":
                value = 1.0
            elif resolution == "no":
                value = 0.0
            elif isinstance(resolution, (int, float)):
                value = float(resolution)
            else:
                return None

            return Resolution(
                question_id=question_id,
                source=self.name,
                date=date.today(),
                value=value,
            )

        # Return current crowd forecast as interim value
        crowd_forecast = data.get("crowd_forecast") or data.get("community_prediction")
        if crowd_forecast is not None:
            try:
                return Resolution(
                    question_id=question_id,
                    source=self.name,
                    date=date.today(),
                    value=float(crowd_forecast),
                )
            except (ValueError, TypeError):
                pass

        return None

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None
