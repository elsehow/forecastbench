"""SQLite storage backend using SQLAlchemy."""

from datetime import date, datetime
from pathlib import Path

from sqlalchemy import Boolean, Date, DateTime, Float, String, Text, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from forecastbench.models import Forecast, Question, QuestionType, Resolution
from forecastbench.storage.base import Storage


class Base(DeclarativeBase):
    pass


class QuestionRow(Base):
    __tablename__ = "questions"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    source: Mapped[str] = mapped_column(String, primary_key=True)
    text: Mapped[str] = mapped_column(Text)
    background: Mapped[str | None] = mapped_column(Text, nullable=True)
    url: Mapped[str | None] = mapped_column(String, nullable=True)
    question_type: Mapped[str] = mapped_column(String, default="binary")
    created_at: Mapped[datetime] = mapped_column(DateTime)
    resolution_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    category: Mapped[str | None] = mapped_column(String, nullable=True)
    resolved: Mapped[bool] = mapped_column(Boolean, default=False)
    resolution_value: Mapped[float | None] = mapped_column(Float, nullable=True)


class ForecastRow(Base):
    __tablename__ = "forecasts"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    question_id: Mapped[str] = mapped_column(String)
    source: Mapped[str] = mapped_column(String)
    forecaster: Mapped[str] = mapped_column(String)
    probability: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime)
    reasoning: Mapped[str | None] = mapped_column(Text, nullable=True)


class ResolutionRow(Base):
    __tablename__ = "resolutions"

    question_id: Mapped[str] = mapped_column(String, primary_key=True)
    source: Mapped[str] = mapped_column(String, primary_key=True)
    date: Mapped[date] = mapped_column(Date, primary_key=True)
    value: Mapped[float] = mapped_column(Float)


class SQLiteStorage(Storage):
    """SQLite storage backend using async SQLAlchemy."""

    def __init__(self, db_path: str | Path = "forecastbench.db"):
        """Initialize SQLite storage.

        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = Path(db_path)
        self._engine = create_async_engine(
            f"sqlite+aiosqlite:///{self.db_path}",
            echo=False,
        )
        self._session_factory = async_sessionmaker(self._engine, expire_on_commit=False)
        self._initialized = False

    async def _ensure_initialized(self) -> None:
        """Create tables if they don't exist."""
        if not self._initialized:
            async with self._engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            self._initialized = True

    async def _get_session(self) -> AsyncSession:
        await self._ensure_initialized()
        return self._session_factory()

    def _question_to_row(self, question: Question) -> QuestionRow:
        return QuestionRow(
            id=question.id,
            source=question.source,
            text=question.text,
            background=question.background,
            url=question.url,
            question_type=question.question_type.value,
            created_at=question.created_at,
            resolution_date=question.resolution_date,
            category=question.category,
            resolved=question.resolved,
            resolution_value=question.resolution_value,
        )

    def _row_to_question(self, row: QuestionRow) -> Question:
        return Question(
            id=row.id,
            source=row.source,
            text=row.text,
            background=row.background,
            url=row.url,
            question_type=QuestionType(row.question_type),
            created_at=row.created_at,
            resolution_date=row.resolution_date,
            category=row.category,
            resolved=row.resolved,
            resolution_value=row.resolution_value,
        )

    def _forecast_to_row(self, forecast: Forecast) -> ForecastRow:
        return ForecastRow(
            question_id=forecast.question_id,
            source=forecast.source,
            forecaster=forecast.forecaster,
            probability=forecast.probability,
            created_at=forecast.created_at,
            reasoning=forecast.reasoning,
        )

    def _row_to_forecast(self, row: ForecastRow) -> Forecast:
        return Forecast(
            question_id=row.question_id,
            source=row.source,
            forecaster=row.forecaster,
            probability=row.probability,
            created_at=row.created_at,
            reasoning=row.reasoning,
        )

    def _resolution_to_row(self, resolution: Resolution) -> ResolutionRow:
        return ResolutionRow(
            question_id=resolution.question_id,
            source=resolution.source,
            date=resolution.date,
            value=resolution.value,
        )

    def _row_to_resolution(self, row: ResolutionRow) -> Resolution:
        return Resolution(
            question_id=row.question_id,
            source=row.source,
            date=row.date,
            value=row.value,
        )

    async def save_question(self, question: Question) -> None:
        async with await self._get_session() as session:
            row = self._question_to_row(question)
            await session.merge(row)
            await session.commit()

    async def save_questions(self, questions: list[Question]) -> None:
        async with await self._get_session() as session:
            for question in questions:
                row = self._question_to_row(question)
                await session.merge(row)
            await session.commit()

    async def get_question(self, source: str, question_id: str) -> Question | None:
        async with await self._get_session() as session:
            result = await session.execute(
                select(QuestionRow).where(
                    QuestionRow.source == source,
                    QuestionRow.id == question_id,
                )
            )
            row = result.scalar_one_or_none()
            return self._row_to_question(row) if row else None

    async def get_questions(
        self,
        source: str | None = None,
        resolved: bool | None = None,
        category: str | None = None,
        limit: int | None = None,
    ) -> list[Question]:
        async with await self._get_session() as session:
            stmt = select(QuestionRow)

            if source is not None:
                stmt = stmt.where(QuestionRow.source == source)
            if resolved is not None:
                stmt = stmt.where(QuestionRow.resolved == resolved)
            if category is not None:
                stmt = stmt.where(QuestionRow.category == category)

            stmt = stmt.order_by(QuestionRow.created_at.desc())

            if limit is not None:
                stmt = stmt.limit(limit)

            result = await session.execute(stmt)
            rows = result.scalars().all()
            return [self._row_to_question(row) for row in rows]

    async def save_forecast(self, forecast: Forecast) -> None:
        async with await self._get_session() as session:
            row = self._forecast_to_row(forecast)
            session.add(row)
            await session.commit()

    async def save_forecasts(self, forecasts: list[Forecast]) -> None:
        async with await self._get_session() as session:
            for forecast in forecasts:
                row = self._forecast_to_row(forecast)
                session.add(row)
            await session.commit()

    async def get_forecasts(
        self,
        question_id: str | None = None,
        source: str | None = None,
        forecaster: str | None = None,
        limit: int | None = None,
    ) -> list[Forecast]:
        async with await self._get_session() as session:
            stmt = select(ForecastRow)

            if question_id is not None:
                stmt = stmt.where(ForecastRow.question_id == question_id)
            if source is not None:
                stmt = stmt.where(ForecastRow.source == source)
            if forecaster is not None:
                stmt = stmt.where(ForecastRow.forecaster == forecaster)

            stmt = stmt.order_by(ForecastRow.created_at.desc())

            if limit is not None:
                stmt = stmt.limit(limit)

            result = await session.execute(stmt)
            rows = result.scalars().all()
            return [self._row_to_forecast(row) for row in rows]

    async def save_resolution(self, resolution: Resolution) -> None:
        async with await self._get_session() as session:
            row = self._resolution_to_row(resolution)
            await session.merge(row)
            await session.commit()

    async def get_resolution(
        self, source: str, question_id: str, resolution_date: date | None = None
    ) -> Resolution | None:
        async with await self._get_session() as session:
            stmt = select(ResolutionRow).where(
                ResolutionRow.source == source,
                ResolutionRow.question_id == question_id,
            )

            if resolution_date is not None:
                stmt = stmt.where(ResolutionRow.date == resolution_date)
            else:
                # Get the latest resolution
                stmt = stmt.order_by(ResolutionRow.date.desc())

            result = await session.execute(stmt)
            row = result.scalars().first()
            return self._row_to_resolution(row) if row else None

    async def get_resolutions(
        self,
        question_id: str | None = None,
        source: str | None = None,
    ) -> list[Resolution]:
        async with await self._get_session() as session:
            stmt = select(ResolutionRow)

            if question_id is not None:
                stmt = stmt.where(ResolutionRow.question_id == question_id)
            if source is not None:
                stmt = stmt.where(ResolutionRow.source == source)

            stmt = stmt.order_by(ResolutionRow.date.desc())

            result = await session.execute(stmt)
            rows = result.scalars().all()
            return [self._row_to_resolution(row) for row in rows]

    async def close(self) -> None:
        await self._engine.dispose()
