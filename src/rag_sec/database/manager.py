from contextlib import asynccontextmanager
from functools import lru_cache
from typing import AsyncGenerator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from rag_sec.config import get_settings
from rag_sec.logging import get_logger
from rag_sec.models import Base


log = get_logger(__name__)


class DatabaseManager:
    """
    Manages the PostgreSQL database connection, sessions,
    pgvector extension, and database lifecycle.
    """

    def __init__(self, database_url: str):
        self.engine: AsyncEngine = create_async_engine(
            database_url,
            pool_pre_ping=True,
        )

        self.session_factory = async_sessionmaker(
            bind=self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

    async def initialize(self) -> None:
        """
        Initialize the database.

        - enables the pgvector extension
        - creates SQLAlchemy tables
        """

        async with self.engine.begin() as connection:

            # pgvector must exist before tables using VECTOR are created
            await connection.execute(
                text("CREATE EXTENSION IF NOT EXISTS vector")
            )

            # Base.metadata.create_all() is synchronous,
            # so run_sync bridges it into the async connection.
            await connection.run_sync(Base.metadata.create_all)

        log.info("database_initialized")

    async def health_check(self) -> bool:
        """
        Verify that PostgreSQL is reachable.
        """

        try:
            async with self.engine.connect() as connection:
                await connection.execute(text("SELECT 1"))

            return True

        except Exception as exc:
            log.error(
                "database_health_check_failed",
                error=str(exc),
            )
            return False

    @asynccontextmanager
    async def session(
        self,
    ) -> AsyncGenerator[AsyncSession, None]:
        """
        Provide a database session with automatic
        commit / rollback / close handling.
        """

        async with self.session_factory() as session:

            try:
                yield session

                await session.commit()

            except Exception:
                await session.rollback()
                raise

    async def close(self) -> None:
        """
        Close the database connection pool.
        """

        await self.engine.dispose()

        log.info("database_connections_closed")


@lru_cache(maxsize=1)
def get_database_manager() -> DatabaseManager:
    """
    Return the application DatabaseManager instance.
    """

    settings = get_settings()

    return DatabaseManager(
        database_url=settings.database_url
    )