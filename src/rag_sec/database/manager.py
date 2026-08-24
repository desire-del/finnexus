from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from functools import lru_cache

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
        async with self.engine.begin() as connection:
            await connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await connection.execute(
                text("CREATE EXTENSION IF NOT EXISTS pg_search CASCADE")
            )

            await connection.run_sync(Base.metadata.create_all)

            await connection.execute(text("DROP VIEW IF EXISTS active_chunks"))

            await connection.execute(
                text(
                    """
                    DO $$
                    BEGIN
                        IF EXISTS (
                            SELECT 1
                            FROM pg_attribute AS attribute
                            JOIN pg_class AS relation
                                ON relation.oid = attribute.attrelid
                            JOIN pg_namespace AS namespace
                                ON namespace.oid = relation.relnamespace
                            WHERE relation.relname = 'chunks'
                                AND namespace.nspname = current_schema()
                                AND attribute.attname = 'embedding'
                                AND format_type(
                                    attribute.atttypid,
                                    attribute.atttypmod
                                ) <> 'vector'
                        ) THEN
                            ALTER TABLE chunks
                            ALTER COLUMN embedding TYPE vector
                            USING embedding::vector;
                        END IF;
                    END
                    $$
                    """
                )
            )

            await connection.execute(
                text(
                    """
                    CREATE INDEX IF NOT EXISTS ix_chunks_text_bm25
                    ON chunks
                    USING bm25 (
                        id,
                        text,
                        filing_id,
                        processing_version_id
                    )
                    WITH (key_field = 'id')
                    """
                )
            )

            await connection.execute(
                text(
                    """
                    CREATE INDEX IF NOT EXISTS ix_chunks_text_fts_english
                    ON chunks
                    USING GIN (
                        to_tsvector('pg_catalog.english'::regconfig, text)
                    )
                    """
                )
            )

            await connection.execute(
                text(
                    """
                    CREATE OR REPLACE VIEW active_chunks AS

                    SELECT
                        c.id,
                        c.chunk_id,
                        c.filing_id,
                        c.processing_version_id,
                        c.chunk_index,

                        c.text,
                        c.embedding,

                        c.section,
                        c.part,
                        c.item,
                        c.page,
                        c.source_url,
                        c.token_count,

                        c.metadata,

                        f.accession_number,
                        f.form_type,
                        f.filing_date,

                        co.cik,
                        co.name AS company_name,
                        co.ticker,

                        pv.embedding_provider,
                        pv.embedding_model,
                        pv.embedding_dimension

                    FROM chunks AS c

                    JOIN processing_versions AS pv
                        ON pv.id = c.processing_version_id

                    JOIN filings AS f
                        ON f.id = c.filing_id

                    JOIN companies AS co
                        ON co.id = f.company_id

                    WHERE
                        pv.status = 'active'
                        AND c.embedding IS NOT NULL
                    """
                )
            )

        log.info("database_initialized")

    async def health_check(self) -> bool:
        """
        Verify that PostgreSQL is reachable.
        """

        try:
            async with self.engine.connect() as connection:
                await connection.execute(text("SELECT 1"))

            return True

        except Exception as exc:  # noqa: BLE001 - health check returns a boolean.
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

    return DatabaseManager(database_url=settings.database_url)
