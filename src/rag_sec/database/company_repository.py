from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_sec.models.company import Company
from rag_sec.schemas.company import CompanyCreate


class CompanyRepository:
    @staticmethod
    async def get_by_cik(session: AsyncSession, cik: int) -> Company | None:
        result = await session.execute(select(Company).where(Company.cik == cik))
        return result.scalar_one_or_none()

    @staticmethod
    async def get_by_ticker(session: AsyncSession, ticker: str) -> Company | None:
        result = await session.execute(
            select(Company).where(Company.ticker == ticker.upper())
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def create(session: AsyncSession, data: CompanyCreate) -> Company:
        company = Company(cik=data.cik, name=data.name, ticker=data.ticker)
        session.add(company)
        await session.flush()
        await session.refresh(company)
        return company

    @classmethod
    async def get_or_create(cls, session: AsyncSession, data: CompanyCreate) -> Company:
        company = await cls.get_by_cik(session, data.cik)
        return company or await cls.create(session, data)
