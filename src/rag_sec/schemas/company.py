from datetime import datetime
from uuid import UUID

from pydantic import Field, field_validator

from rag_sec.schemas.base import FinNexusSchema


class CompanyBase(FinNexusSchema):
    cik: int = Field(
        gt=0,
        description="SEC Central Index Key.",
    )

    name: str = Field(
        min_length=1,
        description="Official company name.",
    )

    ticker: str | None = Field(
        default=None,
        description="Primary stock ticker when available.",
    )

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, value: str | None) -> str | None:
        if value is None:
            return None

        return value.upper()


class CompanyCreate(CompanyBase):
    pass


class CompanyRead(CompanyBase):
    id: UUID
    created_at: datetime
    updated_at: datetime
