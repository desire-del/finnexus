from typing import TYPE_CHECKING

from sqlalchemy import BigInteger, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from rag_sec.models.base import (
    Base,
    TimestampMixin,
    UUIDMixin,
)

if TYPE_CHECKING:
    from rag_sec.models.filing import Filing


class Company(
    Base,
    UUIDMixin,
    TimestampMixin,
):
    __tablename__ = "companies"

    cik: Mapped[int] = mapped_column(
        BigInteger,
        unique=True,
        nullable=False,
        index=True,
    )

    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    ticker: Mapped[str | None] = mapped_column(
        String(20),
        nullable=True,
        index=True,
    )

    filings: Mapped[list["Filing"]] = relationship(
        back_populates="company",
        cascade="all, delete-orphan",
    )
