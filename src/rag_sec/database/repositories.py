"""Backward-compatible repository exports.

New code should import repositories from their dedicated modules.
"""

from rag_sec.database.company_repository import CompanyRepository
from rag_sec.database.filing_repository import FilingRepository
from rag_sec.database.ingestion_repository import IngestionRepository
from rag_sec.database.processing_repository import ProcessingRepository

__all__ = [
    "CompanyRepository",
    "FilingRepository",
    "IngestionRepository",
    "ProcessingRepository",
]
