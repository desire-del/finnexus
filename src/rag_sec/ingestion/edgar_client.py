# src/rag_sec/ingestion/edgar_client.py

import asyncio
import hashlib

from datetime import date, datetime, timezone
from uuid import UUID

from edgar import Company, set_identity

from rag_sec.config import get_settings
from rag_sec.logging import get_logger

from rag_sec.schemas.company import CompanyCreate
from rag_sec.schemas.filing import (
    FilingContent,
    FilingCreate,
    FilingSection,
)


log = get_logger(__name__)


class EdgarClient:
    """
    Adapter around EdgarTools.

    Responsible for:
    - resolving companies
    - discovering SEC filings
    - downloading filing content
    - mapping EdgarTools objects to FinNexus schemas
    """

    def __init__(self) -> None:
        settings = get_settings()

        set_identity(settings.edgar.identity)

        log.info(
            "edgar_client_initialized"
        )

    async def get_company(
        self,
        identifier: str | int,
    ):
        """
        Resolve a company from a ticker or CIK.

        Examples:
            AAPL
            MSFT
            320193
        """

        return await asyncio.to_thread(
            Company,
            identifier,
        )

    async def get_latest_filing(
        self,
        identifier: str | int,
        form_type: str = "10-K",
        *,
        include_amendments: bool = False,
    ):
        """
        Get the latest filing for a company.
        """

        company = await self.get_company(identifier)

        filings = await asyncio.to_thread(
            company.get_filings,
            form=form_type,
        )

        if not include_amendments:
            filings = filings.filter(
                amendments=False
            )

        filing = filings.latest()

        if filing is None:
            raise ValueError(
                f"No {form_type} filing found "
                f"for {identifier}."
            )

        log.info(
            "edgar_filing_discovered",
            cik=filing.cik,
            company=filing.company,
            form=filing.form,
            accession_number=filing.accession_number,
        )

        return company, filing

    @staticmethod
    def to_company_schema(
        company,
    ) -> CompanyCreate:
        return CompanyCreate(
            cik=int(company.cik),
            name=company.name,
            ticker=company.get_ticker(),
        )

    @staticmethod
    def to_filing_schema(
        filing,
        company_id: UUID,
    ) -> FilingCreate:
        return FilingCreate(
            company_id=company_id,

            accession_number=filing.accession_number,
            form_type=filing.form,

            filing_date=EdgarClient._to_date(
                filing.filing_date
            ),

            period_of_report=EdgarClient._to_optional_date(
                filing.report_date
            ),

            acceptance_datetime=filing.acceptance_datetime,

            file_number=filing.file_number or None,

            primary_document=(
                filing.primary_document or None
            ),

            primary_document_description=(
                filing.primary_doc_description or None
            ),

            # Complete SEC submission
            source_uri=filing.text_url,

            # Main document
            filing_url=filing.filing_url,

            # SEC landing page
            homepage_url=filing.homepage_url,

            is_xbrl=bool(filing.is_xbrl),

            is_inline_xbrl=bool(
                filing.is_inline_xbrl
            ),

            is_amendment=filing.form.endswith("/A"),
        )

    async def fetch_content(
        self,
        filing,
        filing_id: UUID,
    ) -> FilingContent:
        """
        Download and extract the filing content.

        Markdown is used because it preserves more
        document structure than plain text.
        """

        content = await asyncio.to_thread(
            filing.markdown
        )

        if not content:
            raise ValueError(
                "EdgarTools returned empty filing content."
            )

        content_hash = self._compute_hash(content)

        source_size = getattr(
            filing,
            "size",
            None,
        )

        if source_size is None:
            source_size = len(
                content.encode("utf-8")
            )

        log.info(
            "edgar_filing_fetched",
            accession_number=(
                filing.accession_number
            ),
            content_length=len(content),
            content_hash=content_hash,
        )

        return FilingContent(
            filing_id=filing_id,

            accession_number=(
                filing.accession_number
            ),

            content=content,

            content_hash=content_hash,

            source_size_bytes=int(source_size),

            source_uri=filing.text_url,

            fetched_at=datetime.now(
                timezone.utc
            ),
        )

    async def extract_sections(
        self,
        filing,
    ) -> list[FilingSection]:
        """
        Extract structured SEC sections using EdgarTools.

        Falls back to the complete filing markdown if
        structured section extraction is unavailable.
        """

        return await asyncio.to_thread(
            self._extract_sections_sync,
            filing,
        )

    @staticmethod
    def _compute_hash(
        content: str,
    ) -> str:
        return hashlib.sha256(
            content.encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _to_date(
        value,
    ) -> date:
        if isinstance(value, date):
            return value

        return date.fromisoformat(
            str(value)
        )

    @staticmethod
    def _to_optional_date(
        value,
    ) -> date | None:
        if value in (None, ""):
            return None

        return EdgarClient._to_date(value)

    @staticmethod
    def _extract_sections_sync(
        filing,
    ) -> list[FilingSection]:

        report = filing.obj()

        sections = []

        if report is not None:
            report_sections = getattr(
                report,
                "sections",
                None,
            )

            if report_sections:
                for name, section in report_sections.items():

                    content = section.markdown()

                    if not content:
                        continue

                    warnings = getattr(
                        section,
                        "warnings",
                        [],
                    ) or []

                    sections.append(
                        FilingSection(
                            name=name,
                            part=getattr(
                                section,
                                "part",
                                None,
                            ),
                            item=getattr(
                                section,
                                "item",
                                None,
                            ),
                            content=content,
                            warnings=[
                                str(w)
                                for w in warnings
                            ],
                        )
                    )

        # Fallback
        if not sections:
            content = filing.markdown()

            if not content:
                raise ValueError(
                    "Unable to extract filing content."
                )

            sections.append(
                FilingSection(
                    name="document",
                    content=content,
                )
            )

        return sections
        