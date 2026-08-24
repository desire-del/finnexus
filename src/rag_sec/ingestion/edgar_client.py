# src/rag_sec/ingestion/edgar_client.py

import asyncio
import hashlib
from datetime import UTC, date, datetime
from uuid import UUID

from edgar import Company, get_by_accession_number, set_identity

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

        log.info("edgar_client_initialized")

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
            filings = filings.filter(amendments=False)

        filing = filings.latest()

        if filing is None:
            raise ValueError(f"No {form_type} filing found for {identifier}.")

        log.info(
            "edgar_filing_discovered",
            cik=filing.cik,
            company=filing.company,
            form=filing.form,
            accession_number=filing.accession_number,
        )

        return company, filing

    async def get_filing_by_accession(
        self,
        accession_number: str,
    ):
        """Resolve an exact SEC filing and its company by accession number."""
        normalized_accession = accession_number.strip()

        if not normalized_accession:
            raise ValueError("Accession number cannot be empty.")

        def load():
            filing = get_by_accession_number(normalized_accession)

            if filing is None:
                raise ValueError(
                    f"SEC filing not found for accession number {normalized_accession}."
                )

            company = filing.get_entity()

            if company is None:
                raise ValueError(
                    f"Unable to resolve company for filing {normalized_accession}."
                )

            return company, filing

        return await asyncio.to_thread(load)

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
        metadata = EdgarClient._filing_metadata(filing)

        return FilingCreate(
            company_id=company_id,
            accession_number=filing.accession_number,
            form_type=filing.form,
            filing_date=EdgarClient._to_date(filing.filing_date),
            period_of_report=EdgarClient._to_optional_date(filing.period_of_report),
            acceptance_datetime=metadata["acceptance_datetime"],
            file_number=metadata["file_number"],
            primary_document=metadata["primary_document"],
            primary_document_description=(metadata["primary_document_description"]),
            # Complete SEC submission
            source_uri=filing.text_url,
            # Main document
            filing_url=filing.filing_url,
            # SEC landing page
            homepage_url=filing.homepage_url,
            is_xbrl=metadata["is_xbrl"],
            is_inline_xbrl=metadata["is_inline_xbrl"],
            is_amendment=filing.form.endswith("/A"),
        )

    @staticmethod
    def _filing_metadata(filing) -> dict:
        """Extract optional metadata from the current EdgarTools model."""
        sgml = filing.sgml()
        header = sgml.header
        attachments = sgml.attachments

        file_numbers = header.file_numbers
        file_number = next(
            (value for value in file_numbers if value),
            None,
        )

        primary_documents = attachments.primary_documents
        primary = primary_documents[0] if primary_documents else None

        attachment_types = {
            str(
                getattr(
                    attachment,
                    "document_type",
                    "",
                )
            ).upper()
            for attachment in attachments
        }
        attachment_descriptions = {
            str(
                getattr(
                    attachment,
                    "description",
                    "",
                )
            ).upper()
            for attachment in attachments
        }
        xbrl_markers = attachment_types | attachment_descriptions

        is_xbrl = any("XBRL" in marker for marker in xbrl_markers)
        is_inline_xbrl = any(
            "INLINE XBRL" in marker or "IXBRL" in marker for marker in xbrl_markers
        )

        return {
            "acceptance_datetime": (header.acceptance_datetime),
            "file_number": file_number,
            "primary_document": (getattr(primary, "document", None)),
            "primary_document_description": (getattr(primary, "description", None)),
            "is_xbrl": is_xbrl,
            "is_inline_xbrl": is_inline_xbrl,
        }

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

        content = await asyncio.to_thread(filing.markdown)

        if not content:
            raise ValueError("EdgarTools returned empty filing content.")

        content_hash = self._compute_hash(content)

        source_size = getattr(
            filing,
            "size",
            None,
        )

        if source_size is None:
            source_size = len(content.encode("utf-8"))

        log.info(
            "edgar_filing_fetched",
            accession_number=(filing.accession_number),
            content_length=len(content),
            content_hash=content_hash,
        )

        return FilingContent(
            filing_id=filing_id,
            accession_number=(filing.accession_number),
            content=content,
            content_hash=content_hash,
            source_size_bytes=int(source_size),
            source_uri=filing.text_url,
            fetched_at=datetime.now(UTC),
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
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @staticmethod
    def _to_date(
        value,
    ) -> date:
        if isinstance(value, date):
            return value

        return date.fromisoformat(str(value))

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

                    warnings = (
                        getattr(
                            section,
                            "warnings",
                            [],
                        )
                        or []
                    )

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
                            warnings=[str(w) for w in warnings],
                        )
                    )

        # Fallback
        if not sections:
            content = filing.markdown()

            if not content:
                raise ValueError("Unable to extract filing content.")

            sections.append(
                FilingSection(
                    name="document",
                    content=content,
                )
            )

        return sections
