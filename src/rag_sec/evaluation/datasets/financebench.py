import json
import re
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, overload

from rag_sec.evaluation.models import (
    EvaluationCase,
    ReferenceEvidence,
)

if TYPE_CHECKING:
    import pandas as pd

    from rag_sec.evaluation.suite import FinanceBenchSuite

_ACCESSION_PATTERN = re.compile(r"\b\d{10}-\d{2}-\d{6}\b")

FINANCEBENCH_COMPANY_TICKERS = {
    "3M": "MMM",
    "AES Corporation": "AES",
    "AMD": "AMD",
    "Activision Blizzard": "ATVI",
    "Adobe": "ADBE",
    "Amazon": "AMZN",
    "Amcor": "AMCR",
    "American Express": "AXP",
    "American Water Works": "AWK",
    "Best Buy": "BBY",
    "Block": "SQ",
    "Boeing": "BA",
    "CVS Health": "CVS",
    "Coca-Cola": "KO",
    "Corning": "GLW",
    "Costco": "COST",
    "Foot Locker": "FL",
    "General Mills": "GIS",
    "JPMorgan": "JPM",
    "Johnson & Johnson": "JNJ",
    "Kraft Heinz": "KHC",
    "Lockheed Martin": "LMT",
    "MGM Resorts": "MGM",
    "Microsoft": "MSFT",
    "Netflix": "NFLX",
    "Nike": "NKE",
    "Paypal": "PYPL",
    "PepsiCo": "PEP",
    "Pfizer": "PFE",
    "Ulta Beauty": "ULTA",
    "Verizon": "VZ",
    "Walmart": "WMT",
}

FINANCEBENCH_COMPANY_CIKS = {
    "Activision Blizzard": 718877,
    "Block": 1512673,
    "Foot Locker": 850209,
}


class FinanceBench(Sequence[EvaluationCase]):
    """Resolved, SEC-compatible FinanceBench cases for notebook evaluation."""

    def __init__(
        self,
        cases: Sequence[EvaluationCase],
        *,
        suite: "FinanceBenchSuite",
        source_case_count: int,
        excluded_case_count: int,
        unsupported_documents: dict[str, str],
    ) -> None:
        self._cases = tuple(cases)
        self._suite = suite
        self.metadata: dict[str, Any] = {
            "name": "financebench",
            "version": "open-source",
            "subset": "sec-compatible",
            "source_case_count": source_case_count,
            "case_count": len(self._cases),
            "excluded_case_count": excluded_case_count,
            "unsupported_documents": dict(unsupported_documents),
        }

    @classmethod
    async def load(cls, root: Path | str | None = None) -> "FinanceBench":
        """Load, resolve, and validate the SEC-compatible FinanceBench subset."""
        from rag_sec.evaluation.suite import (  # Avoid a module import cycle.
            DEFAULT_FINANCEBENCH_ROOT,
            FinanceBenchSuite,
        )

        suite = FinanceBenchSuite(
            Path(root) if root is not None else DEFAULT_FINANCEBENCH_ROOT
        )
        loaded = await suite.load()
        return cls(
            loaded.cases,
            suite=suite,
            source_case_count=len(loaded.resolution.cases),
            excluded_case_count=loaded.excluded_cases,
            unsupported_documents=loaded.resolution.unsupported_documents,
        )

    def __len__(self) -> int:
        return len(self._cases)

    @overload
    def __getitem__(self, index: int) -> EvaluationCase: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[EvaluationCase, ...]: ...

    def __getitem__(
        self, index: int | slice
    ) -> EvaluationCase | tuple[EvaluationCase, ...]:
        return self._cases[index]

    def __iter__(self) -> Iterator[EvaluationCase]:
        return iter(self._cases)

    async def validate_corpus(self) -> None:
        """Fail clearly when required production chunks are unavailable."""
        await self._suite.validate_corpus(list(self._cases))

    def to_records(self) -> list[dict[str, Any]]:
        """Return notebook-friendly rows while retaining normalized objects."""
        records = []
        for case in self._cases:
            metadata = case.metadata
            records.append(
                {
                    "case_id": case.id,
                    "dataset_name": case.dataset_name,
                    "dataset_version": case.dataset_version,
                    "subset_label": metadata.get("dataset_subset_label"),
                    "question": case.question,
                    "question_type": metadata.get("question_type"),
                    "question_reasoning": metadata.get("question_reasoning"),
                    "reference_answer": case.reference_answer,
                    "justification": metadata.get("justification"),
                    "company": case.company,
                    "ticker": case.ticker,
                    "cik": metadata.get("company_cik"),
                    "accession_number": case.accession_number,
                    "document_name": metadata.get("doc_name"),
                    "form_type": case.form_type,
                    "document_period": metadata.get("doc_period"),
                    "document_url": metadata.get("doc_link"),
                    "gics_sector": metadata.get("gics_sector"),
                    "domain_question_num": metadata.get("domain_question_num"),
                    "answerable": case.answerable,
                    "gold_evidence_count": len(case.reference_evidence),
                    "reference_evidence": case.reference_evidence,
                    "tags": case.tags,
                    "case": case,
                }
            )
        return records

    def to_dataframe(self) -> "pd.DataFrame":
        """Return one analysis row per normalized evaluation case."""
        import pandas as pd

        return pd.DataFrame.from_records(self.to_records())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return [json.loads(line) for line in file if line.strip()]


def normalize_form_type(
    doc_type: str | None,
) -> str | None:
    if not doc_type:
        return None

    normalized = doc_type.upper().replace("-", "")

    mapping = {
        "10K": "10-K",
        "10Q": "10-Q",
        "8K": "8-K",
        "10KANNUAL": "10-K",
    }

    return mapping.get(normalized)


def extract_accession_number(
    value: str | None,
) -> str | None:
    if not value:
        return None

    match = _ACCESSION_PATTERN.search(value)

    if match is None:
        return None

    return match.group(0)


def load_financebench(
    questions_path: Path,
    documents_path: Path,
) -> list[EvaluationCase]:
    questions = read_jsonl(questions_path)
    documents = read_jsonl(documents_path)

    document_by_name = {document["doc_name"]: document for document in documents}

    cases: list[EvaluationCase] = []

    for row in questions:
        doc_name = row["doc_name"]

        document = document_by_name.get(
            doc_name,
            {},
        )

        doc_link = document.get("doc_link")

        accession_number = extract_accession_number(doc_link)

        references = [
            ReferenceEvidence(
                text=evidence["evidence_text"],
                document_id=evidence.get(
                    "doc_name",
                    doc_name,
                ),
                accession_number=accession_number,
                page=evidence["evidence_page_num"],
                metadata={
                    "page_index_base": 0,
                    "full_page_text": evidence.get("evidence_text_full_page"),
                },
            )
            for evidence in row.get(
                "evidence",
                [],
            )
        ]

        company = row.get("company")
        company_key = str(company) if company is not None else ""
        case = EvaluationCase(
            id=f"financebench-{row['financebench_id']}",
            dataset_name="financebench",
            dataset_version="open-source",
            question=row["question"],
            ticker=FINANCEBENCH_COMPANY_TICKERS.get(company_key),
            company=company_key or None,
            form_type=normalize_form_type(document.get("doc_type")),
            accession_number=accession_number,
            reference_answer=row.get("answer"),
            reference_evidence=references,
            answerable=True,
            tags=[
                value
                for value in (
                    row.get("question_type"),
                    row.get("question_reasoning"),
                )
                if value
            ],
            metadata={
                "financebench_id": (row["financebench_id"]),
                "domain_question_num": row.get("domain_question_num"),
                "question_type": row.get("question_type"),
                "question_reasoning": row.get("question_reasoning"),
                "doc_name": doc_name,
                "doc_period": document.get("doc_period"),
                "doc_type": document.get("doc_type"),
                "doc_link": doc_link,
                "gics_sector": document.get("gics_sector"),
                "company_cik": FINANCEBENCH_COMPANY_CIKS.get(company_key),
                "justification": row.get("justification"),
                "dataset_subset_label": row.get("dataset_subset_label"),
            },
        )

        cases.append(case)

    return cases
