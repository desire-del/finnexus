import asyncio
import json
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from rag_sec.evaluation.evaluators.matching import normalize_text, token_recall
from rag_sec.evaluation.models import EvaluationCase
from rag_sec.ingestion.edgar_client import EdgarClient
from rag_sec.ingestion.pipeline import IngestionPipeline
from rag_sec.logging import get_logger

_DATED_DOCUMENT_PATTERN = re.compile(
    r"dated[-_](\d{4}-\d{2}-\d{2})",
    re.IGNORECASE,
)
_QUARTER_DOCUMENT_PATTERN = re.compile(
    r"_(\d{4})Q([1-4])_",
    re.IGNORECASE,
)
_QUARTER_END_MONTHS = {
    1: {2, 3, 4},
    2: {5, 6, 7},
    3: {8, 9, 10},
    4: {1, 11, 12},
}
log = get_logger(__name__)


@dataclass
class AccessionResolutionResult:
    cases: list[EvaluationCase]
    resolved_documents: dict[str, str] = field(default_factory=dict)
    cached_documents: dict[str, str] = field(default_factory=dict)
    unresolved_documents: dict[str, str] = field(default_factory=dict)
    unsupported_documents: dict[str, str] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return not self.unresolved_documents

    @property
    def unique_accessions(self) -> list[str]:
        return get_unique_accessions(self.cases)


def _read_accession_cache(path: Path | None) -> dict[str, str]:
    if path is None or not path.is_file():
        return {}

    content = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(content, dict):
        raise TypeError("FinanceBench accession cache must be a JSON object.")

    return {
        str(document_name): str(accession_number)
        for document_name, accession_number in content.items()
        if accession_number
    }


def _write_accession_cache(path: Path | None, cache: dict[str, str]) -> None:
    if path is None:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(dict(sorted(cache.items())), indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _document_name(case: EvaluationCase) -> str:
    value = case.metadata.get("doc_name")

    if not value:
        raise ValueError(f"Evaluation case {case.id} has no FinanceBench doc_name.")

    return str(value)


def _apply_accessions(
    cases: list[EvaluationCase],
    accessions: dict[str, str],
) -> list[EvaluationCase]:
    resolved_cases = []

    for case in cases:
        accession_number = case.accession_number or accessions.get(_document_name(case))

        if accession_number is None:
            resolved_cases.append(case)
            continue

        references = [
            reference.model_copy(update={"accession_number": accession_number})
            for reference in case.reference_evidence
        ]
        resolved_cases.append(
            case.model_copy(
                update={
                    "accession_number": accession_number,
                    "reference_evidence": references,
                }
            )
        )

    return resolved_cases


def _as_date(value: Any) -> date | None:
    if value in (None, ""):
        return None

    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _candidate_years(case: EvaluationCase) -> list[int]:
    period = case.metadata.get("doc_period")

    if not isinstance(period, int):
        raise TypeError("FinanceBench document period is unavailable.")

    if case.form_type == "10-Q":
        return [period - 1, period, period + 1]

    return [period, period + 1]


def _narrow_candidates(case: EvaluationCase, filings: list[Any]) -> list[Any]:
    document_name = _document_name(case)
    dated_match = _DATED_DOCUMENT_PATTERN.search(document_name)

    if dated_match:
        expected_date = date.fromisoformat(dated_match.group(1))
        return [
            filing
            for filing in filings
            if _as_date(getattr(filing, "filing_date", None)) == expected_date
        ]

    period = case.metadata.get("doc_period")

    if case.form_type == "10-K" and isinstance(period, int):
        matching_reports = [
            filing
            for filing in filings
            if (report_date := _as_date(getattr(filing, "report_date", None)))
            and report_date.year == period
        ]

        if matching_reports:
            return matching_reports

    return filings


async def _match_candidate_by_evidence(
    filings: list[Any],
    cases: list[EvaluationCase],
) -> tuple[Any | None, str]:
    reference_texts = [
        normalize_text(str(reference.metadata.get("full_page_text") or reference.text))
        for case in cases
        for reference in case.reference_evidence
        if reference.metadata.get("full_page_text") or reference.text
    ]

    if not reference_texts:
        return None, "multiple SEC candidates and no reference evidence"

    scored_candidates: list[tuple[float, Any]] = []

    for filing in filings:
        try:
            content = await asyncio.to_thread(filing.markdown)
        except Exception as exc:  # noqa: BLE001 - candidate boundary.
            log.warning(
                "financebench_candidate_unreadable",
                accession_number=getattr(filing, "accession_number", None),
                error=str(exc),
            )
            continue

        normalized_content = normalize_text(content or "")
        score = max(
            (
                token_recall(reference_text, normalized_content)
                for reference_text in reference_texts
            ),
            default=0.0,
        )
        scored_candidates.append((score, filing))

    if not scored_candidates:
        return None, "SEC candidates could not be read"

    scored_candidates.sort(key=lambda item: item[0], reverse=True)
    best_score, best_filing = scored_candidates[0]
    second_score = scored_candidates[1][0] if len(scored_candidates) > 1 else 0.0

    if best_score < 0.55:
        return None, f"best evidence match is too weak ({best_score:.3f})"

    if best_score - second_score >= 0.01:
        return best_filing, f"evidence match {best_score:.3f}"

    representative = cases[0]
    quarter_match = _QUARTER_DOCUMENT_PATTERN.search(_document_name(representative))

    if quarter_match:
        expected_year = int(quarter_match.group(1))
        expected_quarter = int(quarter_match.group(2))
        expected_months = _QUARTER_END_MONTHS[expected_quarter]
        period_matches = [
            (score, candidate)
            for score, candidate in scored_candidates
            if (report_date := _as_date(getattr(candidate, "report_date", None)))
            and report_date.year == expected_year
            and report_date.month in expected_months
            and score >= 0.90
        ]

        if len(period_matches) == 1:
            score, filing = period_matches[0]
            return filing, (f"quarter metadata and evidence match {score:.3f}")

    if best_score - second_score < 0.05:
        candidate_summary = ", ".join(
            (
                f"{getattr(candidate, 'accession_number', '?')}"
                f"@{getattr(candidate, 'report_date', '?')}"
                f"={score:.3f}"
            )
            for score, candidate in scored_candidates[:3]
        )
        return None, (
            "SEC candidates are ambiguous "
            f"({best_score:.3f} vs {second_score:.3f}): "
            f"{candidate_summary}"
        )

    return best_filing, f"evidence match {best_score:.3f}"


async def _resolve_document_accession(
    edgar: EdgarClient,
    cases: list[EvaluationCase],
) -> tuple[str | None, str]:
    representative = cases[0]

    company_identifier = (
        representative.metadata.get("company_cik") or representative.ticker
    )

    if not company_identifier:
        return None, "company SEC identifier is unavailable"

    if not representative.form_type:
        return None, "document type is not an SEC filing"

    try:
        company = await edgar.get_company(company_identifier)
        filings = await asyncio.to_thread(
            company.get_filings,
            form=representative.form_type,
            year=_candidate_years(representative),
            amendments=False,
        )
        candidates = _narrow_candidates(
            representative,
            list(filings),
        )
    except Exception as exc:  # noqa: BLE001 - resolution reports per document.
        return None, f"SEC lookup failed: {exc}"

    if not candidates:
        return None, "no SEC filing candidate found"

    if len(candidates) == 1:
        filing = candidates[0]
        reason = "unique SEC metadata match"
    else:
        filing, reason = await _match_candidate_by_evidence(candidates, cases)

        if filing is None:
            return None, reason

    accession_number = getattr(filing, "accession_number", None)

    if not accession_number:
        return None, "matched SEC filing has no accession number"

    return str(accession_number), reason


async def resolve_financebench_accessions(
    cases: list[EvaluationCase],
    *,
    cache_path: Path | None = None,
    edgar: EdgarClient | None = None,
) -> AccessionResolutionResult:
    """Resolve missing accessions from SEC metadata and reference evidence."""
    client = edgar or EdgarClient()
    cache = _read_accession_cache(cache_path)
    documents: dict[str, list[EvaluationCase]] = {}

    for case in cases:
        documents.setdefault(_document_name(case), []).append(case)

    result = AccessionResolutionResult(cases=[])
    resolved_accessions: dict[str, str] = {}

    for document_name, document_cases in sorted(documents.items()):
        existing_accession = next(
            (case.accession_number for case in document_cases if case.accession_number),
            None,
        )

        if existing_accession:
            resolved_accessions[document_name] = existing_accession
            cache[document_name] = existing_accession
            continue

        if not document_cases[0].form_type:
            result.unsupported_documents[document_name] = (
                "document type is not supported by the SEC ingestion pipeline"
            )
            continue

        if cached_accession := cache.get(document_name):
            resolved_accessions[document_name] = cached_accession
            result.cached_documents[document_name] = cached_accession
            continue

        accession_number, reason = await _resolve_document_accession(
            client,
            document_cases,
        )

        if accession_number is None:
            result.unresolved_documents[document_name] = reason
            continue

        resolved_accessions[document_name] = accession_number
        result.resolved_documents[document_name] = accession_number
        cache[document_name] = accession_number
        _write_accession_cache(cache_path, cache)

    _write_accession_cache(cache_path, cache)
    result.cases = _apply_accessions(cases, resolved_accessions)
    return result


@dataclass
class CorpusPreparationResult:
    requested_filings: int

    processed_filings: int = 0
    skipped_filings: int = 0
    failed_filings: int = 0

    missing_accession_case_ids: list[str] = field(default_factory=list)

    failed_accessions: dict[str, str] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return self.failed_filings == 0 and not self.missing_accession_case_ids


def get_unique_accessions(
    cases: list[EvaluationCase],
) -> list[str]:
    """
    Return the unique SEC accession numbers required
    by the evaluation dataset.
    """
    accessions = {case.accession_number for case in cases if case.accession_number}

    return sorted(accessions)


async def prepare_financebench_corpus(
    pipeline: IngestionPipeline,
    cases: list[EvaluationCase],
) -> CorpusPreparationResult:
    """
    Ensure that every FinanceBench filing identified
    by an accession number is processed by the normal
    FinNexus ingestion pipeline.

    Each unique filing is ingested at most once during
    this corpus preparation run.
    """

    missing_accession_case_ids = [
        case.id for case in cases if not case.accession_number
    ]

    accessions = get_unique_accessions(cases)

    result = CorpusPreparationResult(
        requested_filings=len(accessions),
        missing_accession_case_ids=(missing_accession_case_ids),
    )

    for accession_number in accessions:
        try:
            ingestion = await pipeline.ingest_accession(accession_number)

            result.processed_filings += ingestion.filings_processed

            result.skipped_filings += ingestion.filings_skipped

        except Exception as exc:  # noqa: BLE001 - corpus boundary records failures.
            result.failed_filings += 1

            result.failed_accessions[accession_number] = str(exc)

    return result
