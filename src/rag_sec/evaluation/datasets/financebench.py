import json
import re
from pathlib import Path
from typing import Any

from rag_sec.evaluation.models import (
    EvaluationCase,
    ReferenceEvidence,
)

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
                "doc_name": doc_name,
                "doc_period": document.get("doc_period"),
                "doc_type": document.get("doc_type"),
                "doc_link": doc_link,
                "company_cik": FINANCEBENCH_COMPANY_CIKS.get(company_key),
                "justification": row.get("justification"),
                "dataset_subset_label": row.get("dataset_subset_label"),
            },
        )

        cases.append(case)

    return cases
