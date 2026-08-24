import asyncio
import re
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select

from rag_sec.config import get_settings
from rag_sec.database.manager import get_database_manager
from rag_sec.evaluation.corpus.financebench import (
    AccessionResolutionResult,
    resolve_financebench_accessions,
)
from rag_sec.evaluation.datasets.financebench import load_financebench
from rag_sec.evaluation.models import EvaluationCase
from rag_sec.models.filing import Filing
from rag_sec.models.processing_version import ProcessingVersion
from rag_sec.schemas.enums import ProcessingStatus

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FINANCEBENCH_ROOT = PROJECT_ROOT / "data" / "evaluation" / "financebench"
ACCESSION_PATTERN = re.compile(r"^\d{10}-\d{2}-\d{6}$")
SUPPORTED_FORMS = frozenset({"10-K", "10-Q", "8-K"})


@dataclass(frozen=True)
class LoadedSuite:
    cases: list[EvaluationCase]
    resolution: AccessionResolutionResult

    @property
    def excluded_cases(self) -> int:
        return len(self.resolution.cases) - len(self.cases)


class FinanceBenchSuite:
    """Own FinanceBench paths, resolution, validation, and corpus preflight."""

    def __init__(self, root: Path = DEFAULT_FINANCEBENCH_ROOT) -> None:
        self.root = root
        self.questions_path = root / "financebench_open_source.jsonl"
        self.documents_path = root / "financebench_document_information.jsonl"
        self.accession_cache_path = root / "accession_map.json"

    async def load(self) -> LoadedSuite:
        self._validate_input_files()
        source_cases = load_financebench(self.questions_path, self.documents_path)
        resolution = await resolve_financebench_accessions(
            source_cases,
            cache_path=self.accession_cache_path,
        )
        if resolution.unresolved_documents:
            details = "\n".join(
                f"- {name}: {reason}"
                for name, reason in resolution.unresolved_documents.items()
            )
            raise RuntimeError(f"Some SEC documents could not be resolved:\n{details}")

        cases = [case for case in resolution.cases if case.accession_number]
        self._validate_cases(cases)
        return LoadedSuite(cases=cases, resolution=resolution)

    async def validate_corpus(self, cases: list[EvaluationCase]) -> None:
        required = {case.accession_number for case in cases if case.accession_number}
        embedding = get_settings().embedding
        database = get_database_manager()
        try:
            await asyncio.wait_for(database.initialize(), timeout=10)
            async with database.session() as session:
                statement = (
                    select(Filing.accession_number)
                    .join(ProcessingVersion)
                    .where(
                        Filing.accession_number.in_(required),
                        ProcessingVersion.status == ProcessingStatus.ACTIVE.value,
                        ProcessingVersion.embedding_provider
                        == embedding.provider.value,
                        ProcessingVersion.embedding_model == embedding.model_name,
                        ProcessingVersion.embedding_dimension == embedding.dimension,
                    )
                    .distinct()
                )
                result = await asyncio.wait_for(session.execute(statement), timeout=10)
                available = set(result.scalars())
        except TimeoutError as exc:
            raise RuntimeError(
                "Database preflight timed out. Verify PostgreSQL and DATABASE_URL."
            ) from exc
        finally:
            await database.close()

        missing = sorted(required - available)
        if missing:
            details = "\n".join(f"- {accession}" for accession in missing)
            raise RuntimeError(
                "The FinanceBench corpus is incomplete for embedding profile "
                f"{embedding.provider.value}/{embedding.model_name}/"
                f"{embedding.dimension}:\n{details}\n"
                "Run `uv run scripts/prepare_financebench_corpus.py` first."
            )

    def artifact_path(self, filename: str) -> Path:
        return self.root / filename

    @staticmethod
    def embedding_metadata() -> dict[str, str | int]:
        embedding = get_settings().embedding
        return {
            "embedding_provider": embedding.provider.value,
            "embedding_model": embedding.model_name,
            "embedding_dimension": embedding.dimension,
        }

    def _validate_input_files(self) -> None:
        missing = [
            path
            for path in (self.questions_path, self.documents_path)
            if not path.is_file()
        ]
        if missing:
            details = "\n".join(f"- {path}" for path in missing)
            raise FileNotFoundError(f"FinanceBench input files are missing:\n{details}")

    @staticmethod
    def _validate_cases(cases: list[EvaluationCase]) -> None:
        if not cases:
            raise ValueError("No SEC-compatible FinanceBench cases were resolved.")

        errors: list[str] = []
        seen_ids: set[str] = set()
        for case in cases:
            if case.id in seen_ids:
                errors.append(f"{case.id}: duplicate case id")
            seen_ids.add(case.id)
            if not case.question.strip():
                errors.append(f"{case.id}: empty question")
            if not case.ticker:
                errors.append(f"{case.id}: missing ticker")
            if case.form_type not in SUPPORTED_FORMS:
                errors.append(f"{case.id}: unsupported form {case.form_type!r}")
            if not case.accession_number or not ACCESSION_PATTERN.fullmatch(
                case.accession_number
            ):
                errors.append(f"{case.id}: invalid accession {case.accession_number!r}")
            if any(
                evidence.accession_number != case.accession_number
                for evidence in case.reference_evidence
            ):
                errors.append(f"{case.id}: reference accession does not match case")

        if errors:
            preview = "\n".join(f"- {error}" for error in errors[:20])
            suffix = f"\n- ... and {len(errors) - 20} more" if len(errors) > 20 else ""
            raise ValueError(f"FinanceBench validation failed:\n{preview}{suffix}")
