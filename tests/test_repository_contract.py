from rag_sec.database.repositories import IngestionRepository, ProcessingRepository


def test_ingestion_repository_has_get_by_id():
    assert callable(IngestionRepository.get_by_id)


def test_processing_repository_has_get_by_id():
    assert callable(ProcessingRepository.get_by_id)
