from enum import Enum


class SourceAuthority(str, Enum):
    SEC_OFFICIAL = "sec_official"


class FilingStatus(str, Enum):
    DISCOVERED = "discovered"
    FETCHED = "fetched"
    FAILED = "failed"


class ProcessingStatus(str, Enum):
    BUILDING = "building"
    ACTIVE = "active"
    FAILED = "failed"
    SUPERSEDED = "superseded"


class IngestionStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    COMPLETED_WITH_ERRORS = "completed_with_errors"
    FAILED = "failed"


class IngestionStage(str, Enum):
    DISCOVER = "discover"
    FETCH = "fetch"
    EXTRACT = "extract"
    NORMALIZE = "normalize"
    CHUNK = "chunk"
    EMBED = "embed"
    PERSIST = "persist"
    VALIDATE = "validate"
    ACTIVATE = "activate"


class DistanceMetric(str, Enum):
    COSINE = "cosine"
    L2 = "l2"
    INNER_PRODUCT = "inner_product"
