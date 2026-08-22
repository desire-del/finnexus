from rag_sec.config import get_settings
from rag_sec.logging import configure_logging
from rag_sec.logging import get_logger
from rag_sec.observability import configure_observability

def main():
    # Configure logging
    settings = get_settings()
    configure_logging(
        log_level=settings.log_level,
        json_format=settings.json_logging,
        log_file="logs/app.log",  # You can specify a log file path here if needed
        use_stderr=True  # Use stderr for logging output
    )

    log = get_logger(__name__)
    log.info("Application started", environment=settings.environment)


    print(settings.database_url)
    print(settings.observability.provider)
    print(settings.observability.config)

    configure_observability()

if __name__ == "__main__":
    main()