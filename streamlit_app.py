from __future__ import annotations

import atexit
from typing import Any

import streamlit as st

from rag_sec.config import get_settings
from rag_sec.ui import RAGService

st.set_page_config(
    page_title="Finexus | SEC Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=False)
def get_rag_service() -> RAGService:
    """Keep async resources alive across Streamlit reruns."""
    service = RAGService()
    atexit.register(service.close)
    return service


def apply_styles() -> None:
    st.markdown(
        """
        <style>
            .block-container {
                max-width: 1120px;
                padding-top: 2.2rem;
                padding-bottom: 5rem;
            }
            .finexus-hero {
                padding: 1.6rem 1.8rem;
                margin-bottom: 1.5rem;
                border: 1px solid rgba(120, 130, 150, 0.24);
                border-radius: 18px;
                background:
                    radial-gradient(circle at top right,
                        rgba(56, 189, 248, 0.22), transparent 36%),
                    radial-gradient(circle at bottom left,
                        rgba(45, 212, 191, 0.16), transparent 42%),
                    linear-gradient(135deg,
                        rgba(20, 32, 48, 0.96), rgba(14, 22, 34, 0.96));
                box-shadow: 0 18px 60px rgba(2, 8, 23, 0.20);
            }
            .finexus-kicker {
                color: #63ddd1;
                font-size: 0.78rem;
                font-weight: 700;
                letter-spacing: 0.13em;
                text-transform: uppercase;
            }
            .finexus-hero h1 {
                margin: 0.35rem 0 0.45rem;
                color: #f6f8fb;
                font-size: 2.2rem;
            }
            .finexus-hero p {
                margin: 0;
                color: #bdc7d6;
                max-width: 720px;
            }
            [data-testid="stChatMessage"] {
                border: 1px solid rgba(120, 130, 150, 0.18);
                border-radius: 18px;
                padding: 0.55rem 0.85rem;
                box-shadow: 0 8px 28px rgba(2, 8, 23, 0.06);
            }
            [data-testid="stMetric"] {
                border: 1px solid rgba(120, 130, 150, 0.18);
                border-radius: 14px;
                padding: 0.7rem 0.8rem;
                background: rgba(125, 145, 175, 0.06);
            }
            .pipeline-row {
                display: grid;
                grid-template-columns: 110px 1fr 76px;
                gap: 0.75rem;
                align-items: center;
                margin: 0.55rem 0;
                font-size: 0.86rem;
            }
            .pipeline-track {
                height: 9px;
                overflow: hidden;
                border-radius: 999px;
                background: rgba(125, 145, 175, 0.16);
            }
            .pipeline-fill {
                height: 100%;
                min-width: 2px;
                border-radius: 999px;
                box-shadow: 0 0 16px currentColor;
            }
            .pipeline-time {
                color: #8fa0b7;
                text-align: right;
                font-variant-numeric: tabular-nums;
            }
            .evidence-title {
                margin-top: 0.2rem;
                color: #2dd4bf;
                font-size: 0.76rem;
                font-weight: 750;
                letter-spacing: 0.1em;
                text-transform: uppercase;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def initialize_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "session_metrics" not in st.session_state:
        st.session_state.session_metrics = {
            "queries": 0,
            "total_latency_ms": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
        }


def link_citations(content: str, sources: list[dict[str, Any]]) -> str:
    """Turn citation markers into direct links to their cited passages."""
    linked_content = content

    for source in sources:
        source_id = source.get("source_id")
        target = source.get("deep_link") or source.get("source_url")

        if source_id and target:
            linked_content = linked_content.replace(
                f"[{source_id}]",
                f"[[{source_id}]]({target})",
            )

    return linked_content


def source_location(source: dict[str, Any]) -> str:
    location = [
        source.get("part"),
        source.get("item"),
        source.get("section"),
    ]
    values = [str(value) for value in location if value]

    if source.get("page") is not None:
        values.append(f"page {source['page']}")

    if source.get("chunk_index") is not None:
        values.append(f"chunk {source['chunk_index']}")

    return " · ".join(dict.fromkeys(values)) or "Section SEC"


def render_sources(sources: list[dict[str, Any]]) -> None:
    if not sources:
        return

    st.markdown(
        '<div class="evidence-title">Evidence map</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Les citations de la réponse et le bouton « Passage ciblé » ouvrent "
        "directement le texte utilisé dans le filing SEC."
    )
    with st.expander(
        f"{len(sources)} passage(s) cité(s) — ouvrir les preuves",
        expanded=True,
    ):
        for index, source in enumerate(sources):
            company = source.get("company_name") or source.get("ticker") or "SEC"
            form_type = source.get("form_type") or "Filing"
            source_id = source.get("source_id", f"S{index + 1}")
            source_url = source.get("source_url")
            deep_link = source.get("deep_link")

            with st.container(border=True):
                title_column, action_column = st.columns([4, 1.4])

                with title_column:
                    st.markdown(f"**{source_id} · {company} · {form_type}**")
                    st.caption(
                        f"{source_location(source)}  ·  "
                        f"{source.get('filing_date') or 'date inconnue'}"
                    )

                with action_column:
                    if deep_link:
                        st.link_button(
                            "↗ Passage ciblé",
                            deep_link,
                            use_container_width=True,
                        )
                    elif source_url:
                        st.link_button(
                            "↗ Ouvrir le filing",
                            source_url,
                            use_container_width=True,
                        )

                excerpt = source.get("excerpt")

                if excerpt:
                    st.markdown("**Extrait utilisé pour la réponse**")
                    st.markdown(f"> {excerpt}")

                metadata = [
                    source.get("accession_number"),
                    (
                        f"{source['token_count']} tokens"
                        if source.get("token_count") is not None
                        else None
                    ),
                ]
                st.caption(" · ".join(str(value) for value in metadata if value))

                if source_url and deep_link:
                    st.markdown(f"[Voir le filing complet]({source_url})")


def format_duration(milliseconds: float) -> str:
    if milliseconds < 1000:
        return f"{milliseconds:.0f} ms"

    return f"{milliseconds / 1000:.2f} s"


def render_pipeline(metrics: dict[str, Any]) -> None:
    total = float(metrics.get("total_latency_ms", 0))

    if total <= 0:
        return

    phases = [
        ("Embedding", float(metrics.get("embedding_latency_ms", 0)), "#a78bfa"),
        ("Retrieval", float(metrics.get("retrieval_latency_ms", 0)), "#2dd4bf"),
        ("Génération", float(metrics.get("generation_latency_ms", 0)), "#38bdf8"),
    ]
    measured = sum(duration for _label, duration, _color in phases)
    phases.append(("Orchestration", max(0.0, total - measured), "#64748b"))

    rows = []

    for label, duration, color in phases:
        width = min(100.0, max(0.2, duration / total * 100))
        rows.append(
            f"""
            <div class="pipeline-row">
                <span>{label}</span>
                <div class="pipeline-track">
                    <div class="pipeline-fill"
                         style="width:{width:.2f}%;background:{color};color:{color}">
                    </div>
                </div>
                <span class="pipeline-time">{format_duration(duration)}</span>
            </div>
            """
        )

    st.markdown("".join(rows), unsafe_allow_html=True)


def render_metrics(usage: dict[str, Any], metrics: dict[str, Any]) -> None:
    if not metrics:
        return

    estimated = bool(usage.get("estimated"))
    token_prefix = "≈ " if estimated else ""
    columns = st.columns(6)
    columns[0].metric(
        "Latence",
        format_duration(float(metrics.get("total_latency_ms", 0))),
    )
    columns[1].metric(
        "Débit LLM",
        f"{float(metrics.get('generation_throughput_tokens_per_second', 0)):.1f} tok/s",
    )
    columns[2].metric(
        "Tokens entrée",
        f"{token_prefix}{int(usage.get('input_tokens', 0)):,}",
    )
    columns[3].metric(
        "Tokens sortie",
        f"{token_prefix}{int(usage.get('output_tokens', 0)):,}",
    )
    columns[4].metric(
        "Documents",
        int(metrics.get("retrieved_documents", 0)),
    )
    columns[5].metric(
        "Sources citées",
        int(metrics.get("cited_sources", 0)),
    )

    with st.expander("Profil de performance", expanded=False):
        render_pipeline(metrics)
        retrieved_documents = int(metrics.get("retrieved_documents", 0))
        cited_sources = int(metrics.get("cited_sources", 0))
        evidence_coverage = (
            min(1.0, cited_sources / retrieved_documents)
            if retrieved_documents
            else 0.0
        )
        st.progress(
            evidence_coverage,
            text=(
                "Couverture des preuves · "
                f"{cited_sources}/{retrieved_documents} passages retenus"
            ),
        )
        st.caption(
            "Débit retrieval : "
            f"{float(metrics.get('retrieval_throughput_documents_per_second', 0)):.1f} docs/s"
            + (" · Tokens estimés par fallback" if estimated else " · Usage provider exact")
        )


def render_assistant_content(message: dict[str, Any]) -> None:
    sources = message.get("sources", [])
    st.markdown(link_citations(message["content"], sources))
    render_metrics(
        message.get("usage", {}),
        message.get("metrics", {}),
    )
    render_sources(sources)


def render_message(message: dict[str, Any]) -> None:
    role = message["role"]
    avatar = "🧑‍💼" if role == "user" else "📊"

    with st.chat_message(role, avatar=avatar):
        if role == "assistant":
            render_assistant_content(message)
        else:
            st.markdown(message["content"])


def render_sidebar() -> tuple[str, str, str]:
    settings = get_settings()

    with st.sidebar:
        page = st.radio(
            "Navigation",
            options=("Assistant", "Filings"),
            index=0,
        )
        st.divider()
        st.header("Paramètres de recherche")
        ticker = st.text_input(
            "Ticker",
            value="AAPL",
            max_chars=12,
            help="Ticker de la société tel qu’enregistré dans les filings.",
        ).strip().upper()
        form_type = st.selectbox(
            "Type de filing",
            options=("10-K", "10-Q", "8-K", "20-F", "40-F"),
            index=0,
        )

        st.divider()
        st.subheader("Configuration active")
        st.caption("Embedding")
        st.code(
            f"{settings.embedding.provider.value}\n"
            f"{settings.embedding.model_name}\n"
            f"{settings.embedding.dimension} dimensions",
            language=None,
        )
        st.caption("LLM")
        st.code(
            f"{settings.llm.provider.value}\n{settings.llm.model_name}",
            language=None,
        )

        st.divider()
        session_metrics = st.session_state.session_metrics

        if session_metrics["queries"]:
            st.subheader("Session live")
            average_latency = (
                session_metrics["total_latency_ms"]
                / session_metrics["queries"]
            )
            session_columns = st.columns(2)
            session_columns[0].metric(
                "Requêtes",
                session_metrics["queries"],
            )
            session_columns[1].metric(
                "Latence moy.",
                format_duration(average_latency),
            )
            st.caption(
                f"{session_metrics['input_tokens']:,} tokens entrants · "
                f"{session_metrics['output_tokens']:,} tokens sortants"
            )

        if st.button("Effacer la conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.session_metrics = {
                "queries": 0,
                "total_latency_ms": 0.0,
                "input_tokens": 0,
                "output_tokens": 0,
            }
            st.rerun()

        st.caption(
            "Chaque question est traitée indépendamment à partir des "
            "extraits SEC actifs."
        )

    return page, ticker, form_type


def warmup_service(service: RAGService) -> None:
    if service.is_ready:
        return

    status = st.status(
        "Initialisation du moteur RAG…",
        expanded=True,
        state="running",
    )

    try:
        status.write("Connexion à PostgreSQL et préparation de pgvector")
        status.write("Chargement du modèle d’embedding")
        status.write("Initialisation du retriever et du LLM")
        service.warmup()
    except Exception as exc:  # noqa: BLE001 - UI boundary reports failures.
        status.update(
            label="Échec de l’initialisation",
            state="error",
            expanded=True,
        )
        st.error(f"Le moteur RAG n’a pas pu démarrer : {exc}")
        st.stop()
    else:
        status.update(
            label="Moteur RAG prêt",
            state="complete",
            expanded=False,
        )


def render_assistant(service: RAGService, *, ticker: str, form_type: str) -> None:
    for message in st.session_state.messages:
        render_message(message)

    question = st.chat_input(
        "Posez une question sur les résultats, risques ou activités…",
        max_chars=2000,
    )

    if not question:
        return

    if not ticker:
        st.warning("Renseignez un ticker avant de poser votre question.")
        return

    user_message = {
        "role": "user",
        "content": question,
        "sources": [],
    }
    st.session_state.messages.append(user_message)
    render_message(user_message)

    with st.chat_message("assistant", avatar="📊"):
        try:
            with st.spinner(
                f"Analyse du {form_type} de {ticker}…",
            ):
                result = service.answer(
                    question,
                    ticker=ticker,
                    form_type=form_type,
                )
        except Exception as exc:  # noqa: BLE001 - UI boundary reports failures.
            st.error(f"La requête a échoué : {exc}")
            return

        sources = [
            source.model_dump(mode="json", exclude_none=True)
            for source in result.sources
        ]
        usage = result.usage.model_dump(mode="json")
        metrics = result.metrics.model_dump(mode="json")
        assistant_message = {
            "role": "assistant",
            "content": result.answer,
            "sources": sources,
            "usage": usage,
            "metrics": metrics,
        }
        render_assistant_content(assistant_message)

    st.session_state.messages.append(assistant_message)
    session_metrics = st.session_state.session_metrics
    session_metrics["queries"] += 1
    session_metrics["total_latency_ms"] += result.metrics.total_latency_ms
    session_metrics["input_tokens"] += result.usage.input_tokens
    session_metrics["output_tokens"] += result.usage.output_tokens


def filing_records(service: RAGService) -> list[dict[str, Any]]:
    filings = service.list_filings()

    return [
        {
            "Ticker": filing.ticker or str(filing.cik),
            "Société": filing.company_name,
            "Formulaire": filing.form_type,
            "Date": filing.filing_date,
            "Période": filing.period_of_report,
            "Chunks": filing.chunk_count,
            "Accession": filing.accession_number,
            "SEC": filing.source_url,
        }
        for filing in filings
    ]


def render_filings(
    service: RAGService,
    *,
    default_ticker: str,
    default_form_type: str,
) -> None:
    st.subheader("Ingestion SEC")
    st.caption(
        "Télécharge et indexe le dernier filing correspondant avec le "
        "profil d’embedding actif."
    )

    with st.form("ingestion-form", clear_on_submit=False):
        identifier_column, form_column = st.columns([2, 1])

        with identifier_column:
            identifier = st.text_input(
                "Ticker ou CIK",
                value=default_ticker,
                max_chars=20,
                key="ingestion-identifier",
            )

        with form_column:
            ingestion_form_type = st.selectbox(
                "Formulaire SEC",
                options=("10-K", "10-Q", "8-K", "20-F", "40-F"),
                index=("10-K", "10-Q", "8-K", "20-F", "40-F").index(
                    default_form_type
                ),
                key="ingestion-form-type",
            )

        ingest_submitted = st.form_submit_button(
            "Ingérer le dernier filing",
            type="primary",
            use_container_width=True,
        )

    if ingest_submitted:
        if not identifier.strip():
            st.warning("Renseignez un ticker ou un CIK.")
        else:
            status = st.status(
                f"Ingestion de {identifier.upper()} ({ingestion_form_type})…",
                expanded=True,
                state="running",
            )

            try:
                status.write("Découverte et téléchargement depuis SEC EDGAR")
                status.write("Extraction, découpage et création des embeddings")
                status.write("Validation et activation des chunks")
                result = service.ingest(
                    identifier,
                    form_type=ingestion_form_type,
                )
            except Exception as exc:  # noqa: BLE001 - UI boundary.
                status.update(
                    label="Échec de l’ingestion",
                    state="error",
                    expanded=True,
                )
                st.error(str(exc))
            else:
                status.update(
                    label=f"Ingestion terminée — {result.status.value}",
                    state="complete",
                    expanded=False,
                )

                if result.filings_processed:
                    st.success(
                        f"{result.filings_processed} filing indexé avec succès."
                    )
                elif result.filings_skipped:
                    st.info("Ce filing était déjà indexé avec ce profil.")

    st.divider()
    st.subheader("Filings disponibles")

    try:
        records = filing_records(service)
    except Exception as exc:  # noqa: BLE001 - UI boundary reports failures.
        st.error(f"Impossible de charger les filings : {exc}")
        return

    st.caption(
        f"{len(records)} filing(s) actif(s) pour le provider d’embedding courant."
    )

    if not records:
        st.info(
            "Aucun filing n’est encore disponible pour ce profil. "
            "Lancez une ingestion ci-dessus."
        )
        return

    st.dataframe(
        records,
        hide_index=True,
        use_container_width=True,
        column_config={
            "SEC": st.column_config.LinkColumn(
                "Source SEC",
                display_text="Ouvrir",
            ),
            "Chunks": st.column_config.NumberColumn(format="%d"),
        },
    )


def main() -> None:
    apply_styles()
    initialize_state()

    st.markdown(
        """
        <section class="finexus-hero">
            <div class="finexus-kicker">SEC filing intelligence</div>
            <h1>Finexus</h1>
            <p>
                Interrogez les rapports financiers avec des réponses sourcées,
                fondées sur les filings présents dans votre base documentaire.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    page, ticker, form_type = render_sidebar()
    service = get_rag_service()
    warmup_service(service)

    if page == "Assistant":
        render_assistant(
            service,
            ticker=ticker,
            form_type=form_type,
        )
    else:
        render_filings(
            service,
            default_ticker=ticker,
            default_form_type=form_type,
        )


if __name__ == "__main__":
    main()
