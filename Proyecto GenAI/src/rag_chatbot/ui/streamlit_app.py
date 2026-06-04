from pathlib import Path

from rag_chatbot.config import get_settings
from rag_chatbot.ui.api_client import (
    ApiClientError,
    ApiConnectionError,
    ApiTimeoutError,
    RagApiClient,
)


ASSETS_DIR = Path(__file__).parent / "assets"
LOGO_PATH = ASSETS_DIR / "unav_logo.png"

ACCENT_RED = "#b00020"
TEXT_DARK = "#1f1f1f"
TEXT_MUTED = "#666666"
BORDER = "#e5e5e5"
SURFACE = "#f7f7f7"

SUGGESTED_QUESTIONS = [
    "¿Qué dice el protocolo sobre acoso entre estudiantes?",
    "¿Qué establece la política de IA?",
    "¿Qué normas regulan la convivencia en la Universidad de Navarra?",
    "¿Qué información contiene el documento sobre compliance penal?",
    "¿Qué dice la normativa sobre reconocimiento de créditos de grado?",
]


def main() -> None:
    try:
        import streamlit as st
    except ImportError as exc:
        raise RuntimeError(
            "streamlit is not installed. Install/update dependencies with: "
            "python -m pip install -e ."
        ) from exc

    settings = get_settings()
    st.set_page_config(
        page_title="Campus Bot UNAV",
        page_icon="",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_styles(st)
    _init_state(st)

    if False:
        st.markdown('<div class="sidebar-brand">Universidad de Navarra</div>', unsafe_allow_html=True)
        st.caption("Proyecto GenAI")

        st.divider()
        st.subheader("Estado")
        api_base_url = st.text_input("API local", value=settings.api_base_url)
        client = RagApiClient(
            api_base_url,
            timeout=float(settings.api_request_timeout_seconds),
        )

        status_col, index_col = st.columns(2)
        with status_col:
            if st.button("API", use_container_width=True):
                _show_health(st, client)
        with index_col:
            if st.button("Índice", use_container_width=True):
                _show_index_info(st, client)

        st.divider()
        st.subheader("Consulta")
        mode_label = st.radio(
            "Modo de respuesta",
            ["RAG extractivo local", "RAG generativo con Gemini"],
            index=0 if settings.llm_mode_default != "llm" else 1,
            help="Gemini solo se usa si está permitido en .env y hay contexto suficiente.",
        )
        mode = "llm" if mode_label == "RAG generativo con Gemini" else "extractive"
        top_k = st.slider("top_k", min_value=1, max_value=10, value=settings.top_k)
        min_score = st.slider(
            "min_score",
            min_value=0.0,
            max_value=1.0,
            value=float(settings.min_retrieval_score),
            step=0.01,
        )
        show_sources = st.toggle("Mostrar fuentes", value=True)
        show_chunks = st.toggle("Mostrar chunks recuperados", value=False)

        st.divider()
        st.markdown(
            """
            <div class="privacy-note">
              <strong>Privacidad</strong><br>
              Modo local: los documentos no salen del equipo.<br>
              Modo Gemini: solo se envían la pregunta y los fragmentos recuperados.
            </div>
            """,
            unsafe_allow_html=True,
        )

    _render_header(st)
    _render_suggestions(st)
    client, mode, top_k, min_score, show_sources, show_chunks = _render_controls(st, settings)

    question = st.text_area(
        "Pregunta",
        key="question",
        height=120,
        placeholder="Escribe una pregunta sobre los documentos indexados...",
        label_visibility="collapsed",
    )

    action_col, hint_col = st.columns([1, 3])
    with action_col:
        submitted = st.button("Consultar", type="primary", use_container_width=True)
    with hint_col:
        st.caption("La respuesta se basa en documentos normativos y políticas cargadas en el sistema.")

    if submitted:
        _run_query(
            st,
            client,
            question=question,
            mode=mode,
            top_k=int(top_k),
            min_score=float(min_score),
            show_chunks=show_chunks,
        )

    if st.session_state.last_response:
        _render_answer(
            st,
            st.session_state.last_response,
            show_sources=show_sources,
            show_chunks=show_chunks,
        )


def _inject_styles(st) -> None:
    st.markdown(
        f"""
        <style>
        div[data-testid="stDecoration"],
        #MainMenu,
        footer {{
            display: none;
            visibility: hidden;
            height: 0;
        }}
        header[data-testid="stHeader"] {{
            background: transparent;
        }}
        [data-testid="stAppViewContainer"] .main .block-container,
        section.main > div.block-container,
        .main .block-container {{
            max-width: 1080px;
            padding-top: 0.5rem !important;
            margin-top: 0 !important;
        }}
        [data-testid="stVerticalBlock"] {{
            gap: 0.5rem;
        }}
        h1, h2, h3, p, label, span {{
            color: {TEXT_DARK};
        }}
        .rag-eyebrow {{
            color: {ACCENT_RED};
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }}
        .rag-title {{
            font-size: 2.15rem;
            font-weight: 700;
            line-height: 1.15;
            margin-bottom: 0.35rem;
        }}
        .rag-subtitle {{
            color: {TEXT_MUTED};
            font-size: 1rem;
            max-width: 760px;
            margin-bottom: 0.75rem;
        }}
        .header-rule {{
            border-bottom: 1px solid {BORDER};
            margin-top: 0.35rem;
            margin-bottom: 0.85rem;
        }}
        .answer-card, .source-card, .status-card {{
            border: 1px solid {BORDER};
            border-radius: 8px;
            padding: 1rem;
            background: #ffffff;
        }}
        .answer-card {{
            border-left: 4px solid {ACCENT_RED};
            margin-top: 1rem;
        }}
        .source-card {{
            background: {SURFACE};
            margin-bottom: 0.75rem;
        }}
        .metadata-line {{
            color: {TEXT_MUTED};
            font-size: 0.84rem;
        }}
        .sidebar-brand {{
            color: {ACCENT_RED};
            font-weight: 700;
            font-size: 1.05rem;
        }}
        .privacy-note {{
            border: 1px solid {BORDER};
            border-radius: 8px;
            background: {SURFACE};
            padding: 0.85rem;
            margin-bottom: 1rem;
            font-size: 0.86rem;
            color: {TEXT_MUTED};
        }}
        .stButton > button[kind="primary"] {{
            background-color: {ACCENT_RED};
            border-color: {ACCENT_RED};
            color: #ffffff;
        }}
        .stButton > button[kind="primary"] * {{
            color: #ffffff;
        }}
        .stButton > button[kind="primary"]:hover,
        .stButton > button[kind="primary"]:focus {{
            background-color: #8f001a;
            border-color: #8f001a;
            color: #ffffff;
        }}
        .stButton > button[kind="primary"]:hover *,
        .stButton > button[kind="primary"]:focus * {{
            color: #ffffff;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _init_state(st) -> None:
    st.session_state.setdefault("question", "")
    st.session_state.setdefault("last_response", None)


def _render_header(st) -> None:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=150)

    st.markdown(
        """
        <div class="rag-eyebrow">Universidad de Navarra · Proyecto GenAI</div>
        <div class="rag-title">Campus Bot UNAV</div>
        <div class="rag-subtitle">
          Consulta documentos normativos y políticas de la Universidad de Navarra
          mediante recuperación semántica y RAG generativo opcional.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="header-rule"></div>', unsafe_allow_html=True)


def _render_suggestions(st) -> None:
    st.markdown("Preguntas sugeridas")
    columns = st.columns(2)
    for index, question in enumerate(SUGGESTED_QUESTIONS):
        with columns[index % 2]:
            if st.button(question, key=f"suggestion_{index}", use_container_width=True):
                st.session_state.question = question


def _render_controls(st, settings):
    with st.expander("Configuración de consulta", expanded=True):
        status_col, index_col, api_col = st.columns([1, 1, 3])
        api_base_url = api_col.text_input(
            "API local",
            value=settings.api_base_url,
            key="main_api_base_url",
        )
        client = RagApiClient(
            api_base_url,
            timeout=float(settings.api_request_timeout_seconds),
        )

        with status_col:
            st.caption("Estado de API")
            if st.button("Verificar API", key="main_health", use_container_width=True):
                _show_health(st, client)
        with index_col:
            st.caption("Índice vectorial")
            if st.button("Verificar índice", key="main_index", use_container_width=True):
                _show_index_info(st, client)

        mode_col, topk_col, score_col = st.columns([2, 1, 1])
        with mode_col:
            mode_label = st.radio(
                "Modo de respuesta",
                ["RAG extractivo local", "RAG generativo con Gemini"],
                index=0 if settings.llm_mode_default != "llm" else 1,
                help="Gemini solo se usa si está permitido en .env y hay contexto suficiente.",
                horizontal=True,
                key="main_response_mode",
            )
        with topk_col:
            top_k = st.slider(
                "top_k",
                min_value=1,
                max_value=10,
                value=settings.top_k,
                key="main_top_k",
            )
        with score_col:
            min_score = st.slider(
                "min_score",
                min_value=0.0,
                max_value=1.0,
                value=float(settings.min_retrieval_score),
                step=0.01,
                key="main_min_score",
            )

        source_col, chunk_col = st.columns(2)
        with source_col:
            show_sources = st.toggle("Mostrar fuentes", value=True, key="main_show_sources")
        with chunk_col:
            show_chunks = st.toggle(
                "Mostrar chunks recuperados",
                value=False,
                key="main_show_chunks",
            )

        st.markdown(
            """
            <div class="privacy-note">
              <strong>Privacidad</strong><br>
              Modo local: los documentos no salen del equipo.<br>
              Modo Gemini: solo se envían la pregunta y los fragmentos recuperados.
            </div>
            """,
            unsafe_allow_html=True,
        )

    mode = "llm" if mode_label == "RAG generativo con Gemini" else "extractive"
    return client, mode, int(top_k), float(min_score), show_sources, show_chunks


def _run_query(
    st,
    client: RagApiClient,
    *,
    question: str,
    mode: str,
    top_k: int,
    min_score: float,
    show_chunks: bool,
) -> None:
    if not question.strip():
        st.warning("Escribe una pregunta antes de consultar.")
        return

    with st.spinner(_spinner_message(mode)):
        try:
            st.session_state.last_response = client.query(
                question=question.strip(),
                mode=mode,
                top_k=top_k,
                min_score=min_score,
                show_chunks=show_chunks,
            )
        except ApiConnectionError as exc:
            st.session_state.last_response = None
            st.error("No se pudo conectar con la API local.")
            st.info("Levanta la API en otra terminal con: rag-chatbot serve")
            st.caption(str(exc))
        except ApiTimeoutError as exc:
            st.session_state.last_response = None
            st.warning(str(exc))
        except ApiClientError as exc:
            st.session_state.last_response = None
            _render_api_error(st, str(exc))


def _spinner_message(mode: str) -> str:
    if mode == "llm":
        return "Consultando documentos y generando respuesta con Gemini..."
    return "Buscando en los documentos indexados..."


def _show_health(st, client: RagApiClient) -> None:
    try:
        payload = client.health()
    except ApiClientError as exc:
        st.error("API no disponible.")
        st.caption(str(exc))
        return
    st.success(f"API disponible: {payload.get('status', 'unknown')}")


def _show_index_info(st, client: RagApiClient) -> None:
    try:
        payload = client.index_info()
    except ApiClientError as exc:
        st.error("No se pudo consultar el índice.")
        st.caption(str(exc))
        return

    vector_count = payload.get("vector_count")
    if vector_count:
        st.success(f"Índice disponible: {vector_count} vectores")
    else:
        st.warning("Índice vacío o no disponible.")
    st.caption(f"Embeddings: {payload.get('embedding_model', 'desconocido')}")


def _render_answer(st, response: dict, *, show_sources: bool, show_chunks: bool) -> None:
    rejection_reason = response.get("rejection_reason")
    if rejection_reason == "out_of_domain":
        st.info("Pregunta fuera del alcance del corpus.")
        st.caption(
            "Este chatbot está limitado a documentos normativos y políticas de "
            "la Universidad de Navarra cargados en el sistema."
        )
    elif response.get("warning"):
        st.warning(response["warning"])

    if response.get("llm_warning"):
        st.warning(response["llm_warning"])

    mode_label = _mode_label(response)
    st.markdown(
        f"""
        <div class="answer-card">
          <div class="metadata-line">{mode_label}</div>
          <div>{_html_escape(response.get("answer", ""))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not response.get("has_sufficient_context"):
        return

    if show_sources:
        _render_sources(st, response.get("sources") or [])

    if show_chunks:
        _render_chunks(st, response.get("retrieved_chunks") or [])


def _render_sources(st, sources: list[dict]) -> None:
    st.subheader("Fuentes utilizadas")
    if not sources:
        st.info("No hay fuentes disponibles.")
        return

    for index, source in enumerate(sources, start=1):
        label = source.get("source_label") or source.get("file_name") or "Fuente"
        chunk_id = source.get("chunk_id", "")
        snippet = source.get("snippet") or ""
        with st.expander(f"{index}. {label}", expanded=index <= 2):
            st.markdown(f"**Archivo:** {source.get('file_name', 'No disponible')}")
            st.markdown(f"**Página:** {source.get('page_number', 'No disponible')}")
            st.markdown(f"**chunk_id:** `{chunk_id}`")
            if source.get("score") is not None:
                st.markdown(f"**Score:** {source['score']}")
            if snippet:
                st.caption(snippet)


def _render_chunks(st, chunks: list[dict]) -> None:
    st.subheader("Chunks recuperados")
    if not chunks:
        st.info("No hay chunks para mostrar.")
        return

    for index, chunk in enumerate(chunks, start=1):
        title = (
            f"{index}. {chunk.get('file_name')} · página {chunk.get('page_number')} · "
            f"score {chunk.get('score')}"
        )
        with st.expander(title):
            st.write(f"chunk_id: `{chunk.get('chunk_id')}`")
            st.write(chunk.get("snippet") or "")
            if chunk.get("text"):
                st.text(chunk["text"])


def _render_api_error(st, message: str) -> None:
    lower = message.casefold()
    if "empty" in lower or "index" in lower:
        st.warning("El índice vectorial no está listo.")
        st.info("Ejecuta: rag-chatbot build-index --reset")
    elif "allow_external_llm" in lower or "gemini" in lower:
        st.warning("Gemini no está disponible para esta consulta.")
        st.info("Revisa `.env` o usa el modo RAG extractivo local.")
    else:
        st.error("No se pudo completar la consulta.")
    st.caption(message)


def _mode_label(response: dict) -> str:
    used_mode = response.get("mode", "extractive")
    llm_used = response.get("llm_used", False)
    provider = response.get("llm_provider")
    model = response.get("llm_model")

    if used_mode == "llm" or llm_used:
        parts = ["Modo usado: Gemini generativo"]
        if provider:
            parts.append(f"Proveedor: {provider}")
        if model:
            parts.append(f"Modelo: {model}")
        return " · ".join(parts)

    return "Modo usado: RAG extractivo local"


def _html_escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br>")
    )


if __name__ == "__main__":
    main()
