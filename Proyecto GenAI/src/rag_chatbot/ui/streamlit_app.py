from rag_chatbot.config import get_settings
from rag_chatbot.ui.api_client import ApiClientError, ApiConnectionError, RagApiClient


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
        page_title="rag-chatbot",
        page_icon="",
        layout="wide",
    )

    st.title("Chatbot documental local")
    st.caption("Consulta local-first sobre documentos indexados. Sin LLM externo.")

    with st.sidebar:
        st.header("Conexión")
        api_base_url = st.text_input("API local", value=settings.api_base_url)
        client = RagApiClient(api_base_url)

        st.header("Consulta")
        top_k = st.number_input(
            "top_k",
            min_value=1,
            max_value=20,
            value=settings.top_k,
            step=1,
        )
        min_score = st.slider(
            "min_score",
            min_value=0.0,
            max_value=1.0,
            value=float(settings.min_retrieval_score),
            step=0.05,
        )
        show_sources = st.checkbox("Mostrar fuentes", value=True)
        show_chunks = st.checkbox("Mostrar chunks recuperados", value=False)

        st.header("Estado")
        if st.button("Verificar estado"):
            _show_health(st, client)
        if st.button("Consultar índice"):
            _show_index_info(st, client)

        with st.expander("Mantenimiento"):
            st.info(
                "Las operaciones pesadas como ingesta o reindexación se mantienen "
                "fuera de esta interfaz. Usa la CLI o la API local de forma consciente."
            )

    examples = [
        "¿Qué documentos hablan sobre compliance?",
        "¿Qué dice el protocolo sobre acoso entre estudiantes?",
        "¿Qué establece la política de IA?",
        "¿Qué normas regulan la convivencia en la Universidad de Navarra?",
        "¿Qué equipo ganó la última Champions League?",
    ]
    selected_example = st.selectbox("Ejemplos", [""] + examples)
    default_question = selected_example or ""
    question = st.text_area(
        "Pregunta",
        value=default_question,
        height=120,
        placeholder="Escribe una pregunta sobre los documentos indexados...",
    )

    if st.button("Preguntar", type="primary"):
        if not question.strip():
            st.warning("Escribe una pregunta antes de consultar.")
            return

        try:
            response = client.query(
                question=question.strip(),
                top_k=int(top_k),
                min_score=float(min_score),
                show_chunks=show_chunks,
            )
        except ApiConnectionError as exc:
            st.error(str(exc))
            st.info("Levanta la API en otra terminal con: rag-chatbot serve")
            return
        except ApiClientError as exc:
            st.error(str(exc))
            return

        _render_answer(st, response, show_sources=show_sources, show_chunks=show_chunks)


def _show_health(st, client: RagApiClient) -> None:
    try:
        payload = client.health()
    except ApiClientError as exc:
        st.error(str(exc))
        return
    st.success(f"API disponible: {payload.get('status', 'unknown')}")
    st.json(payload)


def _show_index_info(st, client: RagApiClient) -> None:
    try:
        payload = client.index_info()
    except ApiClientError as exc:
        st.error(str(exc))
        return

    st.write(f"Colección: `{payload.get('collection')}`")
    st.write(f"Vectores indexados: `{payload.get('vector_count')}`")
    st.write(f"Modelo embeddings: `{payload.get('embedding_model')}`")
    st.json(payload)


def _render_answer(st, response: dict, *, show_sources: bool, show_chunks: bool) -> None:
    rejection_reason = response.get("rejection_reason")
    if rejection_reason == "out_of_domain":
        st.warning("Pregunta fuera del alcance del corpus.")
        st.info(
            "Este chatbot solo responde sobre documentos normativos y políticas "
            "de la Universidad de Navarra cargados en el sistema."
        )
    elif response.get("warning"):
        st.warning(response["warning"])

    st.subheader("Respuesta")
    st.write(response.get("answer", ""))
    st.caption(f"Contexto suficiente: {response.get('has_sufficient_context')}")

    if not response.get("has_sufficient_context"):
        return

    if show_sources:
        st.subheader("Fuentes")
        sources = response.get("sources") or []
        if not sources:
            st.info("No hay fuentes disponibles.")
        for source in sources:
            st.markdown(
                f"- **{source.get('source_label')}** · "
                f"`{source.get('chunk_id')}`"
            )
            if source.get("snippet"):
                st.caption(source["snippet"])

    if show_chunks:
        st.subheader("Chunks recuperados")
        chunks = response.get("retrieved_chunks") or []
        if not chunks:
            st.info("No hay chunks para mostrar.")
        for chunk in chunks:
            with st.expander(
                f"{chunk.get('file_name')} · página {chunk.get('page_number')} · "
                f"score {chunk.get('score')}"
            ):
                st.write(f"chunk_id: `{chunk.get('chunk_id')}`")
                st.write(chunk.get("snippet") or "")
                if chunk.get("text"):
                    st.text(chunk["text"])


if __name__ == "__main__":
    main()
