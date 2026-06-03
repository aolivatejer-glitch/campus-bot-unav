from rag_chatbot.config import AppSettings, get_settings
from rag_chatbot.rag.pipeline import LocalRagPipeline
from rag_chatbot.retrieval.retriever import SemanticRetriever


def get_api_settings() -> AppSettings:
    return get_settings()


def get_semantic_retriever() -> SemanticRetriever:
    settings = get_api_settings()
    return SemanticRetriever(settings)


def get_rag_pipeline() -> LocalRagPipeline:
    settings = get_api_settings()
    return LocalRagPipeline(settings)
