import ast
import importlib
import os
from dataclasses import dataclass
from enum import StrEnum
from importlib.util import find_spec
from pathlib import Path
from typing import Mapping

from rag_chatbot.config import PROJECT_ROOT, AppSettings, get_settings
from rag_chatbot.evaluation.reporting import JSON_REPORT_NAME
from rag_chatbot.ingestion.discovery import discover_documents


class DiagnosticStatus(StrEnum):
    OK = "OK"
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass(frozen=True)
class DiagnosticCheck:
    status: DiagnosticStatus
    label: str
    message: str


@dataclass(frozen=True)
class ChromaDiagnosticInfo:
    available: bool
    collection_exists: bool = False
    vector_count: int | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class DoctorReport:
    checks: list[DiagnosticCheck]

    @property
    def ok_count(self) -> int:
        return sum(1 for check in self.checks if check.status == DiagnosticStatus.OK)

    @property
    def warning_count(self) -> int:
        return sum(1 for check in self.checks if check.status == DiagnosticStatus.WARNING)

    @property
    def error_count(self) -> int:
        return sum(1 for check in self.checks if check.status == DiagnosticStatus.ERROR)

    @property
    def has_errors(self) -> bool:
        return self.error_count > 0


EXTERNAL_API_ENV_VARS = (
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "ANTHROPIC_API_KEY",
)
EXTERNAL_IMPORT_PREFIXES = (
    "openai",
    "langchain",
    "google.generativeai",
    "google.genai",
)
MAX_LOG_LINE_LENGTH = 2000


def run_doctor(
    settings: AppSettings | None = None,
    *,
    chroma_inspector=None,
) -> DoctorReport:
    """Run local MVP diagnostics without rebuilding data or embeddings."""
    settings = settings or get_settings()
    checks: list[DiagnosticCheck] = []

    checks.append(
        DiagnosticCheck(
            DiagnosticStatus.OK,
            "Configuracion",
            f"Configuracion cargada para {settings.project_name} {settings.version}",
        )
    )

    checks.extend(_check_required_directories(settings))
    checks.extend(_check_documents(settings))
    checks.extend(_check_manifest(settings))
    checks.extend(_check_processed_documents(settings))
    checks.extend(_check_chunks(settings))
    checks.extend(_check_chroma(settings, chroma_inspector=chroma_inspector))
    checks.extend(_check_importable_modules())
    checks.extend(_check_evaluation_files(settings))
    checks.extend(validate_privacy(settings))
    checks.extend(_check_logs(settings))

    return DoctorReport(checks=checks)


def validate_privacy(
    settings: AppSettings | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    source_root: Path | None = None,
) -> list[DiagnosticCheck]:
    settings = settings or get_settings()
    environ = os.environ if environ is None else environ
    source_root = source_root or PROJECT_ROOT / "src" / "rag_chatbot"
    checks: list[DiagnosticCheck] = []

    if settings.allow_external_llm:
        checks.append(
            DiagnosticCheck(
                DiagnosticStatus.ERROR,
                "Privacidad ALLOW_EXTERNAL_LLM",
                "ALLOW_EXTERNAL_LLM esta activo; el MVP debe mantenerse local-first.",
            )
        )
    else:
        checks.append(
            DiagnosticCheck(
                DiagnosticStatus.OK,
                "Privacidad ALLOW_EXTERNAL_LLM",
                "ALLOW_EXTERNAL_LLM=false.",
            )
        )

    if settings.llm_provider.casefold() != "none":
        checks.append(
            DiagnosticCheck(
                DiagnosticStatus.ERROR,
                "Privacidad LLM_PROVIDER",
                f"LLM_PROVIDER={settings.llm_provider}; para el MVP debe ser none.",
            )
        )
    else:
        checks.append(
            DiagnosticCheck(
                DiagnosticStatus.OK,
                "Privacidad LLM_PROVIDER",
                "LLM_PROVIDER=none.",
            )
        )

    active_external_vars = [
        key for key in EXTERNAL_API_ENV_VARS if environ.get(key)
    ]
    if active_external_vars:
        checks.append(
            DiagnosticCheck(
                DiagnosticStatus.WARNING,
                "Privacidad variables externas",
                "Variables de API externas presentes en el entorno: "
                + ", ".join(active_external_vars)
                + ". El flujo actual no debe usarlas.",
            )
        )
    else:
        checks.append(
            DiagnosticCheck(
                DiagnosticStatus.OK,
                "Privacidad variables externas",
                "No se detectaron claves externas activas en el entorno.",
            )
        )

    external_imports = _find_external_service_imports(source_root)
    if external_imports:
        checks.append(
            DiagnosticCheck(
                DiagnosticStatus.ERROR,
                "Privacidad imports externos",
                "Imports externos detectados en codigo fuente: "
                + "; ".join(external_imports),
            )
        )
    else:
        checks.append(
            DiagnosticCheck(
                DiagnosticStatus.OK,
                "Privacidad imports externos",
                "No se detectaron imports de OpenAI, Gemini ni LangChain en src.",
            )
        )

    return checks


def format_doctor_report(report: DoctorReport) -> list[str]:
    return [
        f"[{check.status.value}] {check.label}: {check.message}"
        for check in report.checks
    ]


def inspect_chroma(settings: AppSettings) -> ChromaDiagnosticInfo:
    if find_spec("chromadb") is None:
        return ChromaDiagnosticInfo(
            available=False,
            error_message=(
                "chromadb no esta instalado. Actualiza dependencias con: "
                "python -m pip install -e ."
            ),
        )

    if not settings.chroma_dir.exists():
        return ChromaDiagnosticInfo(
            available=True,
            collection_exists=False,
            error_message=f"No existe la carpeta Chroma: {settings.chroma_dir}",
        )

    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(settings.chroma_dir))
        collection_names = _collection_names(client.list_collections())
        if settings.chroma_collection_name not in collection_names:
            return ChromaDiagnosticInfo(
                available=True,
                collection_exists=False,
                error_message=(
                    "No existe la coleccion configurada: "
                    f"{settings.chroma_collection_name}"
                ),
            )

        collection = client.get_collection(settings.chroma_collection_name)
        return ChromaDiagnosticInfo(
            available=True,
            collection_exists=True,
            vector_count=collection.count(),
        )
    except Exception as exc:
        return ChromaDiagnosticInfo(
            available=True,
            collection_exists=False,
            error_message=str(exc),
        )


def _check_required_directories(settings: AppSettings) -> list[DiagnosticCheck]:
    checks: list[DiagnosticCheck] = []
    for directory in settings.required_directories:
        if directory.exists() and directory.is_dir():
            checks.append(
                DiagnosticCheck(
                    DiagnosticStatus.OK,
                    "Carpeta",
                    f"Existe: {directory}",
                )
            )
        else:
            checks.append(
                DiagnosticCheck(
                    DiagnosticStatus.ERROR,
                    "Carpeta",
                    f"No existe: {directory}. Ejecuta rag-chatbot init-dirs.",
                )
            )
    return checks


def _check_documents(settings: AppSettings) -> list[DiagnosticCheck]:
    documents = discover_documents(settings.documents_dir)
    if documents:
        return [
            DiagnosticCheck(
                DiagnosticStatus.OK,
                "Documentos",
                f"Documentos encontrados: {len(documents)}",
            )
        ]

    return [
        DiagnosticCheck(
            DiagnosticStatus.WARNING,
            "Documentos",
            f"No se detectaron PDF/TXT/DOCX en {settings.documents_dir}.",
        )
    ]


def _check_manifest(settings: AppSettings) -> list[DiagnosticCheck]:
    if settings.manifest_db_path.exists():
        return [
            DiagnosticCheck(
                DiagnosticStatus.OK,
                "Manifest",
                f"Manifest SQLite existe: {settings.manifest_db_path}",
            )
        ]

    if settings.manifest_db_path.parent.exists():
        return [
            DiagnosticCheck(
                DiagnosticStatus.WARNING,
                "Manifest",
                "Manifest no existe todavia; se creara al ejecutar ingest.",
            )
        ]

    return [
        DiagnosticCheck(
            DiagnosticStatus.ERROR,
            "Manifest",
            f"No se puede crear manifest porque falta {settings.manifest_db_path.parent}.",
        )
    ]


def _check_processed_documents(settings: AppSettings) -> list[DiagnosticCheck]:
    processed_count = _count_files(settings.processed_dir, "*.json")
    if processed_count:
        return [
            DiagnosticCheck(
                DiagnosticStatus.OK,
                "Documentos procesados",
                f"JSON procesados encontrados: {processed_count}",
            )
        ]

    return [
        DiagnosticCheck(
            DiagnosticStatus.WARNING,
            "Documentos procesados",
            "No hay documentos procesados. Ejecuta rag-chatbot ingest.",
        )
    ]


def _check_chunks(settings: AppSettings) -> list[DiagnosticCheck]:
    checks: list[DiagnosticCheck] = []
    if settings.chunks_file.exists():
        line_count = _count_non_empty_lines(settings.chunks_file)
        if line_count:
            checks.append(
                DiagnosticCheck(
                    DiagnosticStatus.OK,
                    "Chunks",
                    f"Chunks generados: {line_count}",
                )
            )
        else:
            checks.append(
                DiagnosticCheck(
                    DiagnosticStatus.WARNING,
                    "Chunks",
                    f"{settings.chunks_file} existe pero no contiene chunks.",
                )
            )
    else:
        checks.append(
            DiagnosticCheck(
                DiagnosticStatus.WARNING,
                "Chunks",
                f"No existe {settings.chunks_file}. Ejecuta rag-chatbot build-chunks.",
            )
        )

    manifest_path = settings.chunks_dir / "chunk_manifest.json"
    if manifest_path.exists():
        checks.append(
            DiagnosticCheck(
                DiagnosticStatus.OK,
                "Chunk manifest",
                f"Existe: {manifest_path}",
            )
        )
    else:
        checks.append(
            DiagnosticCheck(
                DiagnosticStatus.WARNING,
                "Chunk manifest",
                "No hay chunk_manifest.json generado todavia.",
            )
        )

    return checks


def _check_chroma(settings: AppSettings, *, chroma_inspector=None) -> list[DiagnosticCheck]:
    inspector = chroma_inspector or inspect_chroma
    info = inspector(settings)
    checks: list[DiagnosticCheck] = []

    if not info.available:
        checks.append(
            DiagnosticCheck(
                DiagnosticStatus.WARNING,
                "Chroma",
                info.error_message or "chromadb no esta disponible.",
            )
        )
        return checks

    checks.append(
        DiagnosticCheck(
            DiagnosticStatus.OK,
            "Chroma",
            "chromadb esta disponible localmente.",
        )
    )

    if not info.collection_exists:
        checks.append(
            DiagnosticCheck(
                DiagnosticStatus.WARNING,
                "Indice Chroma",
                info.error_message
                or f"No existe la coleccion {settings.chroma_collection_name}.",
            )
        )
        return checks

    if info.vector_count is None:
        checks.append(
            DiagnosticCheck(
                DiagnosticStatus.WARNING,
                "Indice Chroma",
                "Coleccion encontrada, pero no se pudo obtener el numero de vectores.",
            )
        )
    elif info.vector_count > 0:
        checks.append(
            DiagnosticCheck(
                DiagnosticStatus.OK,
                "Indice Chroma",
                f"Coleccion {settings.chroma_collection_name}: {info.vector_count} vectores.",
            )
        )
    else:
        checks.append(
            DiagnosticCheck(
                DiagnosticStatus.WARNING,
                "Indice Chroma",
                f"Coleccion {settings.chroma_collection_name} existe pero esta vacia.",
            )
        )

    return checks


def _check_importable_modules() -> list[DiagnosticCheck]:
    modules = [
        ("API importable", "rag_chatbot.api.main"),
        ("UI importable", "rag_chatbot.ui.streamlit_app"),
    ]
    checks: list[DiagnosticCheck] = []

    for label, module_name in modules:
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            checks.append(
                DiagnosticCheck(
                    DiagnosticStatus.ERROR,
                    label,
                    f"No se pudo importar {module_name}: {exc}",
                )
            )
        else:
            checks.append(
                DiagnosticCheck(
                    DiagnosticStatus.OK,
                    label,
                    f"{module_name} importable.",
                )
            )

    if find_spec("streamlit") is None:
        checks.append(
            DiagnosticCheck(
                DiagnosticStatus.WARNING,
                "Streamlit",
                "streamlit no esta instalado en el entorno activo.",
            )
        )
    else:
        checks.append(
            DiagnosticCheck(
                DiagnosticStatus.OK,
                "Streamlit",
                "streamlit esta disponible.",
            )
        )

    return checks


def _check_evaluation_files(settings: AppSettings) -> list[DiagnosticCheck]:
    checks: list[DiagnosticCheck] = []
    if settings.eval_dataset_path.exists():
        checks.append(
            DiagnosticCheck(
                DiagnosticStatus.OK,
                "Evaluacion dataset",
                f"Existe: {settings.eval_dataset_path}",
            )
        )
    else:
        checks.append(
            DiagnosticCheck(
                DiagnosticStatus.WARNING,
                "Evaluacion dataset",
                f"No existe {settings.eval_dataset_path}.",
            )
        )

    report_path = settings.eval_reports_dir / JSON_REPORT_NAME
    if report_path.exists():
        checks.append(
            DiagnosticCheck(
                DiagnosticStatus.OK,
                "Evaluacion reporte",
                f"Ultimo reporte encontrado: {report_path}",
            )
        )
    else:
        checks.append(
            DiagnosticCheck(
                DiagnosticStatus.WARNING,
                "Evaluacion reporte",
                "No hay reporte de evaluacion generado todavia.",
            )
        )

    return checks


def _check_logs(settings: AppSettings) -> list[DiagnosticCheck]:
    if not settings.log_dir.exists():
        return [
            DiagnosticCheck(
                DiagnosticStatus.WARNING,
                "Logs",
                f"No existe la carpeta de logs: {settings.log_dir}",
            )
        ]

    long_lines: list[str] = []
    for log_path in settings.log_dir.glob("*.log"):
        try:
            with log_path.open("r", encoding="utf-8", errors="replace") as file:
                for line_number, line in enumerate(file, start=1):
                    if len(line) > MAX_LOG_LINE_LENGTH:
                        long_lines.append(f"{log_path.name}:{line_number}")
                        break
        except OSError:
            continue

    if long_lines:
        return [
            DiagnosticCheck(
                DiagnosticStatus.WARNING,
                "Logs privacidad",
                "Hay lineas inusualmente largas en logs; revisar que no contengan "
                "texto documental completo: "
                + ", ".join(long_lines),
            )
        ]

    return [
        DiagnosticCheck(
            DiagnosticStatus.OK,
            "Logs privacidad",
            "No se detectaron lineas largas que sugieran texto documental completo.",
        )
    ]


def _find_external_service_imports(source_root: Path) -> list[str]:
    if not source_root.exists():
        return []

    findings: list[str] = []
    for path in sorted(source_root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue

        for node in ast.walk(tree):
            module_names: list[str] = []
            if isinstance(node, ast.Import):
                module_names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                module_names.append(node.module)

            for module_name in module_names:
                if _is_external_service_module(module_name):
                    findings.append(f"{path.name}:{module_name}")

    return findings


def _is_external_service_module(module_name: str) -> bool:
    normalized = module_name.casefold()
    return any(
        normalized == prefix or normalized.startswith(prefix + ".")
        for prefix in EXTERNAL_IMPORT_PREFIXES
    )


def _collection_names(collections) -> set[str]:
    names: set[str] = set()
    for collection in collections:
        if isinstance(collection, str):
            names.add(collection)
        elif hasattr(collection, "name"):
            names.add(collection.name)
    return names


def _count_files(directory: Path, pattern: str) -> int:
    if not directory.exists():
        return 0
    return sum(1 for _ in directory.glob(pattern))


def _count_non_empty_lines(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                count += 1
    return count
