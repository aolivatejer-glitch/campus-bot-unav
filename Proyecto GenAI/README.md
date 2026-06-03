# rag-chatbot

Chatbot documental RAG local-first para documentos normativos y politicas de la
Universidad de Navarra. El sistema ingiere documentos locales, extrae texto,
genera chunks trazables, crea embeddings locales, indexa en Chroma local y
responde de forma extractiva con fuentes.

Por defecto no usa OpenAI, Gemini, LangChain ni ningun LLM externo. El modo
base es extractivo local. Gemini existe solo como modo generativo opcional y
debe activarse de forma explicita en `.env`.

## Estado del MVP

Implementado:

- Configuracion centralizada, logging, CLI y pruebas.
- Ingesta PDF/TXT/DOCX con hashes y manifest SQLite.
- Chunking trazable por documento y pagina.
- Embeddings locales con `sentence-transformers`.
- Chroma persistente en `storage/chroma/`.
- Busqueda semantica con ranking, score, fuente, pagina y `chunk_id`.
- Respuesta RAG extractiva local con rechazo por contexto insuficiente.
- Evaluacion basica con JSONL, reportes JSON/CSV y resumen.
- API local FastAPI.
- UI local Streamlit consumiendo la API.
- Diagnostico local end-to-end con `doctor`.
- Flujo local completo con `run-local-pipeline`.
- Modo generativo opcional con Gemini, protegido por configuracion y guardrails.

No implementado todavia:

- OCR.
- Reranking.
- Docker.
- Autenticacion compleja.
- LLM local generativo.
- Despliegue productivo.

## Requisitos

Requisitos del entorno:

- Windows con PowerShell o CMD.
- Python `>=3.12`. El proyecto se ha usado con Python `3.12.6`.
- `pip` actualizado.
- Documentos fuente en `Documentos/`.
- Internet solo para dos casos puntuales:
  - descargar dependencias o modelos la primera vez;
  - usar Gemini si activas explicitamente el modo LLM.

Dependencias principales declaradas en `pyproject.toml`:

- `PyMuPDF`: lectura de PDF con texto seleccionable.
- `python-docx`: lectura de DOCX.
- `sentence-transformers`: embeddings locales. Si no estuviera instalado en tu
  entorno, instalalo antes de construir el indice.
- `chromadb`: base vectorial local persistente.
- `fastapi` y `uvicorn`: API local.
- `streamlit`: interfaz local.
- `typer`: CLI.
- `pydantic-settings`: configuracion desde `.env`.
- `google-genai`: cliente Gemini opcional.
- `pytest`: pruebas, disponible como extra de desarrollo.

Instalacion recomendada:

```powershell
python -m pip install -e .
```

Para instalar tambien herramientas de desarrollo y pruebas:

```powershell
python -m pip install -e ".[dev]"
```

El proyecto usa `pyproject.toml` como fuente principal de dependencias. No hace
falta un `requirements.txt` separado para la instalacion editable.

Si vas a usar Gemini, configura `.env` y recuerda que solo se llamara al
proveedor externo cuando `ALLOW_EXTERNAL_LLM=true`, `LLM_PROVIDER=gemini` y la
consulta se ejecute con `--mode llm`.

## Estructura

```text
Proyecto GenAI/
  Documentos/
  data/
    processed/
    chunks/
    eval/
      reports/
  storage/
    chroma/
    manifest.sqlite
  logs/
  src/
    rag_chatbot/
      api/
      chunking/
      embeddings/
      evaluation/
      indexing/
      ingestion/
      rag/
      retrieval/
      ui/
  tests/
```

`Documentos/` es la carpeta de entrada configurable. No se mueve ni se borra por
los comandos del proyecto.

## Instalacion

Desde la raiz del proyecto:

PowerShell:

```powershell
python -m pip install -e .
```

Si quieres personalizar rutas o parametros:

```powershell
Copy-Item .env.example .env
```

CMD:

```cmd
python -m pip install -e .
copy .env.example .env
```

Variables principales:

- `DOCUMENTS_DIR`: carpeta de documentos fuente.
- `PROCESSED_DIR`: salida JSON de documentos procesados.
- `CHUNKS_FILE`: archivo JSONL con chunks.
- `MANIFEST_DB_PATH`: manifest SQLite con hashes y estado.
- `CHROMA_DIR`: persistencia local de Chroma.
- `CHROMA_COLLECTION_NAME`: coleccion vectorial.
- `EMBEDDING_MODEL`: modelo local, por defecto `intfloat/multilingual-e5-base`.
- `EMBEDDING_DEVICE`: `cpu` por defecto.
- `TOP_K`, `MIN_RETRIEVAL_SCORE`: parametros de recuperacion.
- `MIN_CONTEXT_CHARS`, `MAX_CONTEXT_CHARS`, `MAX_SOURCES`: control de respuesta.
- `API_HOST`, `API_PORT`, `API_BASE_URL`: API y UI local.
- `API_REQUEST_TIMEOUT_SECONDS`: tiempo maximo de espera del cliente UI para
  consultas a la API. Por defecto `120`.
- `ALLOW_EXTERNAL_LLM=false` y `LLM_PROVIDER=none`: privacidad local-first.

## Flujo recomendado desde cero

PowerShell o CMD:

```powershell
python -m pip install -e .
rag-chatbot doctor
rag-chatbot ingest
rag-chatbot build-chunks
rag-chatbot build-index --reset
rag-chatbot ask "Que documentos hablan sobre compliance?"
rag-chatbot serve
```

En otra terminal:

```powershell
rag-chatbot ui
```

Tambien puedes ejecutar las tres etapas de preparacion de una vez:

```powershell
rag-chatbot run-local-pipeline --reset-index
```

Para una prueba rapida de indexacion parcial:

```powershell
rag-chatbot run-local-pipeline --reset-index --limit 50
```

La evaluacion no se ejecuta automaticamente. Si quieres incluirla:

```powershell
rag-chatbot run-local-pipeline --reset-index --run-eval
```

## Uso desde Windows CMD

La mayoria de comandos del proyecto son iguales en PowerShell y CMD:

```cmd
python -m pip install -e .
rag-chatbot doctor
rag-chatbot ingest
rag-chatbot build-chunks
rag-chatbot build-index --reset
rag-chatbot ask "Que documentos hablan sobre compliance?"
rag-chatbot serve
```

En otra ventana de CMD:

```cmd
rag-chatbot ui
```

Copiar el archivo de configuracion:

```cmd
copy .env.example .env
```

Definir variables temporales solo para la sesion actual de CMD:

```cmd
set API_BASE_URL=http://127.0.0.1:8000
set ALLOW_EXTERNAL_LLM=false
set LLM_PROVIDER=none
```

Si vas a probar Gemini desde CMD, configura `.env` o define temporalmente:

```cmd
set ALLOW_EXTERNAL_LLM=true
set LLM_PROVIDER=gemini
set GEMINI_API_KEY=tu_clave
rag-chatbot ask "Que documentos hablan sobre compliance?" --mode llm
```

Para continuar comandos largos en varias lineas en CMD se usa `^`, no el
backtick de PowerShell:

```cmd
curl -X POST "http://127.0.0.1:8000/query" ^
  -H "Content-Type: application/json" ^
  -d "{\"question\":\"Que documentos hablan sobre compliance?\",\"mode\":\"extractive\",\"top_k\":3}"
```

## Diagnostico

```powershell
rag-chatbot doctor
```

`doctor` valida:

- Configuracion.
- Carpetas requeridas.
- Existencia de `DOCUMENTS_DIR`.
- Documentos PDF/TXT/DOCX detectables.
- Manifest SQLite o posibilidad de crearlo.
- Documentos procesados.
- `chunks.jsonl` y `chunk_manifest.json`.
- Disponibilidad de Chroma.
- Existencia de coleccion y numero de vectores.
- API importable.
- UI importable.
- Dataset y reportes de evaluacion.
- `ALLOW_EXTERNAL_LLM=false`.
- `LLM_PROVIDER=none`.
- Variables de API externas presentes en el entorno.
- Imports de OpenAI, Gemini o LangChain en `src/`.
- Lineas de log inusualmente largas que podrian indicar texto documental completo.

Un reporte puede incluir advertencias normales, por ejemplo si aun no has
ejecutado evaluacion o si el indice esta vacio.

## Comandos principales

Preparacion:

```powershell
rag-chatbot init-dirs
rag-chatbot ingest
rag-chatbot build-chunks
rag-chatbot build-index
rag-chatbot build-index --reset
rag-chatbot index-info
rag-chatbot run-local-pipeline --reset-index
```

Consulta:

```powershell
rag-chatbot search "Que documentos hablan sobre compliance?"
rag-chatbot search "pregunta" --top-k 5 --min-score 0.3
rag-chatbot search "pregunta" --show-text
rag-chatbot ask "Que dice el protocolo sobre acoso entre estudiantes?" --top-k 5
rag-chatbot ask "pregunta" --show-chunks
rag-chatbot ask "pregunta" --hide-sources
```

`search` devuelve evidencias: ranking, score, distancia, fuente, pagina,
`chunk_id` y snippet.

`ask` construye una respuesta extractiva con esas evidencias. Si no hay contexto
suficiente, responde que no puede contestar con seguridad.

Evaluacion:

```powershell
rag-chatbot eval-run
rag-chatbot eval-run --limit 5
rag-chatbot eval-grid
rag-chatbot eval-grid --top-k-values 3,5,8 --min-score-values 0.2,0.3,0.4
rag-chatbot eval-summary
```

API y UI:

```powershell
rag-chatbot serve --host 127.0.0.1 --port 8000
rag-chatbot serve --reload
rag-chatbot ui
rag-chatbot ui --api-base-url http://127.0.0.1:8000
```

## API local

La API reutiliza los pipelines locales. No llama servicios externos.

Endpoints:

- `GET /health`
- `GET /config`
- `POST /retrieve`
- `POST /query`
- `POST /ingest`
- `POST /build-chunks`
- `POST /build-index`
- `GET /index-info`
- `GET /eval-summary`

Prueba rapida:

```powershell
curl http://127.0.0.1:8000/health
```

En PowerShell, si `curl` se comporta como alias, usa:

```powershell
curl.exe http://127.0.0.1:8000/health
```

En CMD:

```cmd
curl http://127.0.0.1:8000/health
```

Consulta RAG extractiva:

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/query" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"question":"Que documentos hablan sobre compliance?","top_k":5}'
```

Consulta equivalente en CMD:

```cmd
curl -X POST "http://127.0.0.1:8000/query" ^
  -H "Content-Type: application/json" ^
  -d "{\"question\":\"Que documentos hablan sobre compliance?\",\"mode\":\"extractive\",\"top_k\":5}"
```

Consulta Gemini opcional desde CMD, si `.env` permite LLM externo:

```cmd
curl -X POST "http://127.0.0.1:8000/query" ^
  -H "Content-Type: application/json" ^
  -d "{\"question\":\"Que documentos hablan sobre compliance?\",\"mode\":\"llm\",\"top_k\":3,\"min_score\":0.84}"
```

`/retrieve` devuelve resultados semanticos crudos. `/query` devuelve una
respuesta extractiva con fuentes. Por defecto no se devuelven chunks completos.

## UI local

La interfaz Streamlit esta pensada para demo academica de una consulta RAG:

```text
Pregunta -> respuesta -> fuentes -> chunks opcionales
```

No implementa memoria conversacional, login ni multiples conversaciones. Conserva
solo la ultima consulta visible para facilitar la demostracion.

Flujo recomendado:

PowerShell o CMD:

```powershell
rag-chatbot serve
```

En otra terminal:

```powershell
rag-chatbot ui
```

Tambien puedes fijar la URL de API:

```cmd
rag-chatbot ui --api-base-url http://127.0.0.1:8000
```

La interfaz permite:

- Escribir una pregunta principal.
- Elegir modo `RAG extractivo local` o `RAG generativo con Gemini`.
- Ajustar `top_k` y `min_score`.
- Mostrar u ocultar fuentes.
- Mostrar chunks recuperados solo si se solicita.
- Ver estado de API, indice vectorial y modelo de embeddings.
- Revisar advertencias de contexto insuficiente, fallback de Gemini o pregunta
  fuera de dominio.

Si las consultas generativas tardan demasiado, ajusta en `.env`:

```env
API_REQUEST_TIMEOUT_SECONDS=120
```

Puedes subirlo, por ejemplo a `180`, si Gemini tarda mas en responder durante
una demo. La UI mostrara un mensaje amigable si se supera ese tiempo.

Ejemplos de preguntas:

- "Que dice el protocolo sobre acoso entre estudiantes?"
- "Que establece la politica de IA?"
- "Que normas regulan la convivencia en la Universidad de Navarra?"
- "Que informacion contiene el documento sobre compliance penal?"
- "Que dice la normativa sobre reconocimiento de creditos de grado?"

Modo extractivo:

```cmd
rag-chatbot ask "Que establece la politica de IA?" --mode extractive
```

Modo Gemini:

```cmd
rag-chatbot ask "Que establece la politica de IA?" --mode llm
```

Para compartir una demo puntual con Cloudflare Tunnel, levanta primero API y UI
en local:

```cmd
rag-chatbot serve
```

En otra terminal:

```cmd
rag-chatbot ui
```

Y en una tercera terminal, si tienes `cloudflared` instalado:

```cmd
cloudflared tunnel --url http://127.0.0.1:8501
```

Limitaciones de esta forma de compartir:

- No hay autenticacion.
- No es despliegue productivo.
- Quien tenga el enlace puede consultar la demo mientras el tunel este activo.
- Si seleccionas modo Gemini y esta permitido en `.env`, se enviaran pregunta y
  fragmentos recuperados al proveedor externo.
- No compartas el tunel con documentos sensibles sin controles adicionales.

## Evaluacion

El dataset esta en:

```text
data/eval/evaluation_questions.jsonl
```

Cada linea es un JSON con:

- `question_id`
- `question`
- `expected_keywords`
- `expected_files`
- `should_have_answer`
- `notes`

Los reportes quedan en:

```text
data/eval/reports/evaluation_report.json
data/eval/reports/evaluation_report.csv
```

Metricas:

- `has_answer`
- `expected_answer_behavior`
- `keyword_hit_rate`
- `expected_file_hit`
- `source_count`
- `retrieved_chunk_count`
- `top_score`
- `avg_score`
- `passed`

No usa LLM-as-judge y no reemplaza validacion humana.

## Calibracion de calidad

La calibracion ayuda a ajustar retrieval y rechazo sin agregar reranking ni LLM
generativo.

Ejecuta una evaluacion normal:

```powershell
rag-chatbot eval-run
rag-chatbot eval-summary
```

`eval-run` genera:

```text
data/eval/reports/evaluation_report.json
data/eval/reports/evaluation_report.csv
```

Cada pregunta incluye `failure_reasons` cuando falla. Motivos frecuentes:

- `expected_file_not_found`: no aparecio ningun archivo esperado en las fuentes.
- `low_keyword_hit_rate`: respuesta o chunks no contienen suficientes keywords esperadas.
- `answered_when_should_reject`: falso positivo; respondio una pregunta fuera de dominio.
- `rejected_when_should_answer`: falso negativo; rechazo una pregunta que debia contestar.
- `no_sources`: respuesta sin fuentes.
- `low_top_score`: mejor resultado con score bajo.
- `answer_too_short`: respuesta extractiva demasiado breve.
- `out_of_domain_not_rejected`: pregunta fuera de dominio no rechazada.

Para comparar parametros:

```powershell
rag-chatbot eval-grid
rag-chatbot eval-grid --limit 5
rag-chatbot eval-grid --top-k-values 3,5,8 --min-score-values 0.2,0.3,0.4
```

Genera:

```text
data/eval/reports/evaluation_grid_report.json
data/eval/reports/evaluation_grid_report.csv
```

El grid calcula:

- `pass_rate`
- `expected_file_hit_rate`
- `false_positive_count`
- `false_negative_count`
- `avg_keyword_hit_rate`
- `avg_top_score`
- `avg_source_count`

La configuracion recomendada se elige con una regla simple:

1. Mayor `pass_rate`.
2. Menor cantidad de falsos positivos.
3. Mayor `expected_file_hit_rate`.
4. Menor `top_k` para eficiencia.

No modifica `.env`; solo recomienda valores.

Interpretacion:

- Muchos falsos positivos: sube `MIN_RETRIEVAL_SCORE` o revisa preguntas fuera de dominio.
- Muchos falsos negativos: baja `MIN_RETRIEVAL_SCORE`, sube `top_k` o revisa chunking.
- Bajo `expected_file_hit_rate`: revisa keywords esperadas, chunking y cobertura del dataset.
- Bajo `keyword_hit_rate`: comprueba si los snippets conservan terminos normativos clave.
- `top_score` bajo en preguntas del dominio: puede faltar texto extraido, haber chunks pobres o necesitar reranking local futuro.

Conviene considerar reranking local despues si:

- El archivo correcto aparece entre los chunks recuperados, pero no en las primeras posiciones.
- `top_k` alto mejora recall pero introduce muchas fuentes irrelevantes.
- Hay falsos positivos que no se corrigen bien solo con `MIN_RETRIEVAL_SCORE`.
- Las preguntas del dominio son semanticamente cercanas entre documentos normativos distintos.

## Configuracion recomendada para demo local

Configuracion candidata tras calibracion con el corpus actual:

```env
TOP_K=3
MIN_RETRIEVAL_SCORE=0.84
MAX_SOURCES=3
MIN_CONTEXT_CHARS=500
ENABLE_DOMAIN_GUARDRAILS=true
```

Esta configuracion busca reducir falsos positivos, es decir, preguntas que estan
fuera del corpus pero reciben una respuesta extractiva. El umbral se sube porque
las preguntas fuera de dominio tambien pueden recuperar fragmentos con scores
moderados/altos.

Un falso positivo ocurre cuando el sistema responde algo que deberia rechazar.
Un falso negativo ocurre cuando rechaza una pregunta que si deberia contestar
con los documentos indexados.

Cuando bajar el umbral:

- Aparecen falsos negativos en preguntas validas del dominio.
- Preguntas esperadas no recuperan ningun chunk.
- El corpus crece con documentos relevantes y la recuperacion se vuelve mas diversa.

Cuando subir el umbral:

- Persisten falsos positivos.
- Preguntas claramente externas recuperan fragmentos genericos.
- La demo permite preferir rechazo conservador sobre cobertura amplia.

Cuando considerar reranking local:

- El documento correcto se recupera, pero queda por debajo de resultados menos relevantes.
- Aumentar `top_k` mejora recall pero mete ruido.
- Ajustar `MIN_RETRIEVAL_SCORE` no separa bien preguntas validas de preguntas externas.

Valida siempre con:

```powershell
rag-chatbot eval-run
rag-chatbot eval-grid --top-k-values 3,5 --min-score-values 0.75,0.8,0.82,0.84,0.86
rag-chatbot eval-summary
```

## Restricciones de dominio

La busqueda vectorial siempre devuelve los chunks mas parecidos dentro del indice,
incluso cuando la pregunta no pertenece al corpus. Por eso el pipeline `ask`
incluye guardrails de dominio antes de construir una respuesta extractiva.

Dominio permitido:

```env
ENABLE_DOMAIN_GUARDRAILS=true
DOMAIN_NAME=normativas y políticas de la Universidad de Navarra
```

El chatbot esta limitado a normativas y politicas de la Universidad de Navarra
cargadas en el sistema. Usa dos señales simples:

- Terminos permitidos, como `compliance`, `convivencia`, `acoso`, `politica de IA`,
  `creditos`, `practicas academicas`, `defensoria universitaria` o `docencia`.
- Terminos claramente fuera de alcance, como `Rey`, `23F`, `Champions League`,
  `recetas medicas`, `prediccion meteorologica`, `restaurantes`, `hoteles` o
  `turismo`.

Si una pregunta contiene un termino bloqueado claro, se rechaza sin consultar el
indice. Si es ambigua, no se rechaza automaticamente: el sistema deja que retrieval,
`MIN_RETRIEVAL_SCORE` y `MIN_CONTEXT_CHARS` decidan si hay evidencia suficiente.

Ejemplos aceptados:

```powershell
rag-chatbot ask "Que documentos hablan sobre compliance?"
rag-chatbot ask "Que dice el protocolo sobre acoso entre estudiantes?"
rag-chatbot ask "Que establece la politica de IA?"
```

Ejemplos rechazados:

```powershell
rag-chatbot ask "Que documentos mencionan al Rey?"
rag-chatbot ask "Que equipo gano la ultima Champions League?"
rag-chatbot ask "Que dice el documento sobre recetas medicas?"
```

Cuando el rechazo es por dominio, la API `/query` devuelve
`rejection_reason=out_of_domain`. La UI muestra una advertencia amigable y no
presenta fuentes ni chunks como si fueran evidencia suficiente.

Limitaciones: esta estrategia no es un clasificador perfecto. Las listas de
terminos son una ayuda, no la unica regla. Si aparecen falsos positivos que no
se corrigen con umbral y guardrails, el siguiente paso razonable seria evaluar
reranking local.

## Limpieza de indices de PDF

Algunos PDF incluyen indices o tablas de contenido con lineas dominadas por
secuencias de puntos, tambien llamadas dot leaders:

```text
3. Evaluacion de riesgos penales ........................................ 4
4. Formacion ............................................................ 4
```

Si esas lineas entran al chunking, consumen espacio, generan embeddings ruidosos
y pueden aparecer en snippets o respuestas aunque no aporten contenido
explicativo. El chunking aplica una limpieza conservadora antes de generar
`chunks.jsonl`:

- Normaliza secuencias largas de puntos sin tocar puntos normales, decimales ni
  abreviaturas.
- Elimina lineas que parecen entradas de indice o tabla de contenido.
- Marca metadata de calidad por chunk: `is_toc_candidate`, `dot_leader_count` y
  `text_quality`.
- Por defecto excluye chunks candidatos a indice para que no compitan en Chroma.

Configuracion:

```env
EXCLUDE_TOC_CHUNKS=true
CLEAN_DOT_LEADERS=true
```

Si ya habia chunks e indice creados antes de esta limpieza, reconstruye ambos:

```powershell
rag-chatbot build-chunks
rag-chatbot build-index --reset
```

La indexacion tambien salta defensivamente chunks que lleguen con
`is_toc_candidate=true`, por compatibilidad con archivos JSONL antiguos o
generados manualmente.

## RAG generativo opcional con Gemini

El modo por defecto sigue siendo extractivo local. En ese modo no se llama a
ningun proveedor externo:

```powershell
rag-chatbot ask "Que documentos hablan sobre compliance?" --mode extractive
```

Gemini solo se usa si se solicita explicitamente y la configuracion lo permite:

```powershell
rag-chatbot ask "Que documentos hablan sobre compliance?" --mode llm
```

Configuracion en `.env`:

```env
ALLOW_EXTERNAL_LLM=true
LLM_PROVIDER=gemini
GEMINI_API_KEY=tu_clave
GEMINI_MODEL=gemini-2.5-flash
LLM_TEMPERATURE=0.2
LLM_MAX_CONTEXT_CHARS=6000
LLM_MODE_DEFAULT=extractive
```

Reglas de seguridad del modo generativo:

- Si `ALLOW_EXTERNAL_LLM=false`, Gemini no se llama.
- Si la pregunta fue rechazada por guardrails de dominio, Gemini no se llama.
- Si el contexto recuperado es insuficiente, Gemini no se llama.
- No se envia el corpus completo, solo los chunks recuperados y aceptados.
- La API key se lee desde `.env` y se oculta en `show-config`.
- Si Gemini falla, el sistema vuelve a la respuesta extractiva local.

El prompt obliga al modelo a responder solo con el contexto recuperado, citar
fuentes y decir que no hay informacion suficiente si falta evidencia. Aun asi,
al usar Gemini, la pregunta y los chunks aceptados salen del entorno local hacia
el proveedor externo. Revisa limites, cuotas y condiciones del free tier de
Gemini antes de usarlo en demos o clases.

API:

```json
{
  "question": "Que documentos hablan sobre compliance?",
  "mode": "llm",
  "top_k": 3,
  "min_score": 0.84
}
```

UI:

1. Levanta la API con `rag-chatbot serve`.
2. Abre la UI con `rag-chatbot ui`.
3. Elige `Extractivo local` o `Gemini generativo` en el panel lateral.

## Errores frecuentes

Indice vacio:

```powershell
rag-chatbot build-index --reset
rag-chatbot index-info
```

No existe `chunks.jsonl`:

```powershell
rag-chatbot build-chunks
```

No hay documentos procesados:

```powershell
rag-chatbot ingest
```

Chroma no instalado:

```powershell
python -m pip install -e .
```

Modelo de embeddings no disponible:

- La primera carga puede necesitar internet para descargar pesos desde Hugging Face.
- Ese paso descarga el modelo, no envia documentos ni chunks.
- Despues puede funcionar desde cache local.

Conflictos Keras/TensorFlow/Transformers:

```cmd
set USE_TF=0
set TF_CPP_MIN_LOG_LEVEL=3
set TF_ENABLE_ONEDNN_OPTS=0
```

El proyecto usa embeddings locales con PyTorch. El provider configura esos
valores por defecto antes de cargar `sentence-transformers`, pero definirlos en
la terminal puede ayudar si el entorno ya cargo librerias antes.

API no levantada para la UI:

```powershell
rag-chatbot serve
```

En otra terminal:

```powershell
rag-chatbot ui
```

## Privacidad y limites

- No exponer la API ni la UI publicamente.
- No hay autenticacion compleja.
- No hay cola de tareas; ingesta e indexacion pueden tardar.
- Por defecto la respuesta es extractiva local.
- El modo Gemini es opcional y envia solo pregunta y chunks recuperados aceptados
  al proveedor externo.
- Puede repetir fragmentos.
- Depende de la calidad de extraccion, chunking e indice.
- No sustituye revision juridica, normativa o experta.

## Tests

PowerShell o CMD:

```powershell
python -m pytest
```

Sin cache de pytest:

```cmd
python -m pytest -p no:cacheprovider
```

Las pruebas unitarias no requieren descargar modelos reales ni tener un indice
Chroma grande poblado.
