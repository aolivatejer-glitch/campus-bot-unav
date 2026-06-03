# rag-chatbot

Chatbot documental RAG local-first para documentos normativos y politicas de la
Universidad de Navarra. El sistema ingiere documentos locales, extrae texto,
genera chunks trazables, crea embeddings locales, indexa en Chroma local y
responde de forma extractiva con fuentes.

Por defecto no usa OpenAI, Gemini, LangChain ni ningun LLM externo. Los
documentos, chunks, preguntas y contexto no deben salir del equipo.

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

No implementado todavia:

- OCR.
- Reranking.
- Docker.
- Autenticacion compleja.
- LLM externo.
- LLM local generativo.
- Despliegue productivo.

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

```powershell
python -m pip install -e .
```

Si quieres personalizar rutas o parametros:

```powershell
Copy-Item .env.example .env
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
- `ALLOW_EXTERNAL_LLM=false` y `LLM_PROVIDER=none`: privacidad local-first.

## Flujo recomendado desde cero

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

Consulta RAG extractiva:

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/query" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"question":"Que documentos hablan sobre compliance?","top_k":5}'
```

`/retrieve` devuelve resultados semanticos crudos. `/query` devuelve una
respuesta extractiva con fuentes. Por defecto no se devuelven chunks completos.

## UI local

Flujo recomendado:

```powershell
rag-chatbot serve
```

En otra terminal:

```powershell
rag-chatbot ui
```

La interfaz permite escribir una pregunta, ajustar `top_k` y `min_score`,
mostrar u ocultar fuentes y mostrar chunks recuperados solo si se solicita.

Ejemplos de preguntas:

- "Que documentos hablan sobre compliance?"
- "Que dice el protocolo sobre acoso entre estudiantes?"
- "Que establece la politica de IA?"
- "Que normas regulan la convivencia en la Universidad de Navarra?"

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
- La respuesta es extractiva, no generativa.
- Puede repetir fragmentos.
- Depende de la calidad de extraccion, chunking e indice.
- No sustituye revision juridica, normativa o experta.

## Tests

```powershell
python -m pytest
```

Las pruebas unitarias no requieren descargar modelos reales ni tener un indice
Chroma grande poblado.
