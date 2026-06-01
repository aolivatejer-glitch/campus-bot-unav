import os
import re
import json
import glob
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def limpiar_texto(texto):
    """Limpia el texto de los PDFs quitando saltos de línea y espacios extra."""
    texto_limpio = texto.replace('\n', ' ')
    texto_limpio = re.sub(r'\s+', ' ', texto_limpio)
    return texto_limpio.strip()

def procesar_documentos():
    print("1. Cargando PDFs desde data/raw con PyMuPDF (Lector avanzado)...")
    documentos_brutos = []
    
    # Buscamos todos los PDFs en la carpeta raw
    archivos_pdf = glob.glob("../data/raw/*.pdf")
    for archivo in archivos_pdf:
        loader = PyMuPDFLoader(archivo)
        documentos_brutos.extend(loader.load())
        
    print(f"-> ¡Se han cargado {len(documentos_brutos)} páginas en total!")

    print("2. Limpiando el texto...")
    for doc in documentos_brutos:
        doc.page_content = limpiar_texto(doc.page_content)

    print("3. Cortando el texto en chunks (fragmentos)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    documentos_fragmentados = text_splitter.split_documents(documentos_brutos)
    print(f"-> El texto se ha dividido en {len(documentos_fragmentados)} fragmentos.")

    print("4. Guardando los datos con los Metadatos inyectados...")
    datos_finales = []
    for chunk in documentos_fragmentados:
        nombre_archivo = os.path.basename(chunk.metadata.get("source", "Desconocido"))
        texto_enriquecido = f"Según el documento oficial '{nombre_archivo}': {chunk.page_content}"
        
        datos_finales.append({
            "contenido": texto_enriquecido,
            "origen": nombre_archivo,
            "pagina": chunk.metadata.get("page", 0)
        })

    ruta_salida = "../data/clean/documentos_procesados.json"
    with open(ruta_salida, "w", encoding="utf-8") as f:
        json.dump(datos_finales, f, ensure_ascii=False, indent=4)
    
    print(f"¡ÉXITO! Los datos limpios están listos en: {ruta_salida}")

if __name__ == "__main__":
    procesar_documentos()