# Script para crear la base de datos vectorial a partir de los archivos markdown
# convertidos usando docling. Aqui se encuentra el proceso de chunking,
# embeddings y creacion de la base de datos vectorial. Esta base vectorial se
# hara con LanceDB y se guardara en el directorio de forma local.

# Librerias
from pathlib import Path

from langchain_community.vectorstores import LanceDB
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from app.rag.chunking_service import chunk_markdown
from app.rag.normalization_service import normalize_markdown

# ------------------------------------------------------------------------------ #
#  Configuraciones generales
# ------------------------------------------------------------------------------ #

MD_DIR = Path("app/data/md")
LANCEDB_DIR = "app/data/lancedb"
TABLE_NAME = "entrevistas"
EMBEDDING_MODEL = "bge-m3:latest"

# ------------------------------------------------------------------------------ #
#  Chunking de documentos markdown
# ------------------------------------------------------------------------------ #


# Listo los archivos markdown
def load_markdown_documents(md_dir: Path = MD_DIR) -> list[tuple[str, str]]:
    files = [p for p in md_dir.iterdir() if p.is_file() and p.suffix == ".md"]
    return [(file.stem, file.read_text(encoding="utf-8")) for file in files]


def split_markdown_documents(documents: list[tuple[str, str]]) -> list[Document]:
    """
    Normaliza y divide documentos Markdown en chunks listos para embeddings.
    """
    chunks_flat: list[Document] = []

    for nombre, contenido in documents:
        normalized_markdown = normalize_markdown(contenido)
        chunk_results = chunk_markdown(
            normalized_markdown,
            metadata={
                "document_id": nombre,
                "source_file": f"{nombre}.md",
                "participant": nombre,
            },
        )
        chunks_flat.extend(
            Document(page_content=chunk.text, metadata=chunk.metadata)
            for chunk in chunk_results
        )

    return chunks_flat


# ------------------------------------------------------------------------------ #
#  Embeddings y creación de la base de datos vectorial
# ------------------------------------------------------------------------------ #


# Construyo la base de datos vectorial a partir de los documentos markdown
def build_vector_db(
    md_dir: Path = MD_DIR,
    uri: str = LANCEDB_DIR,
    table_name: str = TABLE_NAME,
    embedding_model: str = EMBEDDING_MODEL,
):
    """Función para construir la base de datos vectorial a partir de los documentos markdown."""
    # Cargo los documentos markdown desde el directorio "md"
    docs = load_markdown_documents(md_dir)

    # Verifico que haya documentos markdown para procesar
    if not docs:
        raise ValueError("No hay archivos Markdown para procesar")

    # Divido los documentos markdown en chunks
    chunks = split_markdown_documents(docs)

    # Instancio el modelo de embeddings
    embeddings = OllamaEmbeddings(model=embedding_model)

    # Creo la base de datos vectorial a partir de los chunks utilizando el modelo
    # de embeddings
    vector_db = LanceDB.from_documents(
        documents=chunks,
        embedding=embeddings,
        uri=uri,
        table_name=table_name,
    )

    return {
        "vector_db": vector_db,
        "embedding_model": embedding_model,
        "documents_count": len(docs),
        "chunks_count": len(chunks),
        "table_name": table_name,
        "uri": uri,
    }
