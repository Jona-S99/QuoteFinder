# Script para crear la base de datos vectorial a partir de los archivos markdown
# convertidos usando docling. Aqui se encuentra el proceso de chunking,
# embeddings y creacion de la base de datos vectorial. Esta base vectorial se
# hara con LanceDB y se guardara en el directorio de forma local.

# Librerias
import json
from pathlib import Path

from langchain_community.vectorstores import LanceDB
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ------------------------------------------------------------------------------ #
#  Configuraciones generales
# ------------------------------------------------------------------------------ #

MD_DIR = Path("app/data/md")
LANCEDB_DIR = "app/data/lancedb"
TABLE_NAME = "entrevistas"
EMBEDDING_MODEL = "bge-m3:latest"
PARENT_SIZE = 2500
PARENT_OVERLAP = 300
CHILD_SIZE = 600
CHILD_OVERLAP = 100
PARENT_STORE_PATH = Path("app/data/parent_store.json")

# ------------------------------------------------------------------------------ #
#  Chunking de documentos markdown
# ------------------------------------------------------------------------------ #


# Listo los archivos markdown
def load_markdown_documents(md_dir: Path = MD_DIR) -> list[tuple[str, str]]:
    files = [p for p in md_dir.iterdir() if p.is_file() and p.suffix == ".md"]
    return [(file.stem, file.read_text(encoding="utf-8")) for file in files]


# Funcion auxiliar para crear los chunks padres usando un recursive splitter
def split_into_parent_chunks(
    text: str,
    parent_size: int = PARENT_SIZE,
    parent_overlap: int = PARENT_OVERLAP,
) -> list[str]:
    # Esta función crea chunks grandes, llamados "parents"
    # Los parents no se vectorizan directamente: sirven como contexto amplio para el LLM

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_size,
        chunk_overlap=parent_overlap,
        # Estos separadores intentan respetar la estructura markdown antes de cortar por caracteres
        separators=[
            "\n# ",
            "\n## ",
            "\n### ",
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    # Devuelve una lista de textos grandes
    return parent_splitter.split_text(text)


# Funcion auxiliar para crear los chunks padres usando un recursive splitter
def split_into_child_chunks(
    text: str,
    child_size: int = CHILD_SIZE,
    child_overlap: int = CHILD_OVERLAP,
) -> list[str]:
    # Esta funcion crea chunks children
    # Los children si se guardan en LanceDB porque son mejores para busqueda semantica precisa

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_size,
        chunk_overlap=child_overlap,
        # Los children tambien respetan parrafos y frases cuando es posible
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    # Devuelve una lista de textos pequeños
    return child_splitter.split_text(text)


# Funcion principal para chunking
def split_markdown_documents_parent_child(
    documents: list[tuple[str, str]],
    parent_size: int = PARENT_SIZE,
    parent_overlap: int = PARENT_OVERLAP,
    child_size: int = CHILD_SIZE,
    child_overlap: int = CHILD_OVERLAP,
) -> tuple[list[Document], dict]:
    # child_chunks son los documentos pequeños que irán a LanceDB con embeddings
    child_chunks: list[Document] = []

    # parent_store guarda los textos grandes para recuperarlos después por parent_id
    parent_store: dict = {}

    for nombre, contenido in documents:
        # Primero divido el documento completo en bloques grandes
        parent_chunks = split_into_parent_chunks(
            contenido,
            parent_size=parent_size,
            parent_overlap=parent_overlap,
        )

        for parent_index, parent_text in enumerate(parent_chunks):
            # parent_id conecta cada child con su parent
            parent_id = f"{nombre}:parent:{parent_index}"

            # Guardo el parent completo fuera de la vector DB
            parent_store[parent_id] = {
                "text": parent_text,
                "metadata": {
                    "document_id": nombre,
                    "source_file": f"{nombre}.md",
                    "participant": nombre,
                    "parent_id": parent_id,
                    "parent_index": parent_index,
                },
            }

            # Divide cada parent en children pequeños
            child_texts = split_into_child_chunks(
                parent_text,
                child_size=child_size,
                child_overlap=child_overlap,
            )

            for child_index, child_text in enumerate(child_texts):
                # Cada child se convierte en un Document indexable en LanceDB
                child_chunks.append(
                    Document(
                        page_content=child_text,
                        metadata={
                            "document_id": nombre,
                            "source_file": f"{nombre}.md",
                            "participant": nombre,
                            "parent_id": parent_id,
                            "parent_index": parent_index,
                            "child_index": child_index,
                            "chunk_type": "child",
                        },
                    )
                )

    # Devuelve los children para embeddings y el parent_store para contexto
    return child_chunks, parent_store


# ------------------------------------------------------------------------------ #
#  Embeddings y creación de la base de datos vectorial
# ------------------------------------------------------------------------------ #

# Funcion auxiliar para guardar los parent chunks en formato json
def save_parent_store(parent_store: dict, path: Path = PARENT_STORE_PATH) -> None:
    # Creo el directorio si no existe
    path.parent.mkdir(parents=True, exist_ok=True)

    # Guarda los parents en disco para poder recuperarlos durante retrieval
    path.write_text(
        json.dumps(parent_store, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# Construyo la base de datos vectorial a partir de los documentos markdown
def build_vector_db(
    md_dir: Path = MD_DIR,
    uri: str = LANCEDB_DIR,
    table_name: str = TABLE_NAME,
    embedding_model: str = EMBEDDING_MODEL,
    parent_size: int = PARENT_SIZE,
    parent_overlap: int = PARENT_OVERLAP,
    child_size: int = CHILD_SIZE,
    child_overlap: int = CHILD_OVERLAP,
):
    """Función para construir la base de datos vectorial a partir de los documentos markdown."""
    # Cargo los documentos markdown desde el directorio "md"
    docs = load_markdown_documents(md_dir)

    # Verifico que haya documentos markdown para procesar
    if not docs:
        raise ValueError("No hay archivos Markdown para procesar")

    # Divido los documentos markdown en chunks
    child_chunks, parent_store = split_markdown_documents_parent_child(
        docs,
        parent_size=parent_size,
        parent_overlap=parent_overlap,
        child_size=child_size,
        child_overlap=child_overlap,
    )

    # Guardo los parent chunks para luego poder acceder a ellos
    save_parent_store(parent_store)

    # Instancio el modelo de embeddings
    embeddings = OllamaEmbeddings(model=embedding_model)

    # Creo la base de datos vectorial a partir de los chunks utilizando el modelo
    # de embeddings
    vector_db = LanceDB.from_documents(
        documents=child_chunks,
        embedding=embeddings,
        uri=uri,
        table_name=table_name,
    )

    return {
        "vector_db": vector_db,
        "embedding_model": embedding_model,
        "parent_size": parent_size,
        "parent_overlap": parent_overlap,
        "child_size": child_size,
        "child_overlap": child_overlap,
        "documents_count": len(docs),
        "chunks_count": len(child_chunks),
        "parents_count": len(parent_store),
        "table_name": table_name,
        "uri": uri,
    }
