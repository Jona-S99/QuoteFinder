# ---------------------------------------------------------------------------
# Retrieval de chunks relevantes:
# Aqui se cargara la base de datos vectorial creada en el script vector_db.py
# y se creara el retriever a partir de esta base de datos vectorial.
# ---------------------------------------------------------------------------

# Librerias
import json
from pathlib import Path

from langchain_community.vectorstores import LanceDB
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings


# ------------------------------------------------------------------------------ #
#  Configuraciones generales
# ------------------------------------------------------------------------------ #

PARENT_STORE_PATH = Path("app/data/parent_store.json")
EMBEDDING_MODEL = "bge-m3:latest"
TOP_K = 10


# ------------------------------------------------------------------------------ #
#  Cargar parents chunks
# ------------------------------------------------------------------------------ #

def load_parent_store(path: Path = PARENT_STORE_PATH) -> dict:
    """Funcion para cargar los parent chunks"""
    # Carga los chunk parents guardados durante la construccion de la vector DB
    return json.loads(path.read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------ #
#  Retrieval
# ------------------------------------------------------------------------------ #

def config_retriever(
    embedding_model: str = EMBEDDING_MODEL, 
    top_k: int = TOP_K,
):
    """Configurar el retriever con el número de chunks a recuperar."""
    # Instanciamos el modelo de embeddings de Ollama
    embeddings = OllamaEmbeddings(
        model=embedding_model,
    )

    # Cargamos la BD existente
    vector_db = LanceDB(
        uri="app/data/lancedb",
        embedding=embeddings,
        table_name="entrevistas",
    )

    # Creamos el retriever
    retriever = vector_db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},  # número de chunks a recuperar
    )

    return retriever


# Creo una funcion para recuperar los chunks relevantes a partir de una consulta
def retrieve_chunks(
    query: str,
    embedding_model: str = EMBEDDING_MODEL,
    top_k: int = TOP_K,
) -> list[Document]:
    """
    Recupera parents relevantes a partir de una consulta.
    Primero busca child chunks en LanceDB y luego expande cada child a su parent.
    """
    # Aplico la configuracion del retriever
    retriever = config_retriever(
        embedding_model=embedding_model,
        top_k=top_k,
    )

    # Recupera children relevantes desde LanceDB
    child_hits = retriever.invoke(query)

    # Carga los parents guardados en JSON
    parent_store = load_parent_store()

    # Guarda los parents finales que se entregaran al reranker/LLM
    parent_docs: list[Document] = []

    # Evita devolver el mismo parent mas de una vez
    seen_parent_ids = set()

    for child in child_hits:
        # Lee el id del parent asociado al child encontrado
        parent_id = child.metadata.get("parent_id")

        # Si por algun motivo el child no tiene parent_id, lo saltamos
        if not parent_id:
            continue

        # Si ya agregamos este parent, no lo repetimos
        if parent_id in seen_parent_ids:
            continue

        # Busca el parent en el parent_store
        parent = parent_store.get(parent_id)

        # Si el parent_id no existe en el JSON, lo saltamos
        if not parent:
            continue
        
        seen_parent_ids.add(parent_id)

        # Devuelve el texto grande del parent, pero conserva info del child que hizo match
        parent_docs.append(
            Document(
                page_content=parent["text"],
                metadata={
                    **parent["metadata"],
                    "matched_child_text": child.page_content,
                    "matched_child_metadata": child.metadata,
                },
            )
        )

    return parent_docs
