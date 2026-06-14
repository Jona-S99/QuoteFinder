# ---------------------------------------------------------------------------
# Retrieval de chunks relevantes:
# Aqui se cargara la base de datos vectorial creada en el script vector_db.py
# y se creara el retriever a partir de esta base de datos vectorial.
# ---------------------------------------------------------------------------

# Librerias
from langchain_community.vectorstores import LanceDB
from langchain_ollama import OllamaEmbeddings

from app.config import TOP_K


def config_retriever(top_k: int = TOP_K):
    """Configura el retriever con el número de chunks a recuperar."""
    # Instanciamos el modelo de embeddings de Ollama

    embeddings = OllamaEmbeddings(
        model="bge-m3:latest",
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
        search_kwargs={"k": TOP_K},  # número de chunks a recuperar
    )

    return retriever


# Creo una funcion para recuperar los chunks relevantes a partir de una consulta
def retrieve_chunks(query: str) -> list[dict]:
    """
    Recupera los chunks relevantes a partir de una consulta.

    Args:
        query (str): La consulta del usuario.

    Returns:
        list[dict]: Una lista de diccionarios con los chunks relevantes.
    """
    # Configuramos el retriever
    retriever = config_retriever()

    # Utilizamos el retriever para recuperar los chunks relevantes
    relevant_chunks = retriever.invoke(query)

    # Devolvemos los chunks relevantes como una lista de diccionarios
    return relevant_chunks
