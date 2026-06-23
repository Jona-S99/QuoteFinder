# ---------------------------------------------------------------------------
# # Reranking de chunks recuperados:
# - Aqui implemento el reranking de los chunks recuperados
#   a partir de la consulta del usuario.
# - Para esto utilizare el modelo de reranking FlashRank, que
#   es un modelo ligero y eficiente para tareas de reranking.
# - El modelo se cacheara localmente para evitar tener que
#   descargarlo cada vez que se ejecute la app.
# ---------------------------------------------------------------------------


# Librerias
from flashrank import Ranker, RerankRequest


# ------------------------------------------------------------------------------ #
#  Configuraciones generales
# ------------------------------------------------------------------------------ #
RERANK_MODEL = "ms-marco-TinyBERT-L-2-v2"  # modelo de reranking a utilizar
CACHE_MODEL = "app/models/flashrank"  # ruta para cachear el modelo de reranking
TOP_RANKED = 5  # numero de chunks a mantener despues del reranking


# ------------------------------------------------------------------------------ #
#  1. Reranker
# ------------------------------------------------------------------------------ #

def rerank_chunks(query: str, docs: list[dict], top_ranked: int) -> list[dict]:
    """
    Rerankea los chunks recuperados a partir de la consulta del usuario.

    Args:
        query (str): La consulta del usuario.
        docs (list[dict]): Una lista de diccionarios con los chunks recuperados.
            En este caso, utiliza los chiledren chunks para la comparación del
            chunk y la pregunta del usuario

    Returns:
        list[dict]: Una lista de diccionarios con los chunks rerankeados.
    """
    # Instancio el reranker
    ranker = Ranker(
        model_name=RERANK_MODEL,
        cache_dir=CACHE_MODEL,
        max_length=128,
    )

    # Preparo los passages para el reranking, asignando un id a cada chunk y 
    # y asignando el texto del children chunk para la comparacion
    passages = [
        {
            "id": i,
            "text": d.metadata.get("matched_child_text", d.page_content)
        }
        for i, d in enumerate(docs)
    ]

    # Creamos el request de reranking y obtenemos los documentos rerankeados
    rerank_request = RerankRequest(query=query, passages=passages)
    ranked = ranker.rerank(rerank_request)

    # Reconstruir docs en el nuevo orden con los mejores documentos
    docs_by_id = {i: d for i, d in enumerate(docs)}
    reranked_docs = [docs_by_id[item["id"]] for item in ranked[:top_ranked]]
    
    return reranked_docs
