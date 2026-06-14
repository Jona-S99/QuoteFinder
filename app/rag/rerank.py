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

from app.config import CACHE_MODEL, RERANK_MODEL, TOP_RANKED


# Funcion principal para reranker
def rerank_chunks(query: str, docs: list[dict]) -> list[dict]:
    """
    Rerankea los chunks recuperados a partir de la consulta del usuario.

    Args:
        query (str): La consulta del usuario.
        docs (list[dict]): Una lista de diccionarios con los chunks recuperados.

    Returns:
        list[dict]: Una lista de diccionarios con los chunks rerankeados.
    """
    # Instancio el reranker
    ranker = Ranker(
        model_name=RERANK_MODEL,
        cache_dir=CACHE_MODEL,
        max_length=128,
    )

    # Preparo los passages para el reranking, asignando un id a cada chunk y manteniendo la metadata
    passages = [
        {
            "id": i,
            "text": d.page_content,
            "meta": d.metadata,
        }
        for i, d in enumerate(docs)
    ]

    # Creamos el request de reranking y obtenemos los documentos rerankeados
    rerank_request = RerankRequest(query=query, passages=passages)
    ranked = ranker.rerank(rerank_request)

    # Reconstruir docs en el nuevo orden con los mejores documentos
    docs_by_id = {i: d for i, d in enumerate(docs)}
    reranked_docs = [docs_by_id[item["id"]] for item in ranked[:TOP_RANKED]]
    return reranked_docs
