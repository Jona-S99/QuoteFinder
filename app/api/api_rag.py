# ---------------------------------------------------------------------------- #
# # Endpoint para RAG:
# - Aquí se define el endpoint para recibir las consultas del usuario y
#   ejecutar el grafo RAG.
# - El endpoint es del tipo POST y recibe un JSON con la consulta del usuario.
# - La respuesta del endpoint incluye la respuesta final generada por el nodo
#   Response, así como información sobre la cantidad de chunks recuperados y
#   rerankeados.
# ---------------------------------------------------------------------------- #

# Librerias
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.rag.graph import graph_compiled, State

router = APIRouter(tags=["RAG"])


# ---------------------------------------------------------------------------- #
# Clases para request y response del endpoint
# ---------------------------------------------------------------------------- #
class RagQueryRequest(BaseModel):
    # Query del usuario. Se limita para evitar requests absurdamente grandes.
    query: str = Field(
        ...,
        min_length=1,
        max_length=1_000,
        description="Pregunta o tema para buscar citas en las entrevistas.",
    )


class RagQueryResponse(BaseModel):
    # Respuesta final que produjo el nodo Response.
    answer: str

    # Cantidad de chunks encontrados por retrieval.
    retrieved_chunks_count: int

    # Cantidad de chunks que quedaron después del reranking.
    reranked_chunks_count: int


# ---------------------------------------------------------------------------- #
# Funcion para ejecutar el grafo RAG a partir de la consulta del usuario
# ---------------------------------------------------------------------------- #


def run_rag_graph(query: str) -> State:
    """
    Función de servicio mínima.

    Mantiene el endpoint delgado:
    el endpoint valida HTTP, y esta función sabe cómo llamar al grafo.
    """
    initial_state: State = {
        "query": query,
        "retrieved_chunks": [],
        "reranked_chunks": [],
        "final_answer": "",
    }

    return graph_compiled.invoke(initial_state)


# ---------------------------------------------------------------------------- #
# Endpoint para ejecutar el grafo RAG
# ---------------------------------------------------------------------------- #


@router.post(
    "/rag/ask",
    response_model=RagQueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Buscar citas usando el grafo RAG",
)
async def ask_rag(payload: RagQueryRequest) -> RagQueryResponse:
    try:
        # Como graph_compiled.invoke es sincrono, lo movemos a un threadpool.
        # Así FastAPI no queda bloqueado mientras corre retrieval, rerank y LLM.
        final_state = await run_in_threadpool(run_rag_graph, payload.query)

        return RagQueryResponse(
            answer=final_state["final_answer"],
            retrieved_chunks_count=len(final_state["retrieved_chunks"]),
            reranked_chunks_count=len(final_state["reranked_chunks"]),
        )

    except ValueError as e:
        # Util si retrieval/rerank lanza errores esperados de validacion.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    except Exception as e:
        # Error inesperado del pipeline.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error ejecutando el grafo RAG: {str(e)}",
        )
