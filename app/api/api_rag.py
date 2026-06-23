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

from app.rag.graph import LLM_MODEL, State, graph_compiled
from app.rag.rerank import TOP_RANKED
from app.rag.retrieval import EMBEDDING_MODEL, RETRIEVAL_TOP_K

router = APIRouter(tags=["RAG"])


# ---------------------------------------------------------------------------- #
# Clases para request y response del endpoint
# ---------------------------------------------------------------------------- #

# Clase para el input del usuario
class RagQueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=1_000,
        description="Pregunta o tema para buscar citas en las entrevistas.",
    )
    embedding_model: str = Field(
        default=EMBEDDING_MODEL,
        min_length=1,
        description="Modelo de embeddings para hacer retrieval en LanceDB.",
    )
    top_k: int = Field(
        default=RETRIEVAL_TOP_K,
        ge=1,
        le=100,
        description="Cantidad de chunks a recuperar antes del reranking.",
    )
    top_ranked: int = Field(
        default=TOP_RANKED,
        ge=1,
        le=100,
        description="Cantidad de chunks a conservar después del reranking.",
    )
    llm_model: str = Field(
        default=LLM_MODEL,
        min_length=1,
        description="Modelo LLM local que generará la respuesta final.",
    )


# Clase para la respuesta del modelo
class RagQueryResponse(BaseModel):
    answer: str
    retrieved_chunks_count: int
    reranked_chunks_count: int


# ---------------------------------------------------------------------------- #
# Funcion para ejecutar el grafo RAG a partir de la consulta del usuario
# ---------------------------------------------------------------------------- #

def run_rag_graph(payload: RagQueryRequest) -> State:
    initial_state: State = {
        "query": payload.query,
        "embedding_model": payload.embedding_model,
        "top_k": payload.top_k,
        "top_ranked": payload.top_ranked,
        "llm_model": payload.llm_model,
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
        final_state = await run_in_threadpool(run_rag_graph, payload)

        return RagQueryResponse(
            answer=final_state["final_answer"],
            retrieved_chunks_count=len(final_state["retrieved_chunks"]),
            reranked_chunks_count=len(final_state["reranked_chunks"]),
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error ejecutando el grafo RAG: {str(e)}",
        )