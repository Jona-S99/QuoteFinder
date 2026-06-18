# ---------------------------------------------------------------------------
# Router API REST con FastAPI
#  - Aqui defino endpoints relacionados con el retrieval
# --------------------------------------------------------------------------

# Librerias
from pydantic import BaseModel, Field

from fastapi import APIRouter, HTTPException, status
from starlette.concurrency import run_in_threadpool

from app.rag.retrieval import EMBEDDING_MODEL, TOP_K, retrieve_chunks

router = APIRouter(tags=["Retrieval"])


# ---------------------------------------------------------------------------
# 1. LLamar al retrieval
# ---------------------------------------------------------------------------

# Clase para la respuesta
class RetrievalQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1_000)
    embedding_model: str = Field(default=EMBEDDING_MODEL, min_length=1)
    top_k: int = Field(default=TOP_K, ge=1, le=50)


class RetrievedChunk(BaseModel):
    page_content: str
    metadata: dict


@router.post(
    "/retrieval/query",
    response_model=list[RetrievedChunk],
    status_code=status.HTTP_200_OK,
)
async def query_retrieval(payload: RetrievalQueryRequest):
    try:
        # Ejecutamos el retrieval
        docs = await run_in_threadpool(
            retrieve_chunks,
            payload.query,
            payload.embedding_model,
            payload.top_k,
        )

        # Retorno una lista
        return [
            RetrievedChunk(
                page_content=doc.page_content,
                metadata=doc.metadata,
            )
            for doc in docs
        ]

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error ejecutando retrieval: {str(e)}",
        )
