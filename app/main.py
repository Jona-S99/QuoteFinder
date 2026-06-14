# ---------------------------------------------------------------------------
# API REST con FastAPI
# Script principal para levantar la API REST con FastAPI.
# También sirve la UI HTML/CSS para probar el pipeline de forma interactiva.
# ---------------------------------------------------------------------------

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.api_rag import router as rag_router
from app.api.api_vector_db import router as vector_db_router
from app.config import STATIC_DIR, TEMPLATES_DIR

app = FastAPI()
app.include_router(vector_db_router, prefix="/api/v1")
app.include_router(rag_router, prefix="/api/v1")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def read_root():
    return FileResponse(TEMPLATES_DIR / "index.html")
