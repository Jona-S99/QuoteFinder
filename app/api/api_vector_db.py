# ---------------------------------------------------------------------------
# Router API REST con FastAPI
#  - Aqui defino endpoints relacionados con la gestión de la base de datos
#    vectorial. Esto incluye los procesos:
#    1. Subida de los documentos a data/raw
#    2. Conversión de los documentos a Markdown
#    3. Carga de entrevistas de ejemplo para pruebas rápidas de la UI
# ---------------------------------------------------------------------------

import shutil
import subprocess
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.config import ALLOWED_EXTENSIONS, MD_DIR, RAW_DIR
from app.rag.converter import convert_to_markdown
from app.rag.vector_db import build_vector_db

router = APIRouter()

BASE_DIR = Path(__file__).resolve().parents[1]


def _list_files(
    directory: Path, allowed_extensions: set[str] | None = None
) -> list[str]:
    if not directory.exists():
        return []

    files = [
        path.name
        for path in directory.iterdir()
        if path.is_file()
        and (allowed_extensions is None or path.suffix.lower() in allowed_extensions)
    ]
    return sorted(files)


# ---------------------------------------------------------------------------
# 1. Subir documentos PDF/DOCX/Markdown a app/data/raw
# Estos archivos serán almacenados en raw para luego convertirlos a Markdown
# ---------------------------------------------------------------------------


@router.post("/vector-db/upload")
async def upload_docs(
    files: Annotated[
        list[UploadFile], File(description="Archivos PDF, DOCX o Markdown a subir")
    ],
):
    """
    Endpoint para subir archivos PDF, DOCX o Markdown a la carpeta raw.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for file in files:
        if not file.filename:
            raise HTTPException(status_code=400, detail="Archivo sin nombre")

        if Path(file.filename).suffix.lower() not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Archivo {file.filename} no es PDF, DOCX ni Markdown",
            )

        file_path = RAW_DIR / file.filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

    return {
        "message": f"{len(files)} archivos subidos exitosamente",
        "raw_files": _list_files(RAW_DIR, ALLOWED_EXTENSIONS),
    }


# ---------------------------------------------------------------------------
# 2. Convertir documentos PDF/DOCX/Markdown a Markdown
# ---------------------------------------------------------------------------


@router.post("/vector-db/convert")
async def convert_docs_to_markdown():
    """
    Endpoint para convertir los archivos cargados en raw a Markdown.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    MD_DIR.mkdir(parents=True, exist_ok=True)

    input_files = [
        input_file
        for input_file in RAW_DIR.iterdir()
        if input_file.is_file() and input_file.suffix.lower() in ALLOWED_EXTENSIONS
    ]

    for input_file in input_files:
        if input_file.suffix.lower() == ".md":
            destination = MD_DIR / input_file.name
            destination.write_text(
                input_file.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            continue

        convert_to_markdown(
            file_path=input_file,
            output_path=MD_DIR / input_file.with_suffix(".md").name,
        )

    return {
        "message": "Entrevistas convertidas a Markdown exitosamente",
        "converted_count": len(input_files),
        "markdown_files": _list_files(MD_DIR, {".md"}),
    }


# ---------------------------------------------------------------------------
# 3. Elegir el modelo de embeddings para la base vectorial
# ---------------------------------------------------------------------------

# Funcion para listar los modelos de ollama que tiene instalado el usuario
def list_ollama_models() -> list[str]:
    try:
        # Ejecuto el comando en la terminal
        ls_models_or = subprocess.run(
            ["ollama", "list"],
            capture_output = True, # no muestra la salida en la terminal
            text=True, # obligo a que venga como texto, y no en bytes
            check=True # guardo el error por si falla
        )

        # Quito la primera fila, porque es el header de la lista
        ls_models_proc = ls_models_or.stdout.strip().splitlines()
        ls_models_proc = ls_models_proc[1:]

        # Me quedo solo con el nombre del modelo
        ls_models_proc = [parts[0] for model in ls_models_proc if (parts := model.split())]

        return ls_models_proc
        
    except FileNotFoundError:
        raise RuntimeError("Ollama no está instalado o no está disponible en PATH")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error ejecutando ollama list: {e.stderr}")


# Modelo pydantic para la respuesta de la API
class EmbeddingModelsResponse(BaseModel):
    models: list[str]


# Ruta para obtener el listado de modelos
@router.get("/vector-db/embedding-models", response_model=EmbeddingModelsResponse)
async def get_embedding_models() -> EmbeddingModelsResponse:
    try:
        ls_models = list_ollama_models()
        return EmbeddingModelsResponse(models=ls_models)
    
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# 4. Construir base de datos vectorial
# ---------------------------------------------------------------------------

# Modelo pydantic para la entrada a la api: lo que me pide la query
# - En este caso, el modelo en el body, de lo contrario, build_vector_db
#   usara el modelo por defecto que puse en app/rag/vector_db.py ("bge-m3:latest")
class BuildVectorDBRequest(BaseModel):
    embedding_model: str


# Modelo pydantic para la respuesta de la API
class BuildVectorDBResponse(BaseModel):
    message: str
    documents_count: int
    chunks_count: int
    table_name: str
    uri: str
    embedding_model: str


@router.post("/vector-db/build", response_model=BuildVectorDBResponse)
async def build_vector_database(payload: BuildVectorDBRequest):
    """Endpoint para construir la base de datos vectorial a partir de los archivos Markdown en data/md."""
    try:
        # Aplico la funcion para crear la vector db, enviadole el modelo de embedding
        # como parametro
        result = build_vector_db(
            embedding_model=payload.embedding_model,
        )

        return BuildVectorDBResponse(
            message="Base vectorial creada exitosamente",
            documents_count=result["documents_count"],
            chunks_count=result["chunks_count"],
            table_name=result["table_name"],
            uri=result["uri"],
            embedding_model=result["embedding_model"]
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error creando la base vectorial: {str(e)}"
        )


# ------------------------------------------------------------------------------ #
# 5. Limpieza de la base de datos vectorial
# Este endpoint borra todos los archivos dentro de data, reseteando el estado
# para nuevas entrevistas.
# ------------------------------------------------------------------------------ #


def clear_directory(directory: str | Path) -> None:
    """ "
    Función para limpiar el directorio indicado.
    Fue construida para resetear las carpetas dentro de `data`
    """
    directory = Path(directory)

    if not directory.exists():
        return

    for item in directory.iterdir():
        if item.name == ".gitkeep":
            continue
        if item.is_file() or item.is_symlink():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)


@router.post("/vector-db/clean")
async def clean_vector_db():
    """
    Endpoint para limpiar el directorio `data`, eliminando todos los archivos
    relacionados con la base de datos vectorial.\n
    Esto resetea el estado para nuevas entrevistas.
    """
    try:
        clear_directory(directory="app/data")
        return {"message": "Directorio 'data' limpiado exitosamente"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
