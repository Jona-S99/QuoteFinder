# ---------------------------------------------------------------------------
# Script para almacenar variables globales de la app
# ---------------------------------------------------------------------------

from pathlib import Path

# ---------------------------------------------------------------------------
# main: Rutas de directorios fastapi
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"


# ---------------------------------------------------------------------------
# upload_vector_db
# ---------------------------------------------------------------------------

## Rutas de directorios para los datos
DATA_DIR = BASE_DIR / "data"  # directorio principa
RAW_DIR = DATA_DIR / "raw"  # directorio de entrevistas crudas
MD_DIR = DATA_DIR / "md"  # directorio de entrevistas convertidas a Markdown

## Extensoines de archivos permitidas para la conversion
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".md"}
