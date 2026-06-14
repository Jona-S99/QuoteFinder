"""Chunking recursivo de Markdown normalizado para RAG."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter


CHUNK_SIZE = 900
CHUNK_OVERLAP = 120
SEPARATORS = [
    "\n## ",
    "\n### ",
    "\nPregunta ",
    "\nEntrevistador:",
    "\nEntrevistado:",
    "\n\n",
    ". ",
    " ",
]

_HEADER_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


@dataclass
class ChunkResult:
    """Resultado de chunking con texto y metadata enriquecida."""

    text: str
    metadata: dict[str, Any]


def chunk_markdown(markdown: str, metadata: dict[str, Any]) -> list[ChunkResult]:
    """
    Divide Markdown normalizado en chunks usando RecursiveCharacterTextSplitter.

    Args:
        markdown: Markdown limpio proveniente de Docling y normalizacion previa.
        metadata: Metadata base del documento. Se copia en cada chunk.

    Returns:
        Lista de ChunkResult con metadata enriquecida por chunk.
    """
    if not markdown or not markdown.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
        keep_separator=True,
    )
    chunk_texts = [chunk.strip() for chunk in splitter.split_text(markdown)]
    chunk_texts = [chunk for chunk in chunk_texts if chunk]
    chunk_count = len(chunk_texts)
    sections = _extract_sections(markdown)

    results: list[ChunkResult] = []
    search_from = 0

    for chunk_index, text in enumerate(chunk_texts):
        chunk_start = _find_chunk_start(markdown, text, search_from)
        if chunk_start is not None:
            search_from = chunk_start + max(len(text) - CHUNK_OVERLAP, 1)

        enriched_metadata = {
            **metadata,
            "chunk_index": chunk_index,
            "chunk_count": chunk_count,
            "section": _section_for_position(sections, chunk_start),
        }
        results.append(ChunkResult(text=text, metadata=enriched_metadata))

    return results


def _extract_sections(markdown: str) -> list[tuple[int, str]]:
    """Extrae encabezados Markdown con su posicion en el texto."""
    return [
        (match.start(), match.group(2).strip())
        for match in _HEADER_RE.finditer(markdown)
    ]


def _find_chunk_start(markdown: str, chunk: str, search_from: int) -> int | None:
    """
    Ubica aproximadamente el inicio de un chunk dentro del Markdown original.

    LangChain puede conservar separadores o limpiar bordes, por eso se busca
    primero el chunk completo y luego un prefijo suficientemente distintivo.
    """
    full_match = markdown.find(chunk, max(search_from, 0))
    if full_match != -1:
        return full_match

    prefix = chunk[: min(len(chunk), 120)].strip()
    if not prefix:
        return None

    prefix_match = markdown.find(prefix, max(search_from, 0))
    if prefix_match != -1:
        return prefix_match

    fallback_match = markdown.find(prefix)
    if fallback_match != -1:
        return fallback_match

    return None


def _section_for_position(
    sections: list[tuple[int, str]],
    position: int | None,
) -> str | None:
    """Devuelve el encabezado Markdown mas cercano anterior a una posicion."""
    if position is None or not sections:
        return None

    current_section: str | None = None
    for section_position, section_name in sections:
        if section_position > position:
            break
        current_section = section_name

    return current_section
