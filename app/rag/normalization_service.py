"""Normalizacion de Markdown para pipelines RAG."""

from __future__ import annotations

import re


_ROLE_LABEL_RE = re.compile(r"^(Pregunta|Respuesta|Entrevistador|Entrevistado):$")
_HEADER_RE = re.compile(r"^#{1,6}\s+\S")
_LIST_ITEM_RE = re.compile(r"^(\s*[-*+]\s+|\s*\d+\.\s+)")


def normalize_markdown(markdown: str) -> str:
    """
    Normaliza Markdown ya convertido desde PDF/DOCX para chunking posterior.

    La funcion conserva encabezados, listas y marcas de entrevista, pero limpia
    espacios repetidos, saltos excesivos y cortes de linea dentro de parrafos.
    No elimina contenido textual.
    """
    if not markdown:
        return ""

    text = markdown.replace("\r\n", "\n").replace("\r", "\n")
    lines = [_normalize_line(line) for line in text.split("\n")]
    raw_blocks = _group_lines_into_blocks(lines)
    blocks = [_normalize_block(block) for block in raw_blocks]
    blocks = [block for block in blocks if block]
    blocks = _merge_fragmented_paragraphs(blocks)
    blocks = _attach_role_labels(blocks)

    return "\n\n".join(blocks).strip()


def _normalize_line(line: str) -> str:
    """Limpia espacios de una linea sin cambiar su contenido semantico."""
    stripped = line.strip()
    if not stripped:
        return ""
    return re.sub(r"[ \t]+", " ", stripped)


def _group_lines_into_blocks(lines: list[str]) -> list[list[str]]:
    """Agrupa lineas separadas por una o mas lineas vacias."""
    blocks: list[list[str]] = []
    current: list[str] = []

    for line in lines:
        if line:
            current.append(line)
            continue

        if current:
            blocks.append(current)
            current = []

    if current:
        blocks.append(current)

    return blocks


def _normalize_block(lines: list[str]) -> str:
    """Normaliza un bloque manteniendo estructura Markdown relevante."""
    if not lines:
        return ""

    if any(_must_keep_line_break(line) for line in lines):
        return "\n".join(lines).strip()

    return " ".join(lines).strip()


def _must_keep_line_break(line: str) -> bool:
    """Indica si la linea representa estructura que conviene preservar."""
    return bool(
        _HEADER_RE.match(line)
        or _LIST_ITEM_RE.match(line)
        or _ROLE_LABEL_RE.match(line)
        or line.startswith("```")
    )


def _is_structural_block(block: str) -> bool:
    """Detecta bloques que no deben mezclarse con parrafos vecinos."""
    first_line = block.splitlines()[0] if block else ""
    return _must_keep_line_break(first_line)


def _merge_fragmented_paragraphs(blocks: list[str]) -> list[str]:
    """
    Une bloques de parrafo que parecen cortes artificiales del conversor.

    Si un bloque no estructural termina sin puntuacion fuerte, el siguiente
    bloque no estructural se considera continuacion del mismo parrafo.
    """
    merged: list[str] = []

    for block in blocks:
        if not merged:
            merged.append(block)
            continue

        previous = merged[-1]
        if (
            not _is_structural_block(previous)
            and not _is_structural_block(block)
            and _looks_like_continuation(previous, block)
        ):
            merged[-1] = f"{previous} {block}"
        else:
            merged.append(block)

    return merged


def _looks_like_continuation(previous: str, current: str) -> bool:
    """Heuristica conservadora para unir parrafos cortados."""
    if not previous or not current:
        return False
    if previous.endswith((".", "?", "!", ":", ";", '"', "'")):
        return False
    return True


def _attach_role_labels(blocks: list[str]) -> list[str]:
    """Une etiquetas como 'Pregunta:' o 'Respuesta:' con su texto inmediato."""
    attached: list[str] = []
    index = 0

    while index < len(blocks):
        block = blocks[index]
        next_index = index + 1

        if (
            _ROLE_LABEL_RE.match(block)
            and next_index < len(blocks)
            and not _is_structural_block(blocks[next_index])
        ):
            attached.append(f"{block}\n{blocks[next_index]}")
            index += 2
        else:
            attached.append(block)
            index += 1

    return attached
