# -------------------------------------------------- #
# Modulo de conversión de archivos
# Aqui utilizo Docling para convertir archivos PDF
# y DOCX a Markdown
# -------------------------------------------------- #

# Librerias
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption


def save_markdown(markdown: str, output_path: str | Path) -> None:
    """
    Funcion para guardar contenido Markdown en un archivo.
    """
    output_path = Path(output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown)


def build_converter(use_ocr: bool = False) -> DocumentConverter:
    """
    Crea y configura un convertidor de Docling para archivos PDF y DOCX.

    Si `use_ocr` es `True`, habilita OCR para extraer texto desde PDFs
    escaneados o basados en imágenes. La configuración personalizada solo
    se aplica al procesamiento de PDFs.
    """
    # Configura cómo debe procesarse el PDF durante la conversión.
    pdf_options = PdfPipelineOptions(
        do_ocr=use_ocr,
        do_table_structure=False,
    )

    if use_ocr:
        # Si el PDF es escaneado o contiene imágenes, activa OCR con RapidOCR.
        pdf_options.ocr_options = RapidOcrOptions()

    # Crea un converter que solo acepta PDF y DOCX.
    # Las opciones personalizadas se aplican únicamente a PDFs.
    return DocumentConverter(
        allowed_formats=[InputFormat.PDF, InputFormat.DOCX],
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options)},
    )


def convert_to_markdown(
    file_path: str | Path, output_path: str | Path = None, use_ocr: bool = False
):
    """
    Convierte un archivo PDF o DOCX a una cadena en formato Markdown.

    Recibe una ruta como `str` o `Path`, construye el convertidor con la
    configuración indicada y devuelve el contenido convertido como texto.
    """
    # Normaliza la ruta recibida para poder trabajar con str o Path.
    file_path = Path(file_path)

    # Construye el converter con o sin OCR según el tipo de documento esperado.
    converter = build_converter(use_ocr=use_ocr)

    # Ejecuta la conversión del archivo al modelo interno de Docling.
    result = converter.convert(str(file_path))

    # Exporta el documento convertido a texto en formato Markdown.
    if output_path is None:
        output_path = file_path.with_suffix(".md")
    save_markdown(result.document.export_to_markdown(compact_tables=True), output_path)


# Prueba de las funciones
# def main():
#     import os

#     documentos = os.listdir("./entrevistas-prueba/pdf")

#     mds = [convert_to_markdown(f"./entrevistas-prueba/pdf/{x}") for x in documentos]

#     documentos_md = [
#         f"./entrevistas-prueba/md/{doc.split('.')[0]}.md" for doc in documentos
#     ]

#     [save_markdown(md, output_md) for md, output_md in zip(mds, documentos_md)]
