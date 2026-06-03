import fitz
from docx import Document

from rag_chatbot.ingestion.loaders import load_document
from rag_chatbot.schemas import DocumentStatus


def test_txt_extraction_works(tmp_path) -> None:
    path = tmp_path / "sample.txt"
    path.write_text("Hola desde TXT", encoding="utf-8")

    document = load_document(path, "hash", "doc_txt")

    assert document.extraction_status == DocumentStatus.EXTRACTED
    assert document.requires_ocr is False
    assert document.page_count == 1
    assert document.pages[0].text == "Hola desde TXT"


def test_docx_extraction_works(tmp_path) -> None:
    path = tmp_path / "sample.docx"
    docx = Document()
    docx.add_paragraph("Hola desde DOCX")
    docx.save(path)

    document = load_document(path, "hash", "doc_docx")

    assert document.extraction_status == DocumentStatus.EXTRACTED
    assert document.requires_ocr is False
    assert document.page_count == 1
    assert "Hola desde DOCX" in document.pages[0].text


def test_pdf_extraction_works(tmp_path) -> None:
    path = tmp_path / "sample.pdf"
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "Hola desde PDF")
    pdf.save(path)
    pdf.close()

    document = load_document(path, "hash", "doc_pdf")

    assert document.extraction_status == DocumentStatus.EXTRACTED
    assert document.requires_ocr is False
    assert document.page_count == 1
    assert "Hola desde PDF" in document.pages[0].text


def test_pdf_without_text_requires_ocr(tmp_path) -> None:
    path = tmp_path / "blank.pdf"
    pdf = fitz.open()
    pdf.new_page()
    pdf.save(path)
    pdf.close()

    document = load_document(path, "hash", "doc_blank_pdf")

    assert document.extraction_status == DocumentStatus.EMPTY_TEXT
    assert document.requires_ocr is True
    assert document.page_count == 1
    assert document.char_count == 0
