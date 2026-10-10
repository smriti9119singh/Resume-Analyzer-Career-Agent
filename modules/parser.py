"""
PDF and plain-text resume parser with validation and section segmentation.
Uses PyMuPDF (fitz) when parsing binary PDF streams.
"""
import re
from typing import Dict, List
from .config import MAX_FILE_SIZE_BYTES, MIN_RESUME_CHARS, sanitize_untrusted_text


def validate_and_extract_pdf_bytes(pdf_bytes: bytes, filename: str = "resume.pdf") -> Dict:
    """
    Validates PDF file size, magic header (%PDF-), and extracts text via PyMuPDF (fitz).
    """
    if not pdf_bytes or len(pdf_bytes) == 0:
        raise ValueError("Uploaded PDF file is empty (0 bytes).")

    if len(pdf_bytes) > MAX_FILE_SIZE_BYTES:
        raise ValueError(f"File '{filename}' exceeds the 5 MB maximum size limit.")

    if not pdf_bytes[:8].startswith(b"%PDF-"):
        raise ValueError(f"Invalid file format for '{filename}'. Expected a valid %PDF- header.")

    try:
        import fitz  # PyMuPDF
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages_text = [page.get_text("text") for page in doc]
        raw_text = "\n".join(pages_text)
        page_count = len(doc)
        doc.close()
    except ImportError:
        raise RuntimeError("PyMuPDF (fitz) is required for binary PDF parsing. Install via `pip install PyMuPDF`.")
    except Exception as exc:
        raise ValueError(f"Failed to read PDF '{filename}': {exc}") from exc

    return parse_resume_text(raw_text, page_count=page_count)


def parse_resume_text(raw_text: str, page_count: int = 1) -> Dict:
    """
    Sanitizes resume text and segments it into structured sections:
    summary, education, experience, projects, certifications, and skills.
    """
    sanitized, security_flags = sanitize_untrusted_text(raw_text)
    if len(sanitized) < MIN_RESUME_CHARS:
        raise ValueError("Extracted resume text is empty or shorter than minimum threshold (40 chars).")

    lines = [line.strip() for line in sanitized.splitlines() if line.strip()]
    sections: Dict[str, List[str]] = {
        "summary": [],
        "education": [],
        "experience": [],
        "projects": [],
        "certifications": [],
        "skills": [],
    }

    heading_map = {
        "summary": re.compile(r"^(professional\s+)?(summary|profile|objective)$", re.I),
        "education": re.compile(r"^(education|academic\s+background)$", re.I),
        "experience": re.compile(r"^(work\s+|professional\s+)?(experience|employment|internships?)$", re.I),
        "projects": re.compile(r"^(academic\s+|personal\s+|technical\s+)?projects$", re.I),
        "certifications": re.compile(r"^(certifications?|licenses?\s*&?\s*certifications?)$", re.I),
        "skills": re.compile(r"^(technical\s+|core\s+)?(skills|technologies)$", re.I),
    }

    current_section = "header"
    for line in lines[1:]:
        clean_header = re.sub(r"[:\-—|]", "", line).strip()
        matched_key = None
        if len(clean_header) <= 40:
            for key, regex in heading_map.items():
                if regex.match(clean_header):
                    matched_key = key
                    break
        if matched_key:
            current_section = matched_key
            continue

        if current_section in sections:
            sections[current_section].append(line)

    return {
        "candidate_name": lines[0] if lines else "Candidate",
        "sanitized_text": sanitized,
        "word_count": len(sanitized.split()),
        "page_count": page_count,
        "sections": sections,
        "security_flags": security_flags,
    }
