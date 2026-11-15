#!/usr/bin/env python3
"""Ingest JNTUH source PDFs into chunks.json (committed; the demo never re-parses PDFs)."""
import json, re, subprocess, sys
from pathlib import Path

DATA = Path(__file__).parent / "data"
SOURCES = [
    {"file": "R22_BTech_Academic_Regulations.pdf", "title": "JNTUH R22 B.Tech Academic Regulations (Revised)", "url": "https://jntuh.ac.in/uploads/academics/R22B.Tech.RevisedAcademicRegulations2.pdf"},
    {"file": "R22_FAQs.pdf", "title": "JNTUH R22 FAQs - Computer Based Test (CBT)", "url": "https://jntuh.ac.in/uploads/academics/R22FAQs.pdf"},
    {"file": "Promotion_Relaxation_UG.txt", "title": "JNTUH Proceedings - Relaxation of Promotion Criteria (UG)", "url": "https://jntuh.ac.in/uploads/bulletins/RelaxationofPromotionCriteriaforUGStudentsofcertainregulations.pdf"},
]

def pdf_pages(path):
    out = subprocess.run(["pdftotext", "-layout", str(path), "-"], capture_output=True, text=True).stdout
    return out.split("\f")

def txt_pages(path):
    return [path.read_text(encoding="utf-8", errors="ignore")]

def chunk(text, target=1100, overlap=150):
    text = text.replace("\r", "")
    text = re.sub(r"[ \t]+", " ", text)
    # join wrapped lines inside paragraphs: single newline -> space
    text = re.sub(r"\n(?=\S)", " ", text)
    text = re.sub(r"\n+", "\n", text).strip()
    if not text:
        return []
    paras = [p.strip() for p in re.split(r"\n+", text) if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        if len(cur) + len(p) > target and cur:
            chunks.append(cur.strip())
            cur = cur[-overlap:] + "\n" + p
        else:
            cur = (cur + "\n" + p) if cur else p
    if cur.strip():
        chunks.append(cur.strip())
    return chunks

def main():
    all_chunks, cid = [], 0
    for src in SOURCES:
        path = DATA / src["file"]
        pages = txt_pages(path) if path.suffix == ".txt" else pdf_pages(path)
        for pno, page in enumerate(pages, 1):
            for c in chunk(page):
                if len(c) < 80:
                    continue
                cid += 1
                all_chunks.append({"id": cid, "doc": src["title"], "url": src["url"], "page": pno, "text": c})
    Path("chunks.json").write_text(json.dumps(all_chunks, ensure_ascii=False, indent=1))
    print(f"{len(all_chunks)} chunks from {len(SOURCES)} sources")

if __name__ == "__main__":
    main()
