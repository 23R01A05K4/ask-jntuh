# Ask JNTUH — Academic Regulations Q&A (RAG)

Ask questions about **JNTUH B.Tech (R22) academic regulations** in plain English and get answers with citations to the exact official document and page — promotion rules, detention, attendance, credits, CBT exams, grace marks.

**Live demo:** https://ask-jntuh.onrender.com

## The problem
Before every exam and result season, students dig through 100+ page regulation PDFs and scattered circulars to answer simple questions: *How many credits do I need for promotion? What's the detention rule? Is 65% attendance enough?* This project makes those documents conversational.

## How it works
1. **Ingestion** (`ingest.py`): official JNTUH PDFs are parsed (pdftotext + OCR for scanned proceedings) into ~95 overlapping chunks with document/page metadata, stored in `chunks.json`.
2. **Retrieval** (`app/rag.py`): a BM25 ranker (pure Python, no heavy dependencies) with light stemming and an academic-domain synonym map (detention ↔ detained, attendance ↔ attend, …) finds the most relevant chunks.
3. **Answer composition**: IDF-weighted sentence selection picks the most relevant sentences — so distinctive terms like *attendance* outweigh common ones like *required* — and every point is cited with its source document and page. If a free-tier LLM key (`GROQ_API_KEY`) is present, answers switch to fully generative mode with the same cited sources.
4. **Serving** (`app/main.py`): FastAPI backend + a clean chat UI (`app/static/index.html`).

## Data sources (public, official)
- [JNTUH R22 B.Tech Academic Regulations (Revised)](https://jntuh.ac.in/uploads/academics/R22B.Tech.RevisedAcademicRegulations2.pdf)
- [JNTUH R22 FAQs — Computer Based Test](https://jntuh.ac.in/uploads/academics/R22FAQs.pdf)
- [JNTUH Proceedings — Relaxation of Promotion Criteria (UG)](https://jntuh.ac.in/uploads/bulletins/RelaxationofPromotionCriteriaforUGStudentsofcertainregulations.pdf)

## Tech stack
Python · FastAPI · BM25 retrieval (pure stdlib) · vanilla JS chat UI · deployed free on Render · zero-cost stack, no paid APIs required

## Run locally
```bash
pip install -r requirements.txt
uvicorn main:app --reload
# open http://localhost:8000
```

To regenerate the index from the source documents:
```bash
python3 ingest.py   # needs pdftotext + tesseract
```

*Final-year CSE project by Ashritha Akula. Not affiliated with JNTUH — always confirm critical decisions with your college exam cell.*
