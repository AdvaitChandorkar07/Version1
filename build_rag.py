from rag.ingest import build_rag_index

build_rag_index([
    "knowledge/appointment_notes.txt",
    "knowledge/medications.txt",
    "knowledge/faq.txt",
])