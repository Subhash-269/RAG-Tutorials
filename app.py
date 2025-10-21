import os
from src.data_loader import load_all_documents
from src.vectorstore import FaissVectorStore
# from src.search import RAGSearch  # optional

FAISS_DIR = "faiss_store"
FAISS_FILE = "faiss.index"
FAISS_PATH = os.path.join(FAISS_DIR, FAISS_FILE)

if __name__ == "__main__":
    # 1) Load documents
    docs = load_all_documents("data")
    print(f"Loaded {len(docs)} documents.")

    # 2) Prepare vector store
    store = FaissVectorStore(FAISS_DIR)

    # 3) Build the FAISS index if it doesn't exist; otherwise load it
    if not os.path.exists(FAISS_PATH):
        print(f"[INFO] No FAISS index found at {FAISS_PATH}. Building a new index...")
        store.build_from_documents(docs)
        # Persist the index to disk — adjust the method name if your class differs.
        if hasattr(store, "save"):
            store.save()
        elif hasattr(store, "persist"):
            store.persist()
        else:
            print("[WARN] Vector store has no save/persist method; ensure build_from_documents writes the index.")
    else:
        print(f"[INFO] Found existing FAISS index at {FAISS_PATH}. Loading...")
        store.load()

    # 4) Sanity check: if load() is the required step post-build, do it here too.
    try:
        if hasattr(store, "load"):
            store.load()
    except Exception as e:
        print(f"[WARN] load() after build raised: {e}")

    # 5) Test a query
    # results = store.query("What is attention mechanism?", top_k=3)
    results = store.query("What should you do first if someone is injured in an auto accident?", top_k=3)
    print(results)

    # 6) Optional: RAG pipeline
    # rag_search = RAGSearch()
    # query = "What is attention mechanism?"
    # summary = rag_search.search_and_summarize(query, top_k=3)
    # print("Summary:", summary)
