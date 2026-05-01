"""RAG utilities for PDF ingestion and retrieval with ChromaDB."""

from pathlib import Path
from typing import List, Tuple

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.utils.logger import get_logger

logger = get_logger(__name__)


def _ensure_directory(path: Path) -> None:
    """Create a directory if it does not exist.

    Args:
        path: Directory path to create.
    """
    path.mkdir(parents=True, exist_ok=True)


def ingest_pdf_to_chroma(
    pdf_path: str,
    api_key: str,
    persist_dir: str = "chroma_store",
    collection_name: str = "finance_docs",
) -> int:
    """Load a PDF, chunk it, embed chunks, and persist in ChromaDB.

    Args:
        pdf_path: Path to uploaded PDF file.
        api_key: Gemini API key for embeddings.
        persist_dir: Local persistence directory for ChromaDB.
        collection_name: Chroma collection name.

    Returns:
        Number of chunks added.
    """
    logger.info("Ingesting PDF into Chroma: %s", pdf_path)
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = splitter.split_documents(pages)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key,
    )

    persist_path = Path(persist_dir)
    _ensure_directory(persist_path)

    vectordb = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_path),
    )
    vectordb.add_documents(docs)
    logger.info("PDF ingestion complete. Chunks stored: %s", len(docs))
    return len(docs)


def retrieve_context(
    query: str,
    api_key: str,
    persist_dir: str = "chroma_store",
    collection_name: str = "finance_docs",
    k: int = 3,
) -> Tuple[str, List[Document]]:
    """Retrieve top-k relevant chunks from local ChromaDB.

    Args:
        query: User query.
        api_key: Gemini API key for embeddings.
        persist_dir: Chroma persistence directory.
        collection_name: Chroma collection name.
        k: Number of top chunks to retrieve.

    Returns:
        Tuple with joined context text and raw retrieved documents.
    """
    logger.info("Retrieving context for query: %s", query)
    persist_path = Path(persist_dir)
    if not persist_path.exists():
        logger.warning("Chroma persistence directory does not exist: %s", persist_dir)
        return "", []

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key,
    )

    vectordb = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_path),
    )
    docs = vectordb.similarity_search(query, k=k)

    if not docs:
        logger.warning("No retrieval results found")
        return "", []

    joined_context = "\n\n".join(doc.page_content for doc in docs)
    logger.info("Retrieved %s context chunks", len(docs))
    return joined_context, docs
