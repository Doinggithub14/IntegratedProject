"""Streamlit UI for the Autonomous Finance Tutor."""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import streamlit as st
from google import genai
from tenacity import retry, stop_after_attempt, wait_fixed

from src.agents.planner import generate_roadmap
from src.agents.tutor import generate_tutor_response
from src.tools.retriever import ingest_pdf_to_chroma, retrieve_context
from src.utils.logger import get_logger

logger = get_logger(__name__)


@retry(stop=stop_after_attempt(2), wait=wait_fixed(1), reraise=True)
def get_gemini_baseline_answer(question: str, api_key: str) -> str:
    """Generate a simple Gemini response for the baseline step.

    Args:
        question: User question.
        api_key: Gemini API key.

    Returns:
        Model answer text.
    """
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=question,
    )
    text = getattr(response, "text", None)
    return text if text else "I could not generate a response."


def save_uploaded_pdf(uploaded_file) -> str:
    """Persist an uploaded PDF temporarily and return its path.

    Args:
        uploaded_file: Streamlit uploaded file object.

    Returns:
        Local temporary PDF path.
    """
    with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(uploaded_file.read())
        temp_path = temp_pdf.name
    return temp_path


def initialize_session_state() -> None:
    """Initialize Streamlit session variables used by the app."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "roadmap" not in st.session_state:
        st.session_state.roadmap = ""
    if "tutor_output" not in st.session_state:
        st.session_state.tutor_output = ""
    if "retrieved_context" not in st.session_state:
        st.session_state.retrieved_context = ""


def render_chat_history() -> None:
    """Render chat messages from session state."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def main() -> None:
    """Run the Streamlit application."""
    st.set_page_config(page_title="Autonomous Finance Tutor", layout="wide")
    st.title("Autonomous Finance Tutor")
    st.caption("Planner -> Retriever -> Tutor flow with Gemini + CrewAI + ChromaDB")

    initialize_session_state()

    with st.sidebar:
        st.header("Setup")
        api_key = st.text_input("Gemini API Key", type="password")
        uploaded_pdf = st.file_uploader("Upload finance PDF", type=["pdf"])
        enable_rag = st.checkbox("Use uploaded PDF for RAG", value=True)

        if st.button("Ingest PDF"):
            if not api_key:
                st.warning("Missing API key. Please provide your Gemini API key.")
            elif not uploaded_pdf:
                st.warning("No PDF uploaded. Please upload a PDF first.")
            else:
                try:
                    logger.info("Starting PDF ingestion from sidebar action")
                    os.environ["GOOGLE_API_KEY"] = api_key
                    pdf_path = save_uploaded_pdf(uploaded_pdf)
                    chunk_count = ingest_pdf_to_chroma(
                        pdf_path=pdf_path,
                        api_key=api_key,
                        persist_dir="chroma_store",
                    )
                    st.success(f"PDF ingested successfully. Stored {chunk_count} chunks.")
                    Path(pdf_path).unlink(missing_ok=True)
                except Exception as exc:
                    logger.exception("PDF ingestion failed: %s", exc)
                    st.error(f"Failed to ingest PDF: {exc}")

    if not api_key:
        st.warning("Please enter your Gemini API key in the sidebar to continue.")
        return

    os.environ["GOOGLE_API_KEY"] = api_key

    render_chat_history()
    user_query = st.chat_input("Ask a finance question")

    if not user_query:
        return

    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        try:
            logger.info("Received user query: %s", user_query)
            st.markdown("### Baseline Answer (Gemini)")
            baseline_answer = get_gemini_baseline_answer(user_query, api_key=api_key)
            st.markdown(baseline_answer)

            st.markdown("### Planner -> Tutor flow")
            st.info("Planner -> Tutor flow started")
            logger.info("Planner -> Tutor flow")

            roadmap = generate_roadmap(user_query=user_query, api_key=api_key)
            st.session_state.roadmap = roadmap
            st.markdown("#### Learning Roadmap")
            st.markdown(roadmap)

            retrieved_context = ""
            if enable_rag:
                if not uploaded_pdf and not Path("chroma_store").exists():
                    st.warning("No PDF uploaded yet. Please upload and ingest a PDF to use RAG.")

                retrieved_context, retrieved_docs = retrieve_context(
                    query=user_query,
                    api_key=api_key,
                    persist_dir="chroma_store",
                    k=3,
                )
                if not retrieved_docs:
                    st.warning(
                        "No relevant context found in vector store. Falling back to general explanation."
                    )
                    retrieved_context = ""
                st.session_state.retrieved_context = retrieved_context

            st.markdown("#### Retrieved Context (Transparency)")
            if st.session_state.retrieved_context:
                st.code(st.session_state.retrieved_context[:2000])
            else:
                st.caption("No retrieved context available.")

            tutor_output = generate_tutor_response(
                user_query=user_query,
                context=st.session_state.retrieved_context,
                api_key=api_key,
            )
            st.session_state.tutor_output = tutor_output

            st.markdown("#### Tutor Explanation + Quiz")
            st.markdown(tutor_output)

            full_response = (
                "Baseline Answer:\n"
                f"{baseline_answer}\n\n"
                "Roadmap:\n"
                f"{roadmap}\n\n"
                "Tutor Output:\n"
                f"{tutor_output}"
            )
            st.session_state.messages.append(
                {"role": "assistant", "content": full_response}
            )
        except Exception as exc:
            logger.exception("App flow failed: %s", exc)
            st.error(f"API flow failed after retry. Error: {exc}")


if __name__ == "__main__":
    main()
