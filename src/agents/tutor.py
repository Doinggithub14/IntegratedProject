"""Tutor agent module using CrewAI and retrieved context."""

from crewai import Agent, Crew, Task
from tenacity import retry, stop_after_attempt, wait_fixed

from src.utils.logger import get_logger

logger = get_logger(__name__)


def _build_tutor_agent(api_key: str) -> Agent:
    """Build the tutor agent.

    Args:
        api_key: Gemini API key used by CrewAI via LiteLLM.

    Returns:
        Configured CrewAI Agent for teaching.
    """
    return Agent(
        role="Empathetic Finance Tutor",
        goal="Explain finance topics clearly and build confidence with short quizzes.",
        backstory=(
            "You are a supportive personal tutor who explains concepts with simple language, "
            "checks understanding, and gives kind feedback."
        ),
        llm="gemini/gemini-1.5-flash",
        max_iter=3,
        verbose=True,
    )


@retry(stop=stop_after_attempt(2), wait=wait_fixed(1), reraise=True)
def generate_tutor_response(user_query: str, context: str, api_key: str) -> str:
    """Generate explanation, quiz, and encouragement based on user query and context.

    Args:
        user_query: User question.
        context: Retrieved context from RAG. Can be empty.
        api_key: Gemini API key.

    Returns:
        Tutor response as a formatted text.
    """
    logger.info("Tutor Agent started for query: %s", user_query)

    tutor_agent = _build_tutor_agent(api_key=api_key)

    context_block = context.strip() if context.strip() else "No document context available."
    tutoring_task = Task(
        description=(
            "Teach the user about this query: "
            f"'{user_query}'.\n"
            "Use this retrieved context if relevant:\n"
            f"{context_block}\n\n"
            "Output format:\n"
            "1) Explanation (short and clear)\n"
            "2) 2-question quiz\n"
            "3) Empathetic feedback sentence"
        ),
        expected_output=(
            "A clear explanation, a 2-question quiz, and an empathetic feedback line."
        ),
        agent=tutor_agent,
    )

    crew = Crew(agents=[tutor_agent], tasks=[tutoring_task], verbose=True)
    result = crew.kickoff(inputs={"GOOGLE_API_KEY": api_key})

    tutor_text = str(result)
    logger.info("Tutor Agent completed successfully")
    return tutor_text
