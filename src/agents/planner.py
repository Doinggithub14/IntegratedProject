"""Planner agent module using CrewAI for roadmap generation."""

from crewai import Agent, Crew, Task
from tenacity import retry, stop_after_attempt, wait_fixed

from src.utils.logger import get_logger

logger = get_logger(__name__)


def _build_planner_agent(api_key: str) -> Agent:
    """Build the planner agent.

    Args:
        api_key: Gemini API key used by CrewAI via LiteLLM.

    Returns:
        Configured CrewAI Agent for planning.
    """
    return Agent(
        role="Finance Learning Planner",
        goal="Break finance questions into clear beginner-friendly learning steps.",
        backstory=(
            "You are a patient finance curriculum designer. "
            "You create short, actionable roadmaps with practical order."
        ),
        llm="gemini/gemini-1.5-flash",
        max_iter=3,
        verbose=True,
    )


@retry(stop=stop_after_attempt(2), wait=wait_fixed(1), reraise=True)
def generate_roadmap(user_query: str, api_key: str) -> str:
    """Generate a step-by-step roadmap for a finance topic.

    Args:
        user_query: User question/topic.
        api_key: Gemini API key.

    Returns:
        Planner roadmap text.
    """
    logger.info("Planner Agent started for query: %s", user_query)

    planner_agent = _build_planner_agent(api_key=api_key)
    planning_task = Task(
        description=(
            "Create a concise learning roadmap for this query: "
            f"'{user_query}'. "
            "Return 3 to 6 ordered steps. "
            "Each step should be one sentence and beginner-friendly."
        ),
        expected_output=(
            "A numbered roadmap with 3 to 6 steps from fundamentals to practice."
        ),
        agent=planner_agent,
    )

    crew = Crew(agents=[planner_agent], tasks=[planning_task], verbose=True)
    result = crew.kickoff(inputs={"GOOGLE_API_KEY": api_key})

    roadmap_text = str(result)
    logger.info("Planner Agent completed successfully")
    return roadmap_text
