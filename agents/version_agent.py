from pydantic import BaseModel, Field
from typing import Optional, Literal
from langchain.agents import create_agent
from langgraph.types import Command
from langchain_core.runnables import RunnableConfig
from graphs.state import AgentState
from config.models import model
from tools.version_checker import (
    detect_language,
    get_latest_version,
    close_version_checker,
)
from tools.tavily_search import web_search
from config.logger import setup_logger

logger = setup_logger(__name__)


class VersionStateUpdate(BaseModel):
    """Complete version data required for research phase"""

    target_language: Literal["python", "javascript"] = Field(
        description="Detected or specified programming language (e.g. 'python', 'javascript')."
    )
    target_version: str = Field(description="Latest stable version of the library.")
    version_status: Literal["pending", "confirmed", "failed"] = Field(
        default="pending",
        description="Status of version detection.Status must be 'confirmed' when all data is gathered.",
    )
    package_manager: Literal["pip", "npm"] = Field(
        description="Package manager used for installation(e.g. 'PyPI', 'NPM')."
    )
    repository_url: str = Field(description="URL of the library's repository.")
    docs_url: str = Field(description="URL of the library's documentation.")
    release_date: str = Field(
        description="Release date of the latest version in ISO format(YYYY-MM-DD)."
    )


version_agent = create_agent(
    model=model,
    tools=[detect_language, get_latest_version, close_version_checker, web_search],
    response_format=VersionStateUpdate,
    system_prompt="""You are a version_agent whose task is to detect the programming language of a given library and fetch its latest stable version along with relevant metadata.
    Your goal is to populate all fields in the VersionStateUpdate response format accurately.
    Strategy:
    1. If the target language is not provided, use the detect_language tool to identify it based on the library name. If both Python and JavaScript versions exist, default to Python for now.
    2. Once the language is determined, use the get_latest_version tool to fetch the latest version, release date, package manager, repository URL, and documentation URL.
    3. If any critical information (like version number, repository URL, or docs URL) is missing, use the web_search tool to find this information from the web.
    4. Update the version_status to 'confirmed' only when all required information is gathered. If the library cannot be found or an error occurs, set version_status to 'failed' and provide an appropriate error message.
    5. Ensure all URLs are valid and the release date is in ISO format (YYYY-MM-DD).
    6. Always return in the VersionStateUpdate format with all fields populated to the best of your ability.""",
)


async def version_agent_node(
    state: AgentState,
    config: RunnableConfig,
) -> Command[Literal["research_agent", "__end__"]]:
    """Intelligent agent to detect library language and fetch latest version info, with web search fallback"""

    library_name = state["lib_name"]
    logger.info(f"Version agent started for library: {library_name}")

    try:
        result = await version_agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": f"Get the complete version information for the library '{library_name}'",
                    }
                ]
            },
            config=config,
        )
        updates = result["structured_response"]
        logger.info(f"Version agent found version: {updates.target_version}, Status: {updates.version_status}")
        state_updates = updates.model_dump()
        state_updates["lib_name"] = library_name
        state_updates["current_agent"] = "version_agent"
        return Command(
            goto=(
                "research_agent"
                if state_updates.get("version_status") == "confirmed"
                else "__end__"
            ),
            update=state_updates,
        )
    except Exception as e:
        logger.error(f"Version agent failed: {e}")
        return Command(
            goto="__end__",
            update={
                "errors": [f"Version agent failed: {str(e)}"],
                "version_status": "failed",
                "current_agent": "version_agent",
            },
        )
