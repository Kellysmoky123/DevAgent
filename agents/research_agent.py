from pydantic import BaseModel, Field
from typing import List, Dict, Any, Literal, Optional
from langchain.agents import create_agent
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
from graphs.state import AgentState
from config.models import model
from tools.github_tool import github_search
from tools.doc_scraper import scrape_documentation , discover_doc_navigation_links
from tools.tavily_search import web_search
from tools.llamaindex_manager import build_and_index
from config.logger import setup_logger

logger = setup_logger(__name__)


class ResearchStateUpdate(BaseModel):
    """Verified research results — URLs and summaries only, no raw content."""
    verified_doc_urls: List[str] = Field(
        min_length=3,
        description="URLs of documentation pages verified as useful (scraped and checked)."
    )
    changelog_url: Optional[str] = Field(
        default=None,
        description="URL of the library's changelog or release notes page."
    )
    github_repos: List[Dict[str, Any]] = Field(
        min_length=2,
        description="List of relevant GitHub repos with metadata (name, url, stars, language)."
    )
    code_snippets: List[Dict[str, Any]] = Field(
        min_length=3,
        description="Code snippets extracted from repos (code preview, file path, repo url)."
    )
    research_summary: str = Field(
        description="Concise summary of findings: key features, use cases, installation notes."
    )


# NOTE: create_index is NOT a tool — indexing happens outside the agent
research_agent = create_agent(
    model=model,
    tools = [web_search, scrape_documentation,
             discover_doc_navigation_links, github_search],
    response_format = ResearchStateUpdate,
    system_prompt = """You are a research agent that discovers and VERIFIES documentation and code.
    
    IMPORTANT: You only need to verify content quality, NOT memorize it. You will see
    truncated previews (first 3000 chars). That is enough to confirm the page is useful.
    Do NOT repeat or summarize scraped content in your messages — just verify and move on.
    
    Strategy:
    1. Scrape the provided documentation URL. Use discover_doc_navigation_links to find
       more pages. Scrape 3-5 key pages (installation, getting started, API reference).
    2. If docs are insufficient, use web_search to find alternatives.
    3. Use github_search to find code snippets using the library.
    4. Use web_search to find the library's changelog/release notes URL.
    5. Return the verified URLs (not content) and a brief research summary.
    
    Quality checks:
    - Verify scraped previews contain actual documentation, not error pages.
    - Ensure code snippets contain real code, not just README text.
    - If initial URLs fail, search for alternatives."""
    )

async def research_agent_node(state: AgentState, config: RunnableConfig) -> Command[Literal["writer_agent", "__end__"]]:
    library_name = state['lib_name']
    version = state['target_version']
    language = state['target_language']
    doc_url = state['docs_url']

    context = f"""library: {library_name}
    version: {version}
    language: {language}
    Official documentation URL: {doc_url}
    
    Task: Find and verify documentation pages and code snippets for a tutorial.
    Return verified URLs — the indexing will be done separately."""

    try:
        result = await research_agent.ainvoke({
            "messages":[
                {
                    "role":"user",
                    "content":context
                }
            ]
        }, config=config)
        updates = result["structured_response"]

        # ── Index creation OUTSIDE the agent (no LLM tokens used) ────
        logger.info(
            f"Research complete. Indexing {len(updates.verified_doc_urls)} docs "
            f"and {len(updates.code_snippets)} snippets..."
        )
        index_name = await build_and_index(
            library_name=library_name,
            version=version or "latest",
            language=language or "python",
            verified_doc_urls=updates.verified_doc_urls,
            code_snippets=[s for s in updates.code_snippets],
            changelog_url=updates.changelog_url,
        )
        logger.info(f"Index created: {index_name}")

        # Build state updates (no raw content — just URLs + index ref)
        state_updates = {
            "doc_index_id": index_name,
            "doc_content": None,  # full content is in the vector index
            "github_repos": [
                {"name": r.get("name", ""), "url": r.get("url", ""), "stars": r.get("stars", 0)}
                for r in updates.github_repos
            ],
            "code_snippets": updates.code_snippets,
            "research_summary": updates.research_summary,
            "changelog_content": updates.changelog_url,  # just the URL
            "current_agent": "research_agent",
        }

        return Command(
            goto = "writer_agent",
            update = state_updates
        )
    except Exception as e:
        logger.error(f"Research agent failed: {e}")
        return Command(
            goto = "__end__",
            update = {
                "errors": [f"Research agent failed: {str(e)}"],
                "current_agent": "research_agent",
            }
        )
