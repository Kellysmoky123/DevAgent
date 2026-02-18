import os
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from dotenv import load_dotenv
from config.logger import setup_logger

load_dotenv()

logger = setup_logger(__name__)

tavily_api_key = os.getenv("TAVILY_API_KEY")
if not tavily_api_key:
    raise ValueError("TAVILY_API_KEY not found in environment variables.")
search = TavilySearch(api_key=tavily_api_key,
                      max_results=3,
                      search_depth ="advanced"
                      )

@tool
async def web_search(query: str) -> list[dict[str, str]]:
    """Tool to perform web search using TavilySearch.
    Args:
        query (str): The search query string.
    Returns:
        list[dict[str, str]]: A list of search results, each containing 'title', 'url', and 'content'.
    """
    try:
        results = await search.ainvoke(query)
        cleaned_results = []
        for result in results.get("results", []):
            cleaned_results.append({
                "title": result.get("title"),
                "url": result.get("url"),
                "content": result.get("content")
            })
        return cleaned_results
    except Exception as e:
        return [{"error": f"Web search failed: {str(e)}"}]