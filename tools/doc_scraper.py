from langchain_core.tools import tool
from llama_index.readers.web import TrafilaturaWebReader
from typing import Dict
from bs4 import BeautifulSoup
import httpx
import asyncio
from urllib.parse import urljoin, urlparse
from config.logger import setup_logger

logger = setup_logger(__name__)


@tool
async def scrape_documentation(url: str) -> Dict :
    """Scrape a documentation webpage and extract clean text content using TrafilaturaWebReader.
    Args: 
        url(str): The URL of the documentation page to scrape.
    Returns:
        Dict: A dictionary containing the scraped content and metadata(url, success status, error if any).
    """
    reader = TrafilaturaWebReader()
    try:
        documents = await asyncio.to_thread(reader.load_data, [url])
        if not documents:
            return {"url": url, "success": False, "error": "No content extracted from the page."}
        doc = documents[0]
        full_text = doc.text
        # Return only a preview — full content is indexed separately outside the agent
        MAX_PREVIEW = 3000
        return {
            "url": url,
            "success": True,
            "content_preview": full_text[:MAX_PREVIEW],
            "total_chars": len(full_text),
            "truncated": len(full_text) > MAX_PREVIEW,
        }
    except Exception as e:
        return {"url": url, "success": False, "error": str(e)}
    

@tool
async def discover_doc_navigation_links(url:str)-> Dict:
    """Extract internal documentation links from a base documentation page to discover more content.
    Args:
        url(str): The URL of the base documentation page to analyze.
    Returns:
        Dictionary with the list of internal documentation links found.
    """
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(url)
            response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        base_domain = urlparse(url).netloc

        nav_areas = soup.find_all(['nav', 'aside']) + soup.find_all(class_ = ['sidebar','toc','navigation'])
        doc_links = set()
        for area in nav_areas:
            for link in area.find_all('a', href=True):
                href = str(link['href']) 
                full_url = urljoin(url, href)
                if urlparse(full_url).netloc == base_domain:
                    doc_links.add(full_url)
        
        return {
            "success": True,
            "base_url": url,
            "discovered_links": list(doc_links)[:20],
            "total_links_found": len(doc_links)
        }
    except Exception as e:
        return {"success": False, "base_url": url, "error": str(e)}
    
    