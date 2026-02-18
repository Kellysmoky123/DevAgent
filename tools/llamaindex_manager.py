from llama_index.core import VectorStoreIndex , Document , StorageContext
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter
from llama_index.core.node_parser import SentenceSplitter, CodeSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from langchain_core.tools import tool
import chromadb
from typing import List, Dict, Optional
from pathlib import Path
from dotenv import load_dotenv
import asyncio
import os
import shutil
import time


load_dotenv()
from config.logger import setup_logger

logger = setup_logger(__name__)

class LlamaIndexManager:
    """Manages documentation indexing with version aware metadata."""
    def __init__(self, persist_dir: str='.data/indexes'):
        self.persist_dir= Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.embed_model = GoogleGenAIEmbedding(
            model_name="gemini-embedding-001",
            api_key=os.getenv("GOOGLE_API_KEY"),
        )

    def get_index_name(self, library_name: str, version: str, language: str) -> str:
        """Generate consistent index name."""
        return f"{library_name}_{language}_{version}".replace('.', '_')

    def index_exists(self, library_name: str, version: str, language: str) -> bool:
        """Check if a populated index already exists and matches current embedding dimension."""
        index_name = self.get_index_name(library_name, version, language)
        index_path = self.persist_dir / index_name
        
        try:
            chroma_client = chromadb.PersistentClient(path=str(index_path))
            collection = chroma_client.get_or_create_collection(name=index_name)
            
            if collection.count() == 0:
                return False
                
            # Check dimension compatibility (auto-fix for model switching)
            try:
                stored_embedding = collection.peek(limit=1)['embeddings'][0]
                current_dim = len(self.embed_model.get_text_embedding("test"))
                
                if len(stored_embedding) != current_dim:
                    logger.warning(f"Dimension mismatch (stored: {len(stored_embedding)}, current: {current_dim}). Rebuilding index...")
                    del chroma_client # Attempt to close client
                    shutil.rmtree(index_path, ignore_errors=True)
                    return False
            except Exception as e:
                # If checking dimensions fails, assume it's okay or let it rebuild if empty
                pass
                
            return True
        except Exception:
            return False

    def create_index(self,
                    library_name:str,
                    version:str,
                    language:str,
                    doc_content:List[Dict[str,str]],
                    code_snippets:List[Dict[str,str]]) -> str:
        """Create or get version specific index for library documentation and code snippets."""
        index_name = self.get_index_name(library_name, version, language)
        if self.index_exists(library_name, version, language):
            return index_name

        # convert scraped docs to LlamaIndex Documents with metadata
        doc_documents = []
        code_documents = []
        for doc in doc_content:
            metadata = {
                'library_name': library_name,
                'version': version,
                'language': language,
                'source_url': doc.get('url',''),
                'section': doc.get('section',''),
                'doc_type':'documentation'
            }
            doc_documents.append(Document(text=doc['content'], metadata=metadata))

        for snippet in code_snippets:
            # Skip empty or malformed snippets
            code_text = snippet.get('code', snippet.get('content', ''))
            if not code_text:
                continue
                
            metadata = {
                'library_name': library_name,
                'version': version,
                'language': language,
                'repo': snippet.get('repo',''),
                'file_path': snippet.get('path',''),
                'doc_type':'code_snippet'
            }
            code_documents.append(Document(text=code_text, metadata=metadata))

        # create persistent chroma vector store
        chroma_client = chromadb.PersistentClient(path=str(self.persist_dir / index_name))
        chroma_collection = chroma_client.get_or_create_collection(name=index_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        doc_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
        try:
            code_parser = CodeSplitter(
                language=language,
                chunk_lines = 40,
                chunk_lines_overlap = 10,
                max_chars = 1200
            )
        except Exception as e:
            logger.warning(f"CodeSplitter failed for language '{language}': {e}. Using SentenceSplitter as fallback.")
            code_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)

        doc_nodes = doc_parser.get_nodes_from_documents(doc_documents)
        code_nodes = code_parser.get_nodes_from_documents(code_documents)
        nodes = doc_nodes + code_nodes
        # Index with version metadata - processed in batches to respect API limits
        # Create empty index first
        index = VectorStoreIndex(
            nodes=[],
            storage_context=storage_context,
            embed_model=self.embed_model
        )
        
        # Insert nodes in small batches with delay to avoid 429 errors
        batch_size = 5
        logger.info(f"Indexing {len(nodes)} nodes in batches of {batch_size}...")
        
        for i in range(0, len(nodes), batch_size):
            batch = nodes[i : i + batch_size]
            try:
                index.insert_nodes(batch)
                time.sleep(2.0) # Rate limit delay
            except Exception as e:
                logger.warning(f"Error indexing batch {i}: {e}. Retrying with delay...")
                time.sleep(10.0)
                try:
                    index.insert_nodes(batch)
                except Exception:
                    logger.error(f"Failed to retry batch {i}, skipping.")

        return index_name
    
    def query_index(self,
                    index_name:str,
                    query:str,
                    doc_type_filter:str) ->str:
        """Retrieve version specific documentation or code snippets from index."""
        chroma_client = chromadb.PersistentClient(path=str(self.persist_dir / index_name))
        chroma_collection = chroma_client.get_collection(name=index_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

        index = VectorStoreIndex.from_vector_store(vector_store, embed_model=self.embed_model)
        
        # Use retriever instead of query_engine to avoid needing an LLM
        # The agent itself will interpret the retrieved chunks
        filters = None
        if doc_type_filter:
            filters = MetadataFilters(
                filters=[MetadataFilter(key="doc_type", value=doc_type_filter)]
            )
        retriever = index.as_retriever(similarity_top_k=5, filters=filters)
        nodes = retriever.retrieve(query)
        
        # Format retrieved chunks as readable text
        results = []
        for node in nodes:
            meta = node.metadata or {}
            source = meta.get('doc_type', 'unknown')
            results.append(f"[{source}] (score: {node.score:.3f})\n{node.text}\n")
        
        return "\n---\n".join(results) if results else "No relevant content found."
    
manager = LlamaIndexManager()
@tool
async def create_index(library_name:str,
                    version:str,
                    language:str,
                    doc_content:List[Dict[str,str]],
                    code_snippets:List[Dict[str,str]]) -> str:
    """Create version specific index for library documentation and code snippets with metadata
    
    Args:
        library: Name of the library
        version: Version of the library
        language: Programming language of the library
        doc_content:Scraped documentation content to index
        code_snippets:Scraped code snippets to index
    
    Returns:
        Name of the created index
    """
    return await asyncio.to_thread(manager.create_index, library_name, version, language, doc_content, code_snippets)

@tool
async def query_index(index_name:str,
                    query:str,
                    doc_type_filter:str) ->str:
    """Retrieve version specific documentation or code snippets from index
    Args:
        index_name: Name of the index to query
        query: Query to find semantic similarity
        doc_type_filter: documentation or code snippet flilter
        
    Retruns :
        Response string"""
    return await asyncio.to_thread(manager.query_index, index_name, query, doc_type_filter)


async def build_and_index(
    library_name: str,
    version: str,
    language: str,
    verified_doc_urls: list[str],
    code_snippets: list[dict],
    changelog_url: str | None = None,
) -> str:
    """Scrape verified URLs in full and create a vector index — called OUTSIDE the agent.

    Args:
        library_name: Name of the library
        version: Version string
        language: Programming language
        verified_doc_urls: URLs the agent confirmed as useful
        code_snippets: Code snippets from GitHub (already extracted by agent tools)
        changelog_url: Optional changelog URL to scrape

    Returns:
        Index name for later querying
    """
    from llama_index.readers.web import TrafilaturaWebReader

    # Check if index already exists to skip redundant scraping
    if await asyncio.to_thread(manager.index_exists, library_name, version, language):
        index_name = manager.get_index_name(library_name, version, language)
        logger.info(f"Using existing index found for {index_name} - skipping scraping.")
        return index_name

    reader = TrafilaturaWebReader()

    # Scrape all verified doc URLs in full (no truncation)
    doc_content = []
    for url in verified_doc_urls:
        try:
            documents = await asyncio.to_thread(reader.load_data, [url])
            if documents:
                doc_content.append({
                    "url": url,
                    "content": documents[0].text,
                    "section": url.split("/")[-1] if "/" in url else "",
                })
                logger.info(f"Scraped full content from: {url} ({len(documents[0].text)} chars)")
        except Exception as e:
            logger.warning(f"Failed to scrape {url}: {e}")

    # Scrape changelog if provided
    if changelog_url:
        try:
            documents = await asyncio.to_thread(reader.load_data, [changelog_url])
            if documents:
                doc_content.append({
                    "url": changelog_url,
                    "content": documents[0].text,
                    "section": "changelog",
                })
                logger.info(f"Scraped changelog from: {changelog_url}")
        except Exception as e:
            logger.warning(f"Failed to scrape changelog {changelog_url}: {e}")

    # Create the index with full content
    index_name = await asyncio.to_thread(
        manager.create_index,
        library_name,
        version,
        language,
        doc_content,
        code_snippets,
    )

    logger.info(
        f"Created index '{index_name}' with {len(doc_content)} docs "
        f"and {len(code_snippets)} snippets"
    )
    return index_name