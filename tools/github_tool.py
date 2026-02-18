import httpx
import os
import base64
import logging
import ast
from typing import List, Dict, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
from langchain_core.tools import tool
from config.logger import setup_logger

load_dotenv()

# Configure logging
logger = setup_logger(__name__)


@dataclass
class CodeSnippet:
    """Data class for code snippets."""
    code: str
    path: str
    repo: str
    doc_type: str
    function_name: Optional[str] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None


class GitHubTool:
    """Tool to search and retrieve code snippets from GitHub repositories."""
    
    def __init__(self):
        self.token = os.getenv('GITHUB_TOKEN')
        if not self.token:
            logger.warning("GITHUB_TOKEN not found. API rate limits will be severely restricted.")
        
        self.headers = {
            'Authorization': f'token {self.token}' if self.token else '',
            'Accept': 'application/vnd.github.v3+json'
        }
        
        self.exclude_paths = [
            'test/', 'tests/', '__test__/', 'spec/', 
            'example/', 'examples/', 'demo/', 'demos/',
            'tutorial/', 'tutorials/', 'docs/', 'documentation/',
            '.github/', 'node_modules/', 'venv/', 'env/', 'dist/',
            '__pycache__/', 'build/'
        ]
        
        # Reusable client for connection pooling
        self.client = httpx.AsyncClient(timeout=30.0, headers=self.headers)
    
    async def close(self):
        """Close the httpx client."""
        if not self.client.is_closed:
            await self.client.aclose()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, *args):
        await self.close()

    def _should_exclude_path(self, path: str) -> bool:
        """Check if a file path should be excluded."""
        return any(excluded in path.lower() for excluded in self.exclude_paths)

    async def search_repos(
        self,
        library_name: str,
        language: str,
        min_stars: int = 100,
        max_results: int = 5
    ) -> List[Dict]:
        """
        Search GitHub for repositories that use a specific library.
        
        Args:
            library_name: Name of the library to search for
            language: Programming language (python or javascript)
            min_stars: Minimum number of stars
            max_results: Maximum number of repositories to return
            
        Returns:
            List of repository information dictionaries
        """
        logger.info(f"Searching for repos using {library_name} in {language}")
        
        query = f"{library_name} language:{language} stars:>{min_stars}"
        url = "https://api.github.com/search/repositories"
        params = {
            'q': query,
            'sort':'updated',
            'order':'desc',
            'per_page': max_results
        }

        try:
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            repos = [
                {
                    'name': repo['full_name'],
                    'url': repo['html_url'],
                    'stars': repo['stargazers_count'],
                    'language': repo['language'],
                    'default_branch': repo['default_branch'],
                    'description': repo.get('description', ''),
                    'last_updated': repo['pushed_at'],
                    'topics': repo.get('topics', [])
                }
                for repo in data.get('items', [])[:max_results]
            ]
            
            logger.info(f"Found {len(repos)} repositories")
            return repos
                
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
            return []
        except Exception as e:
            logger.error(f"Error searching repositories: {str(e)}")
            return []

    async def get_library_files(
        self,
        repo_name: str,
        library_name: str,
        language: str,
        max_files: int = 10
    ) -> List[Dict]:
        """
        Get file contents from a repository that import/use the specified library.
        
        Args:
            repo_name: Full repository name (owner/repo)
            library_name: Name of the library to search for
            language: Programming language (python or javascript)
            max_files: Maximum number of files to retrieve
            
        Returns:
            List of file content dictionaries
        """
        logger.info(f"Fetching files from {repo_name} that use {library_name}")
        
        # Build search query based on language
        if language.lower() == 'python':
            code_query = f"import {library_name} OR from {library_name} repo:{repo_name} extension:py"
        elif language.lower() == 'javascript':
            code_query = f"require('{library_name}') OR import {library_name} repo:{repo_name} extension:js"
        else:
            logger.error(f"Unsupported language: {language}")
            return []

        url = "https://api.github.com/search/code"
        params = {'q': code_query, 'per_page': max_files}

        try:
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            code_files = response.json().get('items', [])

            # Filter out excluded paths
            code_files = [
                f for f in code_files 
                if not self._should_exclude_path(f['path'])
            ]

            files_content = []
            for file in code_files[:max_files]:
                try:
                    # Fetch file content
                    file_response = await self.client.get(file['url'])
                    file_response.raise_for_status()
                    file_data = file_response.json()

                    # Decode base64 content
                    content = base64.b64decode(file_data['content']).decode('utf-8', errors='ignore')

                    files_content.append({
                        'content': content,
                        'path': file['path'],
                        'repo': repo_name,
                        'url': file['html_url'],
                        'size': file_data.get('size', 0)
                    })
                    
                    logger.info(f"Retrieved file: {file['path']}")
                        
                except Exception as e:
                    logger.warning(f"Failed to fetch file {file['path']}: {str(e)}")
                    continue

            logger.info(f"Retrieved {len(files_content)} files from {repo_name}")
            return files_content
                
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
            return []
        except Exception as e:
            logger.error(f"Error fetching files: {str(e)}")
            return []

    def _extract_python_functions(
        self,
        code: str,
        library_name: str,
        file_path: str,
        repo_name: str
    ) -> List[CodeSnippet]:
        """
        Extract function/class-level snippets that actually use the library.
        
        Two-pass approach:
          Pass 1 → Resolve all aliases and imported symbols from import statements.
                   e.g. `import pandas as pd`        → aliases = {"pd"}
                        `from pandas import read_csv` → symbols = {"read_csv"}
          Pass 2 → Walk each function/class AST and check for:
                   - ast.Attribute nodes where .value.id is in aliases  (pd.DataFrame)
                   - ast.Name nodes where .id is in imported symbols     (read_csv)
        """
        snippets = []

        try:
            tree = ast.parse(code)
            lines = code.splitlines()

            # ----------------------------------------------------------
            # PASS 1: Collect library aliases & imported symbols
            # ----------------------------------------------------------
            library_aliases: set = set()      # e.g. {"pd", "pandas"}
            imported_symbols: set = set()     # e.g. {"read_csv", "DataFrame"}

            for node in ast.walk(tree):
                # import pandas / import pandas as pd
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == library_name:
                            library_aliases.add(alias.asname or alias.name)

                # from pandas import read_csv / from pandas import read_csv as rc
                elif isinstance(node, ast.ImportFrom):
                    if node.module == library_name:
                        for alias in node.names:
                            imported_symbols.add(alias.asname or alias.name)

            # Bare `import pandas` with no alias still means "pandas" is the name
            if library_name not in library_aliases and not imported_symbols:
                library_aliases.add(library_name)

            logger.debug(f"Aliases: {library_aliases}, Symbols: {imported_symbols}")

            # If nothing was imported, this file doesn't actually use the library
            if not library_aliases and not imported_symbols:
                return snippets

            # ----------------------------------------------------------
            # PASS 2: Find functions/classes that use the library
            # ----------------------------------------------------------
            relevant_nodes: list = []

            class LibraryUsageVisitor(ast.NodeVisitor):
                def visit_FunctionDef(self, node):
                    if self._uses_library(node):
                        relevant_nodes.append(node)
                    self.generic_visit(node)

                def visit_AsyncFunctionDef(self, node):
                    if self._uses_library(node):
                        relevant_nodes.append(node)
                    self.generic_visit(node)

                def visit_ClassDef(self, node):
                    if self._uses_library(node):
                        relevant_nodes.append(node)
                    self.generic_visit(node)

                def _uses_library(self, node) -> bool:
                    for child in ast.walk(node):
                        # Catches: pd.DataFrame(), pd.read_csv(), etc.
                        if isinstance(child, ast.Attribute):
                            if isinstance(child.value, ast.Name):
                                if child.value.id in library_aliases:
                                    return True
                        # Catches: read_csv(), DataFrame(), etc.
                        if isinstance(child, ast.Name):
                            if child.id in imported_symbols:
                                return True
                    return False

            LibraryUsageVisitor().visit(tree)

            # ----------------------------------------------------------
            # Deduplicate: skip methods already inside a matched class
            # ----------------------------------------------------------
            covered_ranges: list = []
            for node in relevant_nodes:
                if isinstance(node, ast.ClassDef):
                    covered_ranges.append((node.lineno, node.end_lineno))

            relevant_nodes = [
                node for node in relevant_nodes
                if isinstance(node, ast.ClassDef)
                or not any(start <= node.lineno and node.end_lineno <= end for start, end in covered_ranges)
            ]

            # ----------------------------------------------------------
            # Extract source code blocks from matched nodes
            # ----------------------------------------------------------
            for node in relevant_nodes:
                start = node.lineno - 1
                end = node.end_lineno
                function_code = '\n'.join(lines[start:end])

                snippets.append(CodeSnippet(
                    code=function_code,
                    path=file_path,
                    repo=repo_name,
                    doc_type='python_function',
                    function_name=node.name,
                    line_start=node.lineno,
                    line_end=end
                ))
                logger.debug(f"Extracted function: {node.name} from {file_path}")

        except SyntaxError as e:
            logger.warning(f"Syntax error parsing {file_path}: {str(e)}")
        except Exception as e:
            logger.error(f"Error extracting Python functions from {file_path}: {str(e)}")

        return snippets

    def _extract_javascript_functions(
        self,
        code: str,
        library_name: str,
        file_path: str,
        repo_name: str
    ) -> List[CodeSnippet]:
        """
        Extract function-level snippets that actually use the library.
        
        Two-pass approach (mirrors the Python version's logic):
          Pass 1 → Resolve aliases and named imports from JS import/require statements.
                   e.g. `const pd = require('pandas')`          → aliases = {"pd"}
                        `import pd from 'pandas'`               → aliases = {"pd"}
                        `import { read_csv } from 'pandas'`     → symbols = {"read_csv"}
                        `import { read_csv as rc } from 'pandas'` → symbols = {"rc"}
          Pass 2 → Find functions whose body references those aliases or symbols.
        """
        import re
        snippets = []

        try:
            # ----------------------------------------------------------
            # PASS 1: Collect library aliases & imported symbols
            # ----------------------------------------------------------
            library_aliases: set = set()
            imported_symbols: set = set()
            escaped = re.escape(library_name)

            # const x = require('library') / let x = require("library")
            for match in re.finditer(
                rf'(?:const|let|var)\s+(\w+)\s*=\s*require\(\s*["\']' + escaped + r'["\']\s*\)',
                code
            ):
                library_aliases.add(match.group(1))

            # import x from 'library'  (default import)
            for match in re.finditer(
                rf'import\s+(\w+)\s+from\s+["\']' + escaped + r'["\']',
                code
            ):
                library_aliases.add(match.group(1))

            # import { a, b as c } from 'library'  (named imports, with optional aliases)
            named_block = re.search(
                rf'import\s+\{{([^}}]+)\}}\s+from\s+["\']' + escaped + r'["\']',
                code
            )
            if named_block:
                for item in named_block.group(1).split(','):
                    item = item.strip()
                    if not item:
                        continue
                    # `a as b` → use "b", otherwise use the name directly
                    parts = item.split(' as ')
                    imported_symbols.add(parts[-1].strip())

            # import * as x from 'library'  (namespace import)
            for match in re.finditer(
                rf'import\s+\*\s+as\s+(\w+)\s+from\s+["\']' + escaped + r'["\']',
                code
            ):
                library_aliases.add(match.group(1))

            logger.debug(f"Aliases: {library_aliases}, Symbols: {imported_symbols}")

            # If nothing was imported, this file doesn't actually use the library
            if not library_aliases and not imported_symbols:
                return snippets

            # ----------------------------------------------------------
            # PASS 2: Find functions that use the library
            # ----------------------------------------------------------
            # Pattern captures: function declarations, arrow functions, async variants, class methods
            function_pattern = re.compile(
                r'(?:export\s+)?(?:export\s+default\s+)?'   # optional export
                r'(?:async\s+)?'                             # optional async
                r'(?:'
                r'function\s*\*?\s+(\w+)\s*\([^)]*\)'       # function name(...)
                r'|'
                r'(\w+)\s*=\s*(?:async\s+)?'                # const name = (async)?
                r'(?:\([^)]*\)|(\w+))\s*=>'                 # (...) => or arg =>
                r'|'
                r'(?<![.\w])(\w+)\s*\([^)]*\)\s*(?=\{)'     # method(...)  — no dot/word before it
                r')\s*\{',
                re.MULTILINE
            )

            lines = code.splitlines()

            for match in function_pattern.finditer(code):
                # Resolve function name from whichever capture group matched
                function_name = match.group(1) or match.group(2) or match.group(4) or 'anonymous'
                body_start = match.end() - 1  # position of the opening '{'

                # Find the matching closing brace
                body_end = self._find_closing_brace(code, body_start)
                if body_end == -1:
                    continue

                function_code = code[match.start():body_end + 1]

                # Check if the function body actually uses any alias or symbol
                if not self._js_body_uses_library(function_code, library_aliases, imported_symbols):
                    continue

                # Calculate line numbers
                line_start = code[:match.start()].count('\n') + 1
                line_end = line_start + function_code.count('\n')

                snippets.append(CodeSnippet(
                    code=function_code,
                    path=file_path,
                    repo=repo_name,
                    doc_type='javascript_function',
                    function_name=function_name,
                    line_start=line_start,
                    line_end=line_end
                ))
                logger.debug(f"Extracted function: {function_name} from {file_path}")

        except Exception as e:
            logger.error(f"Error extracting JavaScript functions from {file_path}: {str(e)}")

        return snippets

    @staticmethod
    def _find_closing_brace(code: str, open_pos: int) -> int:
        """Find the matching closing brace for an opening brace at open_pos."""
        depth = 0
        for i in range(open_pos, len(code)):
            if code[i] == '{':
                depth += 1
            elif code[i] == '}':
                depth -= 1
                if depth == 0:
                    return i
        return -1  # no matching brace found

    @staticmethod
    def _js_body_uses_library(body: str, aliases: set, symbols: set) -> bool:
        """
        Check if a JS function body references any library alias or imported symbol.
        Uses word-boundary regex to avoid partial matches.
        
        Aliases match both dot access and direct calls:
            express.Router()  →  alias.
            express()         →  alias(
        Symbols match calls and property access:
            Router()          →  symbol(
            Router.use()      →  symbol.
        """
        import re
        for name in aliases:
            if re.search(rf'\b{re.escape(name)}\s*[.(]', body):
                return True
        for name in symbols:
            if re.search(rf'\b{re.escape(name)}\s*[.(]', body):
                return True
        return False

    def extract_function_snippets(
        self,
        files_content: List[Dict],
        library_name: str,
        language: str
    ) -> List[Dict]:
        """
        Extract function-level code snippets where the library is used.
        
        Args:
            files_content: List of file content dictionaries from get_library_files
            library_name: Name of the library to search for
            language: Programming language (python or javascript)
            
        Returns:
            List of code snippet dictionaries
        """
        logger.info(f"Extracting function snippets for {library_name} in {language}")
        
        all_snippets = []
        
        for file_data in files_content:
            code = file_data['content']
            file_path = file_data['path']
            repo_name = file_data['repo']
            
            if language.lower() == 'python':
                snippets = self._extract_python_functions(code, library_name, file_path, repo_name)
            elif language.lower() == 'javascript':
                snippets = self._extract_javascript_functions(code, library_name, file_path, repo_name)
            else:
                logger.warning(f"Unsupported language for extraction: {language}")
                continue
            
            # Convert to dictionaries, truncate code to reduce token usage
            MAX_SNIPPET_CHARS = 2000
            for snippet in snippets:
                code = snippet.code
                all_snippets.append({
                    'code': code[:MAX_SNIPPET_CHARS],
                    'path': snippet.path,
                    'repo': snippet.repo,
                    'doc_type': snippet.doc_type,
                    'function_name': snippet.function_name,
                    'line_start': snippet.line_start,
                    'line_end': snippet.line_end,
                    'truncated': len(code) > MAX_SNIPPET_CHARS
                })
        
        logger.info(f"Extracted {len(all_snippets)} function snippets")
        return all_snippets


# Example usage for LangGraph integration
async def main():
    """Example usage of the GitHub tool."""
    tool = GitHubTool()
    
    library_name = "pandas"
    language = "python"
    
    # Step 1: Search for repos
    repos = await tool.search_repos(library_name, language, min_stars=50, max_results=3)
    logger.info(f"Found repos: {[r['name'] for r in repos]}")
    
    if repos:
        # Step 2: Get files from first repo
        repo_name = repos[0]['name']
        files = await tool.get_library_files(repo_name, library_name, language, max_files=5)
        logger.info(f"Retrieved {len(files)} files")
        
        # Step 3: Extract function snippets
        snippets = tool.extract_function_snippets(files, library_name, language)
        logger.info(f"Extracted {len(snippets)} snippets")
        
        # Display first snippet
        if snippets:
            for snippet in snippets:
                logger.info(f"\nExample snippet from {snippet['function_name']}:")
                logger.info(f"File: {snippet['path']}")
                logger.info(f"Lines: {snippet['line_start']}-{snippet['line_end']}")
                logger.info(f"Code:\n{snippet['code']}")

github_tool = GitHubTool()
@tool
async def github_search(library_name: str, language: str) -> List[Dict]:
    """Tool to search GitHub for repositories using the specified library and extract code snippets.
        Args: 
            library_name : name of the library
            language : name of the programming language
            
        Returns : List of Dictionary of extracted code snippets with metadata"""
    repos = await github_tool.search_repos(library_name, language, min_stars=50, max_results=2)
    if not repos:
        return []
    
    all_snippets = []
    for repo in repos:
        files = await github_tool.get_library_files(repo['name'], library_name, language, max_files=3)
        snippets = github_tool.extract_function_snippets(files, library_name, language)
        all_snippets.extend(snippets)
    
    return all_snippets



if __name__ == "__main__":
    import asyncio
    asyncio.run(main())