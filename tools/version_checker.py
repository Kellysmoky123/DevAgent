import os
import httpx
import asyncio
from typing import Dict, Optional, Literal
from datetime import datetime
from langchain_core.tools import tool
from config.logger import setup_logger

logger = setup_logger(__name__)


class VersionChecker:
    """Unified interface for checking library versions across different package managers."""

    def __init__(self):
        self.client = httpx.AsyncClient(timeout=15)

    async def close(self):
        """Close the current httpx client session."""
        if not self.client.is_closed:
            await self.client.aclose()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, *args):
        await self.close()

    async def detect_language(self,library_name: str) -> Optional[str]:
        """Auto detect if library is Python or JavaScript by querying both registries."""

        pypi_url = f"https://pypi.org/pypi/{library_name}/json"
        npm_url = f"https://registry.npmjs.org/{library_name}"

        pypi_exists,npm_exists = await asyncio.gather(
            self._check_exists(pypi_url),
            self._check_exists(npm_url))

        if pypi_exists and not npm_exists:
            return 'python'
        elif npm_exists and not pypi_exists:
            return 'javascript'
        elif pypi_exists and npm_exists:
            return 'both'
        return None
    
    async def get_latest_version(self,library_name: str, language: str) -> Dict:
        """Get the latest version and release date of a library from the appropriate registry."""

        if language == 'python':
            return await self._get_pypi_version(library_name)
        elif language == 'javascript':
            return await self._get_npm_version(library_name)
        else:
            return {"error":"Unsupported language specified."}
    


    async def _get_pypi_version(self,package:str) -> Dict:
        """Query PyPI JSON API for latest version"""
        url = f"https://pypi.org/pypi/{package}/json"
        try:
            
            response = await self.client.get(url)
            response.raise_for_status()
            data = response.json()
            info = data.get('info', {})
            version = info.get('version')
            project_urls = info.get('project_urls', {})
            releases = data.get('releases', {})
            latest_release = releases.get(version, [])
            release_date_raw = latest_release[0].get('upload_time') if latest_release else None
            release_date = None
            if release_date_raw:
                try:
                    release_date = datetime.fromisoformat(release_date_raw.replace('Z', '+00:00'))
                except (ValueError, TypeError):
                    release_date = release_date_raw
            return {'version': version, 
                    'release_date': release_date,
                    'package_manager': 'pip',
                    'docs_url':info.get('docs_url') or project_urls.get('Documentation') or project_urls.get('documentation'),
                    'repository_url':project_urls.get('Repository') or project_urls.get('repository')}
        except httpx.HTTPStatusError as e:
            logger.warning(f"HTTP error fetching PyPI version for {package}: {e}")
            return {'error': f"Status {e.response.status_code}: {str(e)}"}
        except Exception as e:
            logger.error(f"Unexpected error fetching PyPI version for {package}: {e}")
            return {'error': str(e)}
            


    async def _get_npm_version(self,package:str) -> Dict:
        """Query npm registry for latest version"""
        url = f"https://registry.npmjs.org/{package}"

        try:
            response = await self.client.get(url)
            response.raise_for_status()
            data = response.json()
            latest_version = data.get("dist-tags", {}).get("latest")
            if not latest_version:
                return {"error": "No version found for this package"}
            version_metadata = data.get("versions", {}).get(latest_version, {})
            release_date_raw = data.get("time", {}).get(latest_version)
            release_date = None
            if release_date_raw:
                try:
                # Handle standard ISO and Z-format safely
                    release_date = datetime.fromisoformat(release_date_raw.replace('Z', '+00:00'))
                except (ValueError, TypeError):
                    release_date = release_date_raw

            repository = version_metadata.get('repository', {})
            repo_url = repository.get('url') if isinstance(repository, dict) else repository

            return {
                'version': latest_version,
                'release_date': release_date,
                'package_manager': 'npm',
                'homepage': version_metadata.get('homepage'),
                'docs_url': None, 
                'repository': repository,
                'repository_url': repo_url
            }
        except httpx.HTTPStatusError as e:
            logger.warning(f"HTTP error fetching npm version for {package}: {e}")
            return {'error': f"Status {e.response.status_code}: {str(e)}"}
        except Exception as e:
            logger.error(f"Unexpected error fetching npm version for {package}: {e}")
            return {'error': str(e)}


    async def _check_exists(self,url:str) -> bool:
        """Check if package exists on PyPI or npm by sending a HEAD request."""
        try:
                response = await self.client.head(url,follow_redirects=True)
                return response.status_code == 200
        except Exception:
            return False
        
checker = VersionChecker()

@tool
async def detect_language(library_name: str) -> Optional[str]:
    """Auto detect if library is Python or JavaScript by querying both registries."""
    return await checker.detect_language(library_name)

@tool
async def get_latest_version(library_name: str, language: str) -> Dict:
    """Get the latest version and release date of a library from the appropriate registry."""
    return await checker.get_latest_version(library_name, language)

@tool
async def close_version_checker() -> None:
    """Close the VersionChecker's httpx client session."""
    await checker.close()
