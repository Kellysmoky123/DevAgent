from pydantic import BaseModel, Field
from typing import Literal
from langchain.agents import create_agent
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
from graphs.state import AgentState
from config.models import model
from config.logger import setup_logger
from tools.llamaindex_manager import query_index

logger = setup_logger(__name__)

class TutorialStateUpdate(BaseModel):
    """Complete tutorial state with metadata."""
    tutorial_draft:str = Field(description="Complete tutorial draft with installation, concepts, and 3 to 5 code examples in a markdown format.")

writer_agent = create_agent(
    model=model,
    tools= [query_index],
    response_format = TutorialStateUpdate,
    system_prompt = """You are a technical documentation writer specilaized in creating Getting started tutorials.
    Your task is to write a comprehensive beginner-freindly Markdown tutorial following this structure :
    ###Tutorial Structure
    1. **Introduction** (1 paragraph)
    - What the library does
    - Key use cases
    
    2. **Installation**
    - version specific installation commands
    - Any prerequisites
    
    3. **Core Concepts** (2-3 paragraphs)
    - Query documentation with "core concepts and architecture"
    - Explain key abstractions.

    4. **Basic Usage** (First example)
    - Query code examples with "basic usage hello world"
    - simple working example with explanation.

    5. **Common Patterns** (2-3 examples)
    - Real world use cases with examples

    ###Tool usage strategy

    1. Use query_index tool to query and gather documentation and code examples from index.

    ###Requirements

    - Atleast 3 code blocks with examples
    - Code must be runnable and version appropriate
    - Use specific version number in install commands
    - Clear section headers
    - No deprecated methods (You will be validated later).
    - Proffesional but freindly tone
    - The tutorial's core concepts and usage examples should focus on fundamental, widely applicable patterns that transcend minor version changes.
    - While installation instructions and code examples should be runnable for the specified version, the explanations should aim for broader understanding rather than being strictly tied to a single version's nuances.
    - Ensure the tutorial is beginner-friendly and focuses on teaching the library's core functionality in a way that is useful across different versions where possible.
    - STRICTLY DO NOT start the response with a horizontal rule (---) or any separator.
    - Start key sections with Level 1 Markdown title (# Title).
    """
)

async def writer_agent_node(state: AgentState, config: RunnableConfig) -> Command[Literal["critique_agent", "__end__"]]:
    logger.info(f"Writer agent started for library: {state['lib_name']}")
    library_name = state['lib_name']
    version = state['target_version']
    language = state['target_language']
    package_manager = state['package_manager']
    index_id = state['doc_index_id']
    research_summary = state['research_summary']

    is_revision = bool(state['critique_feedback'])

    context = f"""
library : {library_name}
version : {version}
language : {language}
package manager : {package_manager}
index_id : {index_id}
research_summary : {research_summary}

{f"REVISION NEEDED - Adress this feedback:\n{state['critique_feedback']}" if is_revision else "Create a new tutorial from scratch"}

Generate a complete "Getting started" tutorial in markdown format."""
    
    try:
        result = await writer_agent.ainvoke({
            "messages":[
                {
                    "role":"user",
                    "content":context
                }
            ]
        }, config=config)
        updates = result["structured_response"]
        state_updates = updates.model_dump()
        state_updates["current_agent"] = "writer_agent"
        state_updates["iteration_count"] = state.get("iteration_count",0) + 1
        return Command(
            goto = "critique_agent",
            update = state_updates
        )
    
    except Exception as e:
        logger.error(f"Writer agent failed: {e}")
        return Command(
            goto = "critique_agent",
            update = {
                "errors": [f"Writer agent failed: {str(e)}"],
                "current_agent": "writer_agent"
            }
        )