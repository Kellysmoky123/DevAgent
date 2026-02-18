from pydantic import BaseModel , Field
from typing import Literal, List
from langchain_core.runnables import RunnableConfig
from langchain.agents import create_agent
from langgraph.types import Command
from graphs.state import AgentState
from config.models import model
from config.logger import setup_logger
from tools.llamaindex_manager import query_index

logger = setup_logger(__name__)


class CritiqueStateUpdate(BaseModel):
    """Validation results and decision of crituque agent."""
    validation_passed: bool = Field(description = "'True' if tutorial is approved, 'False' if needs revison")
    issues_found: List[str] = Field(default_factory = list, description = "List of specific issues found (deprecated methods, errors, clarity problems)")
    critique_feedback: str = Field(description = "Detailed feedback for writer if validation failed , or 'APPROVED' if passed")
    final_markdown:str = Field(description="The approved tutorial markdown (copy from tutorial_draft if approved)")
    next_action:Literal["approve", "revise"] = Field(description = "'approve' to end workflow, 'revise' to send back to writer")

critique_agent = create_agent(
    model=model,
    tools=[query_index],
    response_format = CritiqueStateUpdate,
    system_prompt="""You are tutorial and code reveiwer specializing in tutorial validation and version compatibility.
    Your task is to thoroughly validate the tutorial for the specified library and version.
    
    Things you must check and validate:
    1. Validate if the markdown format of the tutorial is correct
    2. Ensure tutorial mentions correct version of library
    3. Check if any deprecated methods incompatible with the library version is used in tutorial. Use the `query_index` tool to search the provided index for deprecated methods or breaking changes in the changelog. The index name will be provided in the context.
    4. Check if any breaking changes from changelog will affect the tutorial (use `query_index`)
    5. Check all code blocks are syntactically valid
    6. Overall quality of the tutorial
    7. Check if all code blocks are explanied
    8. Verify structure and clarity
    
    If any critical issues found , set next_action = 'revise'
    Otherwise set next_action = 'approve'
    If any minor issues found , like single missing curly braces , comma etc. correct it and copy the content to final_markdown (correction is recommended if the error is very minor rather than going into another loop of revision)
    
    Feedback Format:
    If revising, provide SPECIFIC , ACTIONABLE feedback:
    - 'Line 45: Remove usage of deprecated method X, use Y instead'
    - 'Code block 2: Add explanation of what the code does'
    If approving, set critique_feedback to 'APPROVED' and copy tutorial_draft to final_markdown
    """
)

async def critique_agent_node(state:AgentState, config: RunnableConfig) -> Command[Literal["writer_agent", "__end__"]]:
    logger.info(f"Critique agent started for iteration: {state.get('iteration_count')}")
    max_iterations = state.get("max_iterations",3)
    current_iteration = state.get("iteration_count")

    if current_iteration>=max_iterations:
        return Command(
            goto = "__end__",
            update = {
                "final_markdown": state["tutorial_draft"],
                "validation_passed":False,
                "errors": ["Max iteration reached. Tutorial may have issues."],
                "current_agent": "critique_agent"
            }
        )
    tutorial = state["tutorial_draft"]
    # changelog is now in the index, so we pass the index id
    doc_index_id = state.get("doc_index_id")
    
    context = f"""Tutorial to validate:
    ---
    {tutorial}
    ---
    
    Metadata for validation:
    library:{state["lib_name"]}
    version:{state["target_version"]}
    language:{state["target_language"]}
    index_name:{doc_index_id}
    current iteration:{current_iteration}/{max_iterations}
    Perform comprehensive validation"""

    try:
        result = await critique_agent.ainvoke({
            "messages":[{
                "role":"user",
                "content":context
            }]

            }, config=config)
        updates = result["structured_response"]
        state_updates = updates.model_dump()
        state_updates["current_agent"]= "critique_agent"

        if updates.next_action =="approve":
            logger.info("Critique agent approved the tutorial")
            return Command(
                goto = "__end__",
                update = state_updates
            )
        
        else:
            logger.info(f"Critique agent requested revision: {updates.critique_feedback}")
            return Command(
                goto = "writer_agent",
                update = state_updates
            )
        
    except Exception as e:
        logger.error(f"Critique agent failed: {e}")
        return Command(
            goto = "__end__" ,
            update = {
                "final_markdown": tutorial,
                "validation_passed": False,
                "errors":[f"critique agent failed : {str(e)}"],
                "current_agent": "critique_agent"
            }
        )
