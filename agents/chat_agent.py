from pydantic import BaseModel , Field
from typing import Literal , Optional
from langchain.agents import create_agent
from langgraph.types import Command
from langchain_core.runnables import RunnableConfig
from tools.llamaindex_manager import query_index
from graphs.state import AgentState
from config.models import model
from config.logger import setup_logger

# Configure logging
logger = setup_logger(__name__)

class ChatResponse(BaseModel):
    action:Literal["generate_tutorial","answer_question","clarify"] = Field(description="What action to take: start tutorial generation, answer question directly, or ask for clarification")
    lib_name:Optional[str] = Field(description= "Name of the library in lowercase to avoid case sensitivity issues , if task is to generate tutorial")
    target_language:Optional[Literal["python","javascript"]] = Field(description = "Programming language if mentioned")
    response_message:str = Field(description="Direct answer to user(if action is answer question or clarify), or confirmation message(if action is generate tutorial)")
    additional_code:Optional[str] = Field(description = "Code example if answering question that need demonstration")


chat_agent = create_agent(model = model,
tools = [query_index],
response_format = ChatResponse,
system_prompt = """You are the user-facing chat agent for DevAgent Lab, a tutorial generation system.

## Your Responsibilities:
1. **Tutorial Generation Requests** - Route to backend workflow
2. **Follow-up Questions** - Answer using indexed docs and generated tutorial
3. **Clarifications** - Ask for more info when unclear

## Decision Logic:

### If NO tutorial exists yet:
- Look for library name in message 
- If found → action="generate_tutorial", library_name="X", response_message="I'll generate a tutorial for X..."
- If not found → action="clarify", response_message="Which library would you like to learn?"

### If tutorial ALREADY exists:
- User asking questions about it? → action="answer_question"
  - Use query_index(index_id, question, doc_type) to find answers
  - Include code examples if relevant
- User wants NEW tutorial? → action="generate_tutorial" (extract new library name)

## Examples:

User: "Create FastAPI tutorial"
→ action="generate_tutorial", library_name="FastAPI", response_message="Generating FastAPI tutorial..."

User: "How do I handle authentication?" (after tutorial exists)
→ action="answer_question", query docs, response_message="Here's how authentication works: [explanation + code]"

User: "I want to learn something"
→ action="clarify", response_message="Which library or framework would you like to learn?"

Keep responses friendly and technical. Use code blocks for examples.
Your task is to help users to learn new libraries. Don't answer questions that not related to the tutorial or library learning (IMPORTANT!!!)"""
)

async def chat_agent_node(state:AgentState, config: RunnableConfig):
    logger.info(f"Chat agent processing query: {state['user_query']}")

    user_message = state["user_query"]
    tutorial_exists = bool(state["final_markdown"])
    tutorial_content = state["final_markdown"]
    index_id = state["doc_index_id"]
    library_name = state["lib_name"]
    chat_history = state.get("chat_history", [])

    # Format conversation history for context
    history_text = ""
    if chat_history:
        history_text = "\n    Conversation History (last messages):\n"
        for msg in chat_history[-10:]:
            role = "User" if msg["role"] == "user" else "Agent"
            content = msg["content"][:300]  # Truncate long messages
            history_text += f"    - {role}: {content}\n"

    context = f"""
    User Message: "{user_message}"

    Current State:
    - Tutorial Generated: {tutorial_exists}
    - Library: {library_name if library_name else "None"}
    - Index Available: {bool(index_id)}
    Generated Tutorial Preview:
    {tutorial_content[:1000] if tutorial_exists else "No tutorial generated yet."}
    {history_text}
    Process the user's message and respond appropriately. Use the conversation history to understand context from previous messages.
    """

    try:
        result = await chat_agent.ainvoke({
            "messages":[
                {
                    "role":"user",
                    "content":context
                }
            ]
        
        }, config=config)
        decision = result["structured_response"]
        logger.info(f"Chat agent decision: {decision.action}, Library: {decision.lib_name}")
        
        if decision.action == "generate_tutorial":
            response = decision.response_message
            return Command(
                goto = "version_agent",
                update = {
                    "lib_name": decision.lib_name,
                    "target_language": decision.target_language,
                    "user_intent": "generate_tutorial",
                    "session_mode": "generating",
                    "qa_response": response,  # Confirmation message
                    "current_agent": "chat_agent"
                }

            )
        elif decision.action == "answer_question":
            full_response = decision.response_message
            if decision.additional_code :
                full_response += f"\n\n```{decision.additional_code}\n```"

            return Command(
                goto = "__end__",
                update = {
                    "qa_response": full_response,
                    "user_intent": "ask_question",
                    "session_mode": "interactive",
                    "current_agent": "chat_agent"
                }
            )
        else :
            return Command(
                goto = "__end__",
                update = {
                    "qa_response": decision.response_message,
                    "user_intent": "unclear",
                    "session_mode": "interactive",
                    "current_agent": "chat_agent"
                }
            )
        
    except Exception as e:
        logger.error(f"Chat agent failed: {e}")
        return Command(
            goto = "__end__",
            update = {
                "qa_response": "I'm having trouble understanding. Could you rephrase your request",
                "errors": [f"Chat agent failed : {str(e)}"],
                "current_agent": "chat_agent"
            }
        )