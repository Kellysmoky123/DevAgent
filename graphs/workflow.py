from langgraph.graph import StateGraph, START, END
from graphs.state import AgentState
from agents.chat_agent import chat_agent_node
from agents.version_agent import version_agent_node
from agents.research_agent import research_agent_node
from agents.writer_agent import writer_agent_node
from agents.critique_agent import critique_agent_node

def build_devagent_workflow():
    """Build DevAgent Lab with 5 agents (merged chat interface)."""
    
    workflow = StateGraph(AgentState)
    
    workflow.add_node("chat_agent", chat_agent_node)
    workflow.add_node("version_agent", version_agent_node)
    workflow.add_node("research_agent", research_agent_node)
    workflow.add_node("writer_agent", writer_agent_node)
    workflow.add_node("critique_agent", critique_agent_node)
    workflow.add_edge(START, "chat_agent")
    
    return workflow.compile()
