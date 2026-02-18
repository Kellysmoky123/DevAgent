from typing import TypedDict, List ,Dict , Any, Optional , Literal, Annotated
import operator

class AgentState(TypedDict):
    lib_name: str
    user_query: str
    target_language:Optional[Literal['python','javascript']]

    # Version tracking for accuracy
    target_version: Optional[str]
    version_status: Literal['pending','confirmed','failed']
    package_manager: Optional[Literal['pip','npm']]
    release_date: Optional[str]
    repository_url: Optional[str]

    # Documentation sources
    docs_url: Optional[str]
    doc_content:Optional[str]
    doc_index_id: Optional[str]

    # Github code
    github_repos: List[Dict[str, str]]
    code_snippets: Annotated[List[Dict], operator.add]

    #Changelog data
    changelog_content: Optional[str]

    # Agent output
    research_summary: Optional[str]
    tutorial_draft: Optional[str]
    critique_feedback: Optional[str]

    # Critique validation
    validation_passed: Optional[bool]
    issues_found: List[str]
    next_action: Optional[Literal['approve', 'revise']]

    # Control flow
    current_agent: Literal['version_agent','research_agent','writer_agent','critique_agent','chat_agent']
    iteration_count: int
    max_iterations: int
    is_complete: bool
    errors: Annotated[List[str], operator.add]

    #Final output
    final_markdown: Optional[str]
    metadata: Dict[str, Any]

    # Chat / session
    chat_history: List[Dict[str, str]]  # Last N messages [{role, content}, ...]
    qa_response: Optional[str]
    user_intent: Optional[Literal['generate_tutorial', 'ask_question', 'unclear']]
    session_mode: Optional[Literal['generating', 'interactive']]

    