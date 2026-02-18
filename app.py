"""
DevAgent Lab — Streamlit Chat Interface
Run with: streamlit run app.py
"""
import streamlit as st
import asyncio
import os
from dotenv import load_dotenv
from config.logger import setup_logger
from config.langfuse_config import get_langfuse_config, flush_langfuse_traces


load_dotenv()

# ── Page Config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DevAgent Lab",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

*, html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(160deg, #0a0a0f 0%, #121220 40%, #0d1117 100%); }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; }

/* ── Landing page ── */
.landing-container {
    display: flex; flex-direction: column; align-items: center;
    justify-content: center; min-height: 70vh; text-align: center;
    animation: fadeIn 0.8s ease-out;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}
.logo-placeholder {
    width: 90px; height: 90px; border-radius: 22px;
    background: linear-gradient(135deg, #667eea, #764ba2);
    display: flex; align-items: center; justify-content: center;
    font-size: 40px; margin-bottom: 1.2rem;
    box-shadow: 0 8px 32px rgba(102,126,234,0.30);
}
.brand-title {
    font-size: 2.8rem; font-weight: 800;
    background: linear-gradient(135deg, #667eea 0%, #a78bfa 50%, #f093fb 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem; letter-spacing: -1px;
}
.brand-subtitle {
    color: #8b8fa3; font-size: 1.05rem; font-weight: 400;
    margin-bottom: 2.5rem; max-width: 480px;
}

/* ── Chat messages ── */
.chat-msg {
    display: flex; gap: 12px; margin-bottom: 1rem;
    animation: msgSlide 0.35s ease-out;
}
@keyframes msgSlide {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
.chat-msg.user { flex-direction: row-reverse; }
.chat-msg.user .msg-bubble {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: #fff; border-radius: 18px 18px 4px 18px; max-width: 75%;
}
.chat-msg.agent .msg-bubble {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.08);
    color: #e1e4eb; border-radius: 18px 18px 18px 4px; max-width: 75%;
}
.msg-bubble { padding: 12px 18px; font-size: 0.92rem; line-height: 1.55; }
.msg-avatar {
    width: 34px; height: 34px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 16px; flex-shrink: 0; margin-top: 2px;
}
.msg-avatar.user-av  { background: linear-gradient(135deg, #667eea, #764ba2); }
.msg-avatar.agent-av { background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.12); }

/* ── Live progress ("thinking") ── */
.thinking-box {
    background: rgba(102,126,234,0.06);
    border: 1px solid rgba(102,126,234,0.15);
    border-radius: 14px; padding: 14px 18px; margin: 0.8rem 0;
}
.thinking-header {
    color: #a78bfa; font-weight: 600; font-size: 0.88rem;
    margin-bottom: 10px; display: flex; align-items: center; gap: 6px;
}
.step-item {
    display: flex; align-items: center; gap: 8px;
    padding: 5px 0; font-size: 0.84rem; color: #8b8fa3;
}
.step-item.done   { color: #6ee7b7; }
.step-item.active { color: #e1e4eb; font-weight: 500; }
.step-dot {
    width: 8px; height: 8px; background: #667eea;
    border-radius: 50%; animation: pulse 1.2s infinite; flex-shrink: 0;
}
@keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:0.3; } }

/* ── Collapsed thinking (after completion) ── */
.thinking-collapsed {
    background: rgba(102,126,234,0.04);
    border: 1px solid rgba(102,126,234,0.10);
    border-radius: 12px; margin: 0.6rem 0; overflow: hidden;
}
.thinking-collapsed summary {
    padding: 10px 16px; color: #8b8fa3; font-size: 0.82rem;
    cursor: pointer; display: flex; align-items: center; gap: 6px;
    list-style: none; user-select: none;
}
.thinking-collapsed summary::-webkit-details-marker { display: none; }
.thinking-collapsed summary::before {
    content: '▸'; font-size: 10px; transition: transform 0.2s;
}
.thinking-collapsed[open] summary::before { transform: rotate(90deg); }
.thinking-collapsed .steps-inner { padding: 0 16px 12px 16px; }

/* ── Streamlit input overrides ── */
.stChatInput > div {
    border-radius: 16px !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    background: rgba(255,255,255,0.04) !important;
}
.stChatInput textarea { color: #e1e4eb !important; }
</style>
""", unsafe_allow_html=True)


# ── Session state init ───────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_started" not in st.session_state:
    st.session_state.chat_started = False
if "processing" not in st.session_state:
    st.session_state.processing = False
# Persist key workflow results across follow-up messages
if "last_result" not in st.session_state:
    st.session_state.last_result = {}

logger = setup_logger(__name__)


# ── Constants ────────────────────────────────────────────────────────────
STEP_LABELS = {
    "chat_agent":     ("💬", "Understanding your request"),
    "version_agent":  ("🔍", "Detecting language & fetching version"),
    "research_agent": ("📚", "Researching docs, GitHub & code snippets"),
    "writer_agent":   ("✍️", "Writing tutorial draft"),
    "critique_agent": ("🧐", "Validating & reviewing tutorial"),
}


# ── Helper: build initial AgentState ─────────────────────────────────────
def build_initial_state(user_query: str) -> dict:
    # Get last 5 messages for conversation memory
    recent = st.session_state.messages[-10:]  # last 5 pairs (user+agent)
    chat_history = [
        {"role": m["role"], "content": m["content"]}
        for m in recent
        if m["role"] in ("user", "agent")
    ][-10:]  # cap at 10 entries

    # Carry over key results from previous runs
    prev = st.session_state.last_result

    return {
        "lib_name": prev.get("lib_name", ""),
        "user_query": user_query,
        "target_language": prev.get("target_language"),
        "target_version": prev.get("target_version"),
        "version_status": "pending",
        "package_manager": prev.get("package_manager"),
        "release_date": prev.get("release_date"),
        "repository_url": prev.get("repository_url"),
        "docs_url": prev.get("docs_url"),
        "doc_content": None,
        "doc_index_id": prev.get("doc_index_id"),
        "github_repos": [],
        "code_snippets": [],
        "changelog_content": None,
        "research_summary": prev.get("research_summary"),
        "tutorial_draft": None,
        "critique_feedback": None,
        "validation_passed": None,
        "issues_found": [],
        "next_action": None,
        "current_agent": "chat_agent",
        "iteration_count": 0,
        "max_iterations": 3,
        "is_complete": False,
        "errors": [],
        "final_markdown": prev.get("final_markdown"),
        "metadata": prev.get("metadata", {}),
        "chat_history": chat_history,
        "qa_response": None,
        "user_intent": None,
        "session_mode": prev.get("session_mode"),
    }


# ── Helper: run workflow with live streaming ─────────────────────────────
async def run_workflow_streaming(user_query: str, progress_container):
    """Stream workflow execution and update progress in real-time."""
    from graphs.workflow import build_devagent_workflow
    from config.langfuse_config import get_langfuse_config

    app = build_devagent_workflow()
    initial_state = build_initial_state(user_query)

    completed_steps = []
    current_step = None
    final_result = None

    async for event in app.astream(initial_state, config=get_langfuse_config(), stream_mode="updates"):
        for node_name, node_output in event.items():
            if node_name == "__end__":
                continue

            # Mark previous step as done
            if current_step and current_step != node_name:
                completed_steps.append(current_step)
            current_step = node_name

            # Render live progress
            with progress_container:
                _render_progress(completed_steps, current_step)

            final_result = node_output

    # Mark last step as done
    if current_step:
        completed_steps.append(current_step)

    return final_result, completed_steps


def _render_progress(completed_steps, current_step):
    """Render the live step-by-step progress."""
    lines = []
    for step in completed_steps:
        _, label = STEP_LABELS.get(step, ("⚙️", step))
        lines.append(f'<div class="step-item done"><span>✅</span> {label}</div>')
    if current_step:
        _, label = STEP_LABELS.get(current_step, ("⚙️", current_step))
        lines.append(f'<div class="step-item active"><span class="step-dot"></span> {label}</div>')

    st.markdown(f"""
    <div class="thinking-box">
        <div class="thinking-header">⚡ Working on it…</div>
        {"".join(lines)}
    </div>
    """, unsafe_allow_html=True)


def get_agent_response(result: dict) -> str:
    """Extract the best response from the workflow result."""
    if result.get("qa_response"):
        return result["qa_response"]
    if result.get("final_markdown"):
        return result["final_markdown"]
    if result.get("tutorial_draft"):
        return result["tutorial_draft"]
    if result.get("errors"):
        return "⚠️ Something went wrong:\n" + "\n".join(f"• {e}" for e in result["errors"])
    return "I couldn't process that request. Could you try rephrasing?"


def render_message(role: str, content: str):
    if role == "user":
        st.markdown(f"""
        <div class="chat-msg user">
            <div class="msg-avatar user-av">👤</div>
            <div class="msg-bubble">{content}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-msg agent">
            <div class="msg-avatar agent-av">🤖</div>
            <div class="msg-bubble">{content}</div>
        </div>
        """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════
# MAIN UI
# ═════════════════════════════════════════════════════════════════════════

if not st.session_state.chat_started:
    # ── LANDING PAGE ─────────────────────────────────────────────────────
    st.markdown("""
    <div class="landing-container">
        <div class="logo-placeholder">🤖</div>
        <div class="brand-title">DevAgent</div>
        <div class="brand-subtitle">
            AI-powered tutorial generator. Ask me to create a getting-started
            guide for any library, or ask a coding question.
        </div>
    </div>
    """, unsafe_allow_html=True)

    prompt = st.chat_input("Ask DevAgent anything…", key="landing_input")
    if prompt:
        st.session_state.chat_started = True
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.processing = True
        st.rerun()

else:
    # ── CHAT INTERFACE ───────────────────────────────────────────────────
    # Header
    st.markdown("""
    <div style="display:flex;align-items:center;gap:12px;padding:0.5rem 0 1.2rem 0;border-bottom:1px solid rgba(255,255,255,0.06);margin-bottom:1rem;">
        <div style="width:36px;height:36px;border-radius:10px;background:linear-gradient(135deg,#667eea,#764ba2);display:flex;align-items:center;justify-content:center;font-size:18px;">🤖</div>
        <div>
            <div style="color:#e1e4eb;font-weight:600;font-size:1rem;">DevAgent</div>
            <div style="color:#6b7280;font-size:0.75rem;">AI Tutorial Generator</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Render existing messages
    for msg in st.session_state.messages:
        if msg["role"] == "agent":
            # Show collapsed thinking steps if available
            if msg.get("steps"):
                steps_html = ""
                for step in msg["steps"]:
                    _, label = STEP_LABELS.get(step, ("⚙️", step))
                    steps_html += f'<div class="step-item done"><span>✅</span> {label}</div>'
                st.markdown(f"""
                <details class="thinking-collapsed">
                    <summary>⚡ Worked through {len(msg["steps"])} steps</summary>
                    <div class="steps-inner">{steps_html}</div>
                </details>
                """, unsafe_allow_html=True)

            if msg.get("is_markdown"):
                st.markdown(msg["content"])
            else:
                render_message("agent", msg["content"])
        else:
            render_message(msg["role"], msg["content"])

    # Process pending request
    if st.session_state.processing:
        progress_area = st.empty()
        last_user_msg = st.session_state.messages[-1]["content"]

        try:
            result, steps = asyncio.run(
                run_workflow_streaming(last_user_msg, progress_area)
            )
            progress_area.empty()

            response = get_agent_response(result)
            is_long_markdown = len(response) > 500 and response.startswith("#")
            st.session_state.messages.append({
                "role": "agent",
                "content": response,
                "is_markdown": is_long_markdown,
                "steps": steps,
            })
            # Persist key results for follow-up memory
            st.session_state.last_result = {
                "lib_name": result.get("lib_name", ""),
                "target_language": result.get("target_language"),
                "target_version": result.get("target_version"),
                "package_manager": result.get("package_manager"),
                "release_date": result.get("release_date"),
                "repository_url": result.get("repository_url"),
                "docs_url": result.get("docs_url"),
                "doc_index_id": result.get("doc_index_id"),
                "research_summary": result.get("research_summary"),
                "final_markdown": result.get("final_markdown"),
                "metadata": result.get("metadata", {}),
                "session_mode": result.get("session_mode"),
            }
        except Exception as e:
            progress_area.empty()
            logger.error(f"Workflow failed: {e}")
            flush_langfuse_traces()
            st.session_state.messages.append({
                "role": "agent",
                "content": f"❌ Error: {str(e)}",
                "is_markdown": False,
                "steps": [],
            })

        st.session_state.processing = False
        st.rerun()

    # Chat input for follow-ups
    follow_up = st.chat_input("Type a message…", key="chat_input")
    if follow_up:
        st.session_state.messages.append({"role": "user", "content": follow_up})
        st.session_state.processing = True
        st.rerun()
