import json
import os
from collections import defaultdict
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import requests
import sseclient
import streamlit as st

from models import (
    ChartEventData,
    DataAgentRunRequest,
    ErrorEventData,
    Message,
    MessageContentItem,
    StatusEventData,
    TableEventData,
    TextContentItem,
    TextDeltaEventData,
    ThinkingDeltaEventData,
    ThinkingEventData,
    ToolResultEventData,
    ToolUseEventData,
)

# Configuration from secrets.toml (or fallback to environment variables)
# Reference: https://docs.streamlit.io/develop/concepts/connections/secrets-management
def get_config(key: str, default: str = "") -> str:
    """Get configuration from st.secrets or environment variables."""
    try:
        return st.secrets["cortex_agent"].get(key, default)
    except (KeyError, FileNotFoundError):
        # Fallback to environment variables for backward compatibility
        env_key = f"CORTEX_AGENT_DEMO_{key}"
        return os.getenv(env_key, default)

PAT = get_config("PAT")
HOST = get_config("HOST")
DATABASE = get_config("DATABASE", "SNOWFLAKE_INTELLIGENCE")
SCHEMA = get_config("SCHEMA", "AGENTS")
AGENT = get_config("AGENT", "SALES_INTELLIGENCE_AGENT")
ORIGIN_APPLICATION = get_config("ORIGIN_APPLICATION", "cortex_demo")


def create_thread() -> Optional[int]:
    """
    Create a new thread using the Threads REST API.
    Reference: https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-agents-threads-rest-api
    
    Returns the thread UUID (integer) or None if creation failed.
    """
    try:
        resp = requests.post(
            url=f"https://{HOST}/api/v2/cortex/threads",
            json={"origin_application": ORIGIN_APPLICATION},
            headers={
                "Authorization": f'Bearer {PAT}"',
                "Content-Type": "application/json",
            },
            verify=False,
        )
        if resp.status_code < 400:
            # Response is a JSON object with thread metadata
            response_data = resp.json()
            thread_id = response_data.get("thread_id")
            return int(thread_id) if thread_id else None
        else:
            st.error(f"Failed to create thread: {resp.status_code} - {resp.text}")
            return None
    except Exception as e:
        st.error(f"Error creating thread: {str(e)}")
        return None


def submit_feedback(
    request_id: str,
    positive: bool,
    feedback_message: Optional[str] = None,
) -> bool:
    """
    Submit feedback for a specific agent response using the Feedback REST API.
    Reference: https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-agents-feedback-rest-api
    """
    # Use the current thread_id for feedback, default to 0 if not using threads
    thread_id = st.session_state.get("thread_id", 0)
    
    feedback_body = {
        "orig_request_id": request_id,
        "positive": positive,
        "thread_id": thread_id,
    }
    if feedback_message:
        feedback_body["feedback_message"] = feedback_message

    try:
        resp = requests.post(
            url=f"https://{HOST}/api/v2/databases/{DATABASE}/schemas/{SCHEMA}/agents/{AGENT}:feedback",
            json=feedback_body,
            headers={
                "Authorization": f'Bearer {PAT}"',
                "Content-Type": "application/json",
            },
            verify=False,
        )
        if resp.status_code < 400:
            return True
        else:
            st.error(f"Failed to submit feedback: {resp.status_code} - {resp.text}")
            return False
    except Exception as e:
        st.error(f"Error submitting feedback: {str(e)}")
        return False


def agent_run(thread_id: Optional[int] = None, parent_message_id: Optional[int] = None) -> requests.Response:
    """
    Calls the REST API and returns a streaming client.
    
    Reference: https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-agents-threads
    
    When using threads:
    - thread_id: The thread UUID
    - parent_message_id: 0 for new thread, or last assistant message_id to continue
    - messages: Only the current user message (not full history)
    """
    # Build the request body
    if thread_id is not None and parent_message_id is not None:
        # Using threads - only send the latest user message
        latest_message = st.session_state.messages[-1] if st.session_state.messages else None
        messages_to_send = [latest_message] if latest_message else []
        
        request_body = DataAgentRunRequest(
            model="claude-4-sonnet",
            messages=messages_to_send,
            thread_id=thread_id,
            parent_message_id=parent_message_id,
        )
    else:
        # Not using threads - send full message history
        request_body = DataAgentRunRequest(
            model="claude-4-sonnet",
            messages=st.session_state.messages,
        )
    
    resp = requests.post(
        url=f"https://{HOST}/api/v2/databases/{DATABASE}/schemas/{SCHEMA}/agents/{AGENT}:run",
        data=request_body.to_json(),
        headers={
            "Authorization": f'Bearer {PAT}"',
            "Content-Type": "application/json",
        },
        stream=True,
        verify=False,
    )
    if resp.status_code < 400:
        return resp  # type: ignore
    else:
        raise Exception(f"Failed request with status {resp.status_code}: {resp.text}")


def stream_events(response: requests.Response) -> Tuple[Optional[int], Optional[int]]:
    """
    Stream events from the response and render them.
    
    Returns a tuple of (user_message_id, assistant_message_id) from metadata events.
    Reference: https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-agents-threads#read-the-returned-message-ids
    """
    content = st.container()
    # Content index to container section mapping
    content_map = defaultdict(content.empty)
    # Content index to text buffer
    buffers = defaultdict(str)
    spinner = st.spinner("Waiting for response...")
    spinner.__enter__()
    
    # Track message IDs from metadata events
    user_message_id = None
    assistant_message_id = None

    events = sseclient.SSEClient(response).events()
    for event in events:
        match event.event:
            case "metadata":
                # Capture message IDs for thread continuation
                # Actual format: {"metadata": {"role":"user","message_id":123}}
                # Reference: https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-agents-threads#read-the-returned-message-ids
                try:
                    data = json.loads(event.data)
                    # The metadata is nested inside a "metadata" key
                    metadata = data.get("metadata", data)  # Fallback to data itself if no nested metadata
                    if metadata.get("role") == "user":
                        user_message_id = metadata.get("message_id")
                    elif metadata.get("role") == "assistant":
                        assistant_message_id = metadata.get("message_id")
                except json.JSONDecodeError:
                    pass
            case "response.status":
                spinner.__exit__(None, None, None)
                data = StatusEventData.from_json(event.data)
                spinner = st.spinner(data.message)
                spinner.__enter__()
            case "response.text.delta":
                data = TextDeltaEventData.from_json(event.data)
                buffers[data.content_index] += data.text
                content_map[data.content_index].write(buffers[data.content_index])
            case "response.thinking.delta":
                data = ThinkingDeltaEventData.from_json(event.data)
                buffers[data.content_index] += data.text
                content_map[data.content_index].expander(
                    "Thinking", expanded=True
                ).write(buffers[data.content_index])
            case "response.thinking":
                # Thinking done, close the expander
                data = ThinkingEventData.from_json(event.data)
                content_map[data.content_index].expander("Thinking").write(data.text)
            case "response.tool_use":
                data = ToolUseEventData.from_json(event.data)
                content_map[data.content_index].expander("Tool use").json(data)
            case "response.tool_result":
                data = ToolResultEventData.from_json(event.data)
                content_map[data.content_index].expander("Tool result").json(data)
            case "response.chart":
                data = ChartEventData.from_json(event.data)
                spec = json.loads(data.chart_spec)
                content_map[data.content_index].vega_lite_chart(
                    spec,
                    use_container_width=True,
                )
            case "response.table":
                data = TableEventData.from_json(event.data)
                data_array = np.array(data.result_set.data)
                column_names = [
                    col.name for col in data.result_set.result_set_meta_data.row_type
                ]
                content_map[data.content_index].dataframe(
                    pd.DataFrame(data_array, columns=column_names)
                )
            case "error":
                data = ErrorEventData.from_json(event.data)
                st.error(f"Error: {data.message} (code: {data.code})")
                # Remove last user message, so we can retry from last successful response.
                st.session_state.messages.pop()
                spinner.__exit__(None, None, None)
                return (None, None)
            case "response":
                data = Message.from_json(event.data)
                st.session_state.messages.append(data)
    spinner.__exit__(None, None, None)
    
    return (user_message_id, assistant_message_id)


def render_feedback_ui(request_id: str, message_index: int):
    """Render feedback buttons for a specific response."""
    feedback_key = f"feedback_{message_index}"
    
    # Check if feedback was already submitted for this message
    if feedback_key in st.session_state.get("submitted_feedback", set()):
        st.caption("✓ Feedback submitted")
        return
    
    # Check if feedback form should be shown
    feedback_type = st.session_state.get(f"feedback_type_{message_index}")
    
    if feedback_type is None:
        # Show thumbs up/down buttons
        col1, col2, col3 = st.columns([1, 1, 8])
        
        with col1:
            if st.button("👍", key=f"thumbs_up_{message_index}", help="This response was helpful"):
                st.session_state[f"feedback_type_{message_index}"] = "positive"
                st.rerun()
        
        with col2:
            if st.button("👎", key=f"thumbs_down_{message_index}", help="This response needs improvement"):
                st.session_state[f"feedback_type_{message_index}"] = "negative"
                st.rerun()
    else:
        # Show feedback text input form
        is_positive = feedback_type == "positive"
        emoji = "👍" if is_positive else "👎"
        
        with st.form(key=f"feedback_form_{message_index}"):
            st.write(f"{emoji} Add a comment (max 25 characters):")
            feedback_text = st.text_input(
                "Your feedback:",
                max_chars=25,
                key=f"feedback_text_{message_index}",
                placeholder="Enter feedback...",
            )
            
            col_submit, col_cancel = st.columns(2)
            with col_submit:
                if st.form_submit_button("Submit"):
                    if submit_feedback(
                        request_id,
                        positive=is_positive,
                        feedback_message=feedback_text if feedback_text else None,
                    ):
                        if "submitted_feedback" not in st.session_state:
                            st.session_state.submitted_feedback = set()
                        st.session_state.submitted_feedback.add(feedback_key)
                        del st.session_state[f"feedback_type_{message_index}"]
                        st.toast("Thanks for your feedback!", icon=emoji)
                        st.rerun()
            with col_cancel:
                if st.form_submit_button("Cancel"):
                    del st.session_state[f"feedback_type_{message_index}"]
                    st.rerun()


def save_current_thread_state():
    """Save the current thread's state to thread_history."""
    thread_id = st.session_state.get("thread_id")
    if thread_id and "thread_history" in st.session_state:
        st.session_state.thread_history[thread_id] = {
            "messages": st.session_state.messages.copy() if st.session_state.messages else [],
            "request_ids": st.session_state.request_ids.copy() if st.session_state.get("request_ids") else {},
            "submitted_feedback": st.session_state.submitted_feedback.copy() if st.session_state.get("submitted_feedback") else set(),
            "last_assistant_message_id": st.session_state.get("last_assistant_message_id"),
            "parent_message_id": st.session_state.get("parent_message_id"),
        }


def switch_to_thread(thread_id: int):
    """Switch to a different thread and restore its state."""
    # Save current thread state first
    save_current_thread_state()
    
    # Load the selected thread's state
    thread_data = st.session_state.thread_history.get(thread_id, {})
    st.session_state.thread_id = thread_id
    st.session_state.messages = thread_data.get("messages", []).copy()
    st.session_state.request_ids = thread_data.get("request_ids", {}).copy()
    st.session_state.submitted_feedback = thread_data.get("submitted_feedback", set()).copy()
    st.session_state.last_assistant_message_id = thread_data.get("last_assistant_message_id")
    
    # For continuing conversation, parent_message_id should be the last assistant message ID
    # If no conversation yet (new thread), use 0
    last_asst_id = thread_data.get("last_assistant_message_id")
    if last_asst_id is not None:
        st.session_state.parent_message_id = last_asst_id
    else:
        st.session_state.parent_message_id = 0


def start_new_thread():
    """Start a new thread and reset conversation state."""
    # Create a new thread
    thread_id = create_thread()
    if thread_id:
        st.session_state.thread_id = thread_id
        st.session_state.parent_message_id = 0  # 0 indicates start of thread
        st.session_state.last_assistant_message_id = None
        st.session_state.messages = []
        st.session_state.request_ids = {}
        st.session_state.submitted_feedback = set()
        # Add to thread history
        if "thread_history" not in st.session_state:
            st.session_state.thread_history = {}
        st.session_state.thread_history[thread_id] = {
            "messages": [],
            "request_ids": {},
            "submitted_feedback": set(),
            "last_assistant_message_id": None,
            "parent_message_id": 0,
        }
        st.toast(f"New thread created: {thread_id}", icon="🧵")
        st.rerun()


def process_new_message(prompt: str) -> None:
    message = Message(
        role="user",
        content=[MessageContentItem(TextContentItem(type="text", text=prompt))],
    )
    render_message(message)
    st.session_state.messages.append(message)

    with st.chat_message("assistant"):
        with st.spinner("Sending request..."):
            # Get thread info
            thread_id = st.session_state.get("thread_id")
            
            # Determine parent_message_id for thread continuation
            # Reference: https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-agents-threads#continue-the-conversation
            # - Use 0 for the first message in a thread
            # - Use last assistant message_id for subsequent messages
            last_assistant_id = st.session_state.get("last_assistant_message_id")
            if last_assistant_id is not None:
                parent_message_id = last_assistant_id
            else:
                parent_message_id = 0
            
            response = agent_run(thread_id=thread_id, parent_message_id=parent_message_id)
        
        # Get and store the request_id
        request_id = response.headers.get('X-Snowflake-Request-Id')
        message_index = len(st.session_state.messages)  # Index for the upcoming assistant message
        
        # Store request_id mapped to message index for feedback
        if "request_ids" not in st.session_state:
            st.session_state.request_ids = {}
        st.session_state.request_ids[message_index] = request_id
        
        # Display debug info: thread_id, parent_message_id used for this request
        thread_id_display = st.session_state.get("thread_id", "N/A")
        st.markdown(f"```request_id: {request_id} | thread_id: {thread_id_display} | parent_msg_id: {parent_message_id}```")
        
        # Stream events and capture message IDs
        user_msg_id, assistant_msg_id = stream_events(response)
        
        # Update for next turn: use this assistant's message_id as the parent
        # Reference: https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-agents-threads#continue-the-conversation
        if assistant_msg_id is not None:
            st.session_state.last_assistant_message_id = assistant_msg_id
            st.session_state.parent_message_id = assistant_msg_id
        
        # Save thread state after processing message
        save_current_thread_state()
        
        # Show feedback UI after the response
        if request_id:
            render_feedback_ui(request_id, message_index)


def render_message(msg: Message, message_index: Optional[int] = None):
    with st.chat_message(msg.role):
        for content_item in msg.content:
            match content_item.actual_instance.type:
                case "text":
                    st.markdown(content_item.actual_instance.text)
                case "chart":
                    spec = json.loads(content_item.actual_instance.chart.chart_spec)
                    st.vega_lite_chart(spec, use_container_width=True)
                case "table":
                    data_array = np.array(
                        content_item.actual_instance.table.result_set.data
                    )
                    column_names = [
                        col.name
                        for col in content_item.actual_instance.table.result_set.result_set_meta_data.row_type
                    ]
                    st.dataframe(pd.DataFrame(data_array, columns=column_names))
                case _:
                    st.expander(content_item.actual_instance.type).json(
                        content_item.actual_instance.to_json()
                    )
        
        # Show feedback UI for assistant messages that have a stored request_id
        if msg.role == "assistant" and message_index is not None:
            request_id = st.session_state.get("request_ids", {}).get(message_index)
            if request_id:
                render_feedback_ui(request_id, message_index)


# ============== INITIALIZATION ==============
# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []

if "request_ids" not in st.session_state:
    st.session_state.request_ids = {}

if "submitted_feedback" not in st.session_state:
    st.session_state.submitted_feedback = set()

# Thread history: dict mapping thread_id -> {messages, request_ids, submitted_feedback, last_assistant_message_id, parent_message_id}
if "thread_history" not in st.session_state:
    st.session_state.thread_history = {}

# Auto-create a new thread on first boot
if "thread_id" not in st.session_state or st.session_state.thread_id is None:
    with st.spinner("Initializing new thread..."):
        initial_thread_id = create_thread()
        if initial_thread_id:
            st.session_state.thread_id = initial_thread_id
            st.session_state.parent_message_id = 0
            st.session_state.last_assistant_message_id = None
            st.session_state.messages = []
            st.session_state.request_ids = {}
            st.session_state.submitted_feedback = set()
            # Store in thread history
            st.session_state.thread_history[initial_thread_id] = {
                "messages": [],
                "request_ids": {},
                "submitted_feedback": set(),
                "last_assistant_message_id": None,
                "parent_message_id": 0,
            }


# ============== SIDEBAR ==============
with st.sidebar:
    st.header("Thread Management")
    
    # Thread selector dropdown
    thread_ids = list(st.session_state.thread_history.keys())
    current_thread = st.session_state.get("thread_id")
    
    if thread_ids:
        # Create display labels for threads
        thread_options = {tid: f"Thread {tid}" for tid in thread_ids}
        
        # Find current index
        current_index = thread_ids.index(current_thread) if current_thread in thread_ids else 0
        
        selected_thread = st.selectbox(
            "Select Thread",
            options=thread_ids,
            index=current_index,
            format_func=lambda x: f"🧵 Thread {x}" + (" (active)" if x == current_thread else ""),
            key="thread_selector"
        )
        
        # Switch thread if selection changed
        if selected_thread != current_thread:
            switch_to_thread(selected_thread)
            st.rerun()
    
    # Display current thread info
    if current_thread:
        st.caption(f"Thread ID: {current_thread}")
        last_msg_id = st.session_state.get("last_assistant_message_id")
        parent_msg_id = st.session_state.get("parent_message_id")
        st.caption(f"Last assistant msg ID: {last_msg_id if last_msg_id else 'None'}")
        st.caption(f"Next parent msg ID: {last_msg_id if last_msg_id else 0}")
        msg_count = len(st.session_state.messages)
        st.caption(f"Messages: {msg_count}")
    
    st.divider()
    
    # New Thread button
    if st.button("➕ New Thread", use_container_width=True, help="Start a new conversation thread"):
        # Save current thread state before creating new
        save_current_thread_state()
        start_new_thread()
    
    # Clear current thread's chat
    if st.button("🗑️ Clear Chat", use_container_width=True, help="Clear messages in current thread"):
        st.session_state.messages = []
        st.session_state.request_ids = {}
        st.session_state.submitted_feedback = set()
        st.session_state.parent_message_id = 0
        st.session_state.last_assistant_message_id = None
        save_current_thread_state()
        st.rerun()
    
    # Show thread count
    st.divider()
    st.caption(f"Total threads: {len(st.session_state.thread_history)}")


# ============== MAIN CONTENT ==============
st.title("Cortex Agent")

# Render existing messages with their message index for feedback
for idx, message in enumerate(st.session_state.messages):
    render_message(message, message_index=idx)

if user_input := st.chat_input("What is your question?"):
    process_new_message(prompt=user_input)
