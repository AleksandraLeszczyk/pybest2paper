from typing import Any

import json


def parse_event_to_markdown(event: Any) -> str:
    """
    Parses a single LangChain message event into a clean Markdown string.
    """
    # Check if the event is a dictionary (common in JSON API contexts)
    # or an object with a .content attribute
    content = getattr(event, 'content', '')
    msg_type = type(event).__name__

    md_output = ""

    if msg_type == "AIMessage":
        
        # Handle Tool Calls
        tool_calls = getattr(event, 'tool_calls', [])
        if tool_calls:
            # md_output += "\n\n**🛠️ Action: Calling Expert**\n"
            for tc in tool_calls:
                args = json.dumps(tc.get('args', {}), indent=2)
                md_output += f"- **🛠️ Action: calling expert** `{tc.get('name')}`<br> **Task:**\n{args}\n\n"

    elif msg_type == "ToolMessage":
        name = getattr(event, 'name', 'Expert')
        md_output = f"#### 🧱 Expert `{name}` answers: <br>{content}\n"

    return md_output.strip()

