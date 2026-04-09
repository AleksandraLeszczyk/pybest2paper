import logging
import os

import html
import json
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from piai.literature_sage import literature_sage
from piai.calculation_mage import calculation_mage
from piai.viz_creator import viz_creator

logger = logging.getLogger()
load_dotenv(override=True)


PI_PROMPT = """You are a principal investigator in a research project in area of quantum chemistry. 
You are a critical thinker who has an eye for detail and do not tolerate errors.
You enjoy exploring new ideas but you always stick to facts and always inform if any thought is not supported by external source.
You finish your analysis with conclusion. Don't suggest further investigation - just do it without user's prompt.
Your goal is to study the research project by planning, collecting data, and making conclusions based on evidence.
You coordinate a group of experts.
You use tools:

**LiteratureReview**
  - best for establishing foundation and planning,
  - has an access to knowledge base,
  - finds relevant publications, and based on his findings, writes a summary of past research and provides known molecular properties and geometries,
  - suggest computation methodology for a current research task but does not know tools that are available now.

**ScientificComputing**
 - makes calculations and provides actual results using PyBEST quantum chemistry library,
 - knows available computational tools and libraries best,
 - returns code and its output analysis.

**VizCreator**
 - creates plots (they are send directly to user, no need to handle them yourself),
 - requires numerical input data to plot with description of variables and purpose of figure,
 - ask only after you obtain data from other experts.

Your job is:
1. Create step-by-step plan for research project. If you are not sure if your idea for a task is feasible, you can ask any expert if he can perform it and if he has any suggestions.
2. Assign tasks for experts. Review results of each task before you do next step. If the results are not satisfying, you modify task and assign a task once again with modified requirements.
3. Write all the assumptions and reasoning that you make as you go with research project.
4a. If calculations were performed successfuly: Prepare a final answer that has a structure of scientific publication that contains abstract, introduction, theory, computational details, results, conclusions, and references.
4b. If calculations were not performed successfully: Prepare a final answer that contains project description, research plan, theoretical background, necessary code snippets and list of further requirements to progress.

Never answer with question.
All the planning steps, intermediate results, and thinking should be formatted in HTML.
The final response must be formatted as markdown with latex equations enclosed by $$.
Here is a research project you are assigned with:
"""


@tool
def LiteratureReview(question: str) -> dict:
    """Search for information."""
    logger.info("Asking Literature Sage: %s" % question)
    return literature_sage.invoke({"messages": [HumanMessage(question)]})["messages"][
        -1
    ].content


@tool
def ScientificComputing(question: str) -> list[str]:
    """Write and execute code."""
    logger.info("Asking Code Mage: %s" % question)
    return calculation_mage.invoke({"messages": [HumanMessage(question)]})["messages"][
        -1
    ].content


@tool
def VizCreator(question: str) -> list[str]:
    """Creates interactive pictures and saves them to registry."""
    logger.info("Asking VizCreator: %s" % question)
    viz_creator.invoke({"messages": [HumanMessage(question)]})["messages"][-1].content


model_principal_investigator = ChatOpenAI(
    model=os.environ["MODEL_PRINCIPAL_INVESTIGATOR"]
)
principal_investigator = create_agent(
    model_principal_investigator,
    tools=[
        LiteratureReview,
        ScientificComputing,
        VizCreator,
    ],
    system_prompt=SystemMessage(content=[{"type": "text", "text": PI_PROMPT}]),
    debug=True,
)


def format_context(context: list[str | Document]):
    result = ""
    for doc in context:
        result += (
            f"<span style='color: #ff7800;'>Source: {doc.metadata['source']}</span>\n\n"
        )
        result += doc.page_content + "\n\n"
    return result


def chat_with_principal_investigator(
    history: list[dict], research_progress: str = "🔬 Research Progress\n"
):
    """
    Generator function that streams agent events to the Gradio UI.
    Yields: (updated_history, event_markdown_string)
    """
    # Convert full Gradio history to LangChain messages
    langchain_messages = []
    for msg in history:
        content = msg["content"]
        # Handle multimodal content (list with text/files)
        if isinstance(content, list):
            text = content[0]["text"]
        else:
            text = content

        if msg["role"] == "user":
            langchain_messages.append(HumanMessage(content=text))
        elif msg["role"] == "assistant":
            langchain_messages.append(AIMessage(content=text))
        elif msg["role"] == "system":
            langchain_messages.append(SystemMessage(content=text))

    # Initialize accumulation strings
    full_answer = ""

    # Pass the full message history instead of just the last message
    for chunk in principal_investigator.stream(
        {"messages": langchain_messages}, stream_mode="values"
    ):
        # 1. Update the Event Log (Right Panel)

        print("\n\n NEW EVENT \n\n", type(chunk["messages"][-1]), chunk["messages"][-1])
        new_event_text = parse_event_to_html(chunk["messages"][-1])
        research_progress += f"\n{new_event_text}"

        # 2. Update the Final Answer (if available in the current chunk)
        if "messages" in chunk:
            last_msg = chunk["messages"][-1]
            if hasattr(last_msg, "content") and isinstance(last_msg.content, str):
                full_answer = last_msg.content

        # Yield current state to the UI
        # history = history + [{"role": "assistant", "content": full_answer + f"""<iframe src="gradio_api/file/plot.html" width="100%" height="500px"></iframe>"""}]
        yield history, research_progress

    # Final yield to lock in the result
    history.append({"role": "assistant", "content": full_answer})
    print(history)
    yield history, research_progress


## UTILS


def parse_event_to_html(event: AIMessage | ToolMessage) -> str:
    """
    Parses a LangChain event to HTML with dark mode support and overflow protection.
    """
    content = getattr(event, "content", "")
    msg_type = type(event).__name__

    # CSS for responsiveness and theme adaptation
    style_header = """
    <style>
        .event-container { 
            word-wrap: break-word; 
            overflow-wrap: break-word; 
            white-space: normal;
            font-family: sans-serif;
            margin-bottom: 15px;
        }
        .code-block {
            white-space: pre-wrap; 
            word-break: break-all;
            background: rgba(0,0,0,0.05);
            padding: 8px;
            border-radius: 4px;
            font-size: 0.9em;
        }
        @media (prefers-color-scheme: dark) {
            .code-block { background: rgba(255,255,255,0.1); color: #e5e7eb; }
            .tool-call { border-left-color: #60a5fa !important; }
            .tool-response { background: #064e3b !important; border-color: #059669 !important; color: #ecfdf5; }
        }
    </style>
    """

    html_output = "<br>"

    if msg_type == "AIMessage":
        tool_calls = getattr(event, "tool_calls", [])
        if isinstance(content, str):
            if not content.startswith("#"):
                html_output += (
                    f'<div class="event-container tool-call" style="border-left: 4px solid #3b82f6; padding-left: 12px;">'
                    f"<b>🛠️ Action: thinking</b><br>"
                    f'{content}'
                    f"</div>"
                )
        if tool_calls:
            for tc in tool_calls:
                name = html.escape(tc.get("name", "unknown"))
                args = html.escape(json.dumps(tc.get("args", {}), indent=2))

                html_output += (
                    f'<div class="event-container tool-call" style="border-left: 4px solid #3b82f6; padding-left: 12px;">'
                    f"<b>🛠️ Action: calling <code>{name}</code></b><br>"
                    f'<b>Task:</b><pre class="code-block"><code>{args}</code></pre>'
                    f"</div>"
                )


    elif msg_type == "ToolMessage":
        # Escape the name for safety, but allow 'content' to render as raw HTML
        name = html.escape(getattr(event, "name", "Expert"))

        if name == "VizCreator":
            figures = sorted([
                i for i in os.listdir("artifacts")
                if i.startswith("fig") and (i.endswith("html") or i.endswith("png"))
            ])

            html_output = (
                f'<div class="event-container tool-response" style="background: #f0fdf4; border: 1px solid #bbf7d0; padding: 12px; border-radius: 8px; color: #166534;">'
                f'<h4 style="margin: 0 0 8px 0;">🎨 Figure {len(figures)}</h4>'
                f"""<iframe src="gradio_api/file/artifacts/{figures[-1]}" width="100%" height="500px"></iframe>"""
                f"</div>"
            )

        else:
            # Other tool responses
            html_output = (
                f'<div class="event-container tool-response" style="background: #f0fdf4; border: 1px solid #bbf7d0; padding: 12px; border-radius: 8px; color: #166534;">'
                f'<h4 style="margin: 0 0 8px 0;">🧱 <code>{name}</code> results</h4>'
                f'<div style="line-height: 1.5;">{content}</div>'
                f"</div>"
            )

    return f"{style_header}{html_output.strip()}" if html_output else ""
