import logging
import os

from langchain.tools import tool
from langchain.agents import create_agent
from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from piai.literature_sage import literature_sage
from piai.calculation_mage import calculation_mage
from piai.utils import parse_event_to_markdown

logger = logging.getLogger()
load_dotenv(override=True)


PI_PROMPT = """You are a principal investigator in a research project in area of quantum chemistry. 
You are a critical thinker who has an eye for detail and do not tolerate errors.
You enjoy exploring new ideas but you always stick to facts and always inform if any thought is not supported by external source.
You finish your analysis with conclusion. Don't suggest further investigation - just do it without user's prompt.
You coordinate a group of experts.
You can assign tasks for a group of experts using tools. This is a list of experts:

**LiteratureSage**
  - best for establishing foundation and planning,
  - has an access to knowledge base,
  - finds relevant publications, and based on his findings, writes a summary of past research and provides known molecular properties and geometries,
  - suggest computation methodology for a current research task but does not know tools that are available now.

**CalculationMage**
 - makes calculations and provides actual results using PyBEST quantum chemistry library,
 - knows available computational tools and libraries best,
 - returns code and its output analysis.

Your job is:
1. Create step-by-step plan for research project. If you are not sure if your idea for a task is feasible, you can ask any expert if he can perform it and if he has any suggestions.
2. Assign tasks for experts. Review results of each task before you do next step. If the results are not satisfying, you modify task and assign a task once again with modified requirements.
3. Write all the assumptions and reasoning that you make as you go with research project.
4a. If calculations were performed successfuly: Prepare a final answer that has a structure of scientific publication that contains abstract, introduction, theory, computational details, results, conclusions, and references.
4b. If calculations were not performed successfully: Prepare a final answer that contains project description, research plan, theoretical background, necessary code snippets and list of further requirements to progress.

Never answer with question.
Good luck! Here is a research project you are assigned with:
"""


@tool
def LiteratureSage(question: str) -> dict:
    """Search for information."""
    logger.info("Asking Literature Sage: %s" % question)
    return literature_sage.invoke(
        {"messages": [HumanMessage(question)]}
    )["messages"][-1].content


@tool
def CalculationMage(question: str) -> list[str]:
    """Write and execute code."""
    logger.info("Asking Code Mage: %s" % question)
    return calculation_mage.invoke(
        {"messages": [HumanMessage(question)]}
    )["messages"][-1].content[-1]["text"]


model_principal_investigator = ChatOpenAI(
    model=os.environ["MODEL_PRINCIPAL_INVESTIGATOR"]
)
principal_investigator = create_agent(
    model_principal_investigator,
    tools=[
        LiteratureSage,
        CalculationMage,
    ],
    system_prompt=SystemMessage(
        content=[{ "type": "text", "text": PI_PROMPT}]
    ),
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


def chat_with_principal_investigator(history: list[dict]):
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
    event_accumulator = "### 🔬 Research Progress\n"
    full_answer = ""

    # Pass the full message history instead of just the last message
    for chunk in principal_investigator.stream(
        {"messages": langchain_messages},
        stream_mode="values"
    ):
        # 1. Update the Event Log (Right Panel)
        new_event_text = parse_event_to_markdown(chunk["messages"][-1])
        event_accumulator += f"\n{new_event_text}"

        # 2. Update the Final Answer (if available in the current chunk)
        if "messages" in chunk:
            last_msg = chunk["messages"][-1]
            if isinstance(last_msg, AIMessage):
                full_answer = last_msg.content

        # Yield current state to the UI
        temp_history = history + [{"role": "assistant", "content": full_answer}]
        yield temp_history, event_accumulator

    # Final yield to lock in the result
    history.append({"role": "assistant", "content": full_answer})
    yield history, event_accumulator