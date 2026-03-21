import logging
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from app.principal_investigator import principal_investogator

logger = logging.getLogger()


def test_principal_investigator():
    messages = principal_investogator.invoke(
        {"messages": [HumanMessage("What are best orbitals for heavy-element containing molecules?")]}
    )
    used_tool = False # not yet
    for message in messages["messages"]:
        logger.info(str(type(message)) + str(message.content))
        if isinstance(message, ToolMessage):
            used_tool = True
    answer = messages["messages"][-1]
    assert isinstance(answer, AIMessage)
    assert len(answer.content) > 1
    assert used_tool, "Tool must be used."

