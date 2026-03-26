import logging
from langchain_core.messages import HumanMessage, AIMessage
from piai.literature_sage import literature_sage

logger = logging.getLogger()


def test_literature_sage():
    messages = literature_sage.invoke(
        {"messages": [HumanMessage("When pCCD orbitals are better tha RHF orbitals?")]}
    )
    logger.info("Literature Sage answered: %s" % messages)
    answer = messages["messages"][-1]
    assert isinstance(answer, AIMessage)
    assert len(answer.content) > 1
    important_keywords = ["strong", "correlation", "degener", "multireference"]
    found_keywords = []
    for i in important_keywords:
        if i in answer.content:
            found_keywords.append(i)
    assert (
        len(found_keywords) > 1
    ), f"Found insufficient number of keywords: {found_keywords}"
