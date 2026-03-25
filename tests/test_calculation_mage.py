import logging
from langchain_core.messages import HumanMessage, AIMessage
from piai.calculation_mage import calculation_mage

logger = logging.getLogger()


def test_calculation_mage():
    messages = calculation_mage.invoke(
        {"messages": [HumanMessage("Find ground state energy of water.")]}
    )
    logger.info("Calculation Mage answered: %s" % messages)
    answer = messages["messages"][-1]
    assert isinstance(answer, AIMessage)
    assert len(answer.content[-1]["text"]) > 1
    assert "RHF" in answer.content[-1]["text"]