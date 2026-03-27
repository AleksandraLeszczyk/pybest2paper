import logging
import os

from langchain_core.messages import HumanMessage, AIMessage
from piai.viz_creator import viz_creator, figure_registry

logger = logging.getLogger()


def test_viz_creator():
    messages = viz_creator.invoke(
        {"messages": [HumanMessage("Vizualize tuples of height, weight, group values: (1,2,a), (2,4,b), (3,6,a).")]}
    )
    logger.info("Viz Creator answered: %s" % messages)
    answer = messages["messages"][-1]
    assert isinstance(answer, AIMessage)
    assert len(answer.content) > 1
    figures = [i for i in os.listdir("artifacts") if i.startswith("fig") and i.endswith("png")]
    assert len(figures)
