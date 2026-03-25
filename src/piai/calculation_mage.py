import httpx
import logging
import os
from pathlib import Path

from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import SystemMessage 

from piai.dev_assets.pybest_example import mock_water_calculations

logger = logging.getLogger()

# In a real environment, this URL would point to your actual MCP server
MCP_SERVER_URL = os.environ.get("MCP_SERVER_URL", "not_available")
MODEL = os.environ["MODEL_CODE_WRITING"]
CODE_DB_NAME = os.environ["CODE_DB_NAME"]


PROMPT_CALCULATION_MAGE = """
You are a python developer specialized in quantum chemistry.
You find answers to user's questions by writing code, executing it, and analysing output.
You can only use following libraries: PyBEST.

Here are steps that you must follow:
1. **Always** use the `search_code` tool to learn how to use the quantum chemistry libraries BEFORE writing code. 
2. Write code that provides answer to user's question.
3. Execute code using `execute_code_via_mcp` tool
4. Analyse code output. If there are errors, fix them and execute code again.
5a. If the code run succeded, write a brief summary that contain:
- methodology (e.g. choice of orbital basis, wave function model, Hamiltonian) with reasoning
- all code snippets
- relevant and precise answer to user's question that contains numbers obtained in calculations. Never make up number yourself.
5b. If there are problems with running code, write a brief summary that contain:
- methodology (e.g. choice of orbital basis, wave function model, Hamiltonian) with reasoning
- all code snippets
- state clearly that calculations could not be run.
"""


@tool
def execute_code_via_mcp(code: str) -> str:
    """
    Executes Python code in an isolated MCP server environment.
    Pass the exact python code to run. Returns the stdout or error traceback.
    """
    logger.info("Executing code via MCP.")
    try:
        # Mocking the MCP HTTP payload standard
        payload = {"jsonrpc": "2.0", "method": "execute_code", "params": {"code": code}, "id": 1}
        
        # For demonstration, we mock the HTTP call if the server isn't reachable
        # response = httpx.post(MCP_SERVER_URL, json=payload, timeout=10.0)
        # return response.json().get("result", "Execution failed")

        return mock_water_calculations()
        
        return f"Successfully sent request to MCP server. MCP server is not enabled. Return code as an output."
    except Exception as e:
        return f"Execution Error: {str(e)}"


@tool
def search_code(query: str) -> list[str]:
    """Search for code snippets and docs."""
    logger.info(f"Searching code: {query}")
    return retriever.invoke(query, k=10)


embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Chroma(persist_directory=CODE_DB_NAME, embedding_function=embeddings)
retriever = vectordb.as_retriever()
model_calculation_mage = ChatOpenAI(temperature=0, model_name=MODEL)

calculation_mage = create_agent(
    model_calculation_mage,
    tools=[execute_code_via_mcp, search_code],
    system_prompt=SystemMessage(
        content=[
            {
                "type": "text",
                "text": PROMPT_CALCULATION_MAGE,
            }
        ]
    ),
)
