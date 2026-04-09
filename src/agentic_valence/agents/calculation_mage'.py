import httpx
import logging
import os
import re

from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import SystemMessage 
from mcp.server.fastmcp import FastMCP

from pybest.cc import RfpCCSD, RCCSD
from pybest.geminals import ROOpCCD
from pybest.gbasis import (
    compute_cholesky_eri,
    compute_kinetic,
    compute_nuclear,
    compute_nuclear_repulsion,
    compute_overlap,
    get_gobasis,
)
from pybest.linalg import CholeskyLinalgFactory
from pybest.localization import PipekMezey
from pybest.occ_model import AufbauOccModel
from pybest.part import get_mulliken_operators
from pybest.wrappers import RHF

from piai.dev_assets.pybest_example import mock_h2_calculations

logger = logging.getLogger()

# In a real environment, this URL would point to your actual MCP server
MCP_SERVER_URL = os.environ.get("MCP_SERVER_URL", "not_available")
MODEL = os.environ["MODEL_CODE_WRITING"]
CODE_DB_NAME = os.environ["CODE_DB_NAME"]

mcp = FastMCP("ScientificComputation", json_response=True)


PROMPT_CALCULATION_MAGE = """
You are a python developer specialized in quantum chemistry.
You find answers to user's questions by writing code, executing it, and analysing output.
You answer in html format.
You can only use following libraries: PyBEST.

Here are steps that you must follow:
1. **Always** use the `search_code` tool to learn how to use the quantum chemistry libraries BEFORE writing code. 
2. Write code that provides answer to user's question.
3. Execute code using `execute_code_via_mcp` tool or use 'quick_ccsd' if MCP server is not available.
4. Analyse code output. If there are errors, fix them and execute code again.
5a. If there are problems with running code, write a brief summary that contain:
- methodology (e.g. choice of orbital basis, wave function model, Hamiltonian) with reasoning
- all code snippets
- state clearly that calculations could not be run.
5b. If the code run succeded, write a brief summary that contain:
- methodology (e.g. choice of orbital basis, wave function model, Hamiltonian) with reasoning
- all code snippets
- relevant and precise answer to user's question that contains numbers obtained in calculations. Never make up number yourself.
"""


@tool
def execute_code_via_mcp(code: str) -> str:
    """Executes Python code in an isolated MCP server environment.
    Pass the exact python code to run. Returns the stdout or error traceback.
    """
    logger.info("Executing code via MCP.")
    try:
        # Mocking the MCP HTTP payload standard
        payload = {"jsonrpc": "2.0", "method": "execute_code", "params": {"code": code}, "id": 1}
        
        # For demonstration, we mock the HTTP call if the server isn't reachable
        # response = httpx.post(MCP_SERVER_URL, json=payload, timeout=10.0)
        # return response.json().get("result", "Execution failed")

        return mock_h2_calculations()
        
        return f"Successfully sent request to MCP server. MCP server is not enabled. Return code as an output."
    except Exception as e:
        return f"Execution Error: {str(e)}"


@tool
def search_code(query: str) -> list[str]:
    """Search for code snippets and docs."""
    logger.info(f"Searching code: {query}")
    return retriever.invoke(query, k=10)


@tool
def quick_ccsd(molecule_xyz: str, code: str) -> dict:
    """Simplified method for calculations when MCP server is not available.

    Args:
        molecule_xyz (str)
        code (str): code to run

    Returns:
        dict: Contains keys: 'xyz', 'code', 'rhf', 'ccsd'
    """
    with open("mol.xyz", "w") as file:
        file.write(molecule_xyz)

    # get basis
    pattern = r"get_gobasis\s*\(\s*(['\"]?)([^,'\"\s)]+)\1"
    match = re.search(pattern, code)
    if match:
        basis_name =  match.group(2)

    # get the XYZ file from PyBEST's test data directory
    basis = get_gobasis(basis_name, "mol.xyz")
    lf = CholeskyLinalgFactory(basis.nbasis)
    occ_model = AufbauOccModel(basis, ncore=0)
    olp = compute_overlap(basis)

    # Hamiltoniam
    kin = compute_kinetic(basis)
    ne = compute_nuclear(basis)
    eri = compute_cholesky_eri(basis, threshold=1e-8)
    nr = compute_nuclear_repulsion(basis)

    # HF
    orb_a = lf.create_orbital(basis.nbasis)
    hf = RHF(lf, occ_model)
    hf_output = hf(kin, ne, eri, nr, olp, orb_a)

    # Localize orbitals to improve pCCD convergence
    mulliken = get_mulliken_operators(basis)
    loc = PipekMezey(lf, occ_model, mulliken)
    loc(orb_a, "occ")
    loc(orb_a, "virt")

    # CCSD
    ccsd = RCCSD(lf, occ_model)
    ccsd_output = ccsd(hf_output, kin, ne, eri, solver="krylov")

    return {
        "xyz": molecule_xyz,
        "rhf": hf_output,
        "ccsd": ccsd_output,
        "code": code,
    }


embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Chroma(persist_directory=CODE_DB_NAME, embedding_function=embeddings)
retriever = vectordb.as_retriever()
model_calculation_mage = ChatOpenAI(temperature=0, model_name=MODEL)

calculation_mage = create_agent(
    model_calculation_mage,
    tools=[
        search_code,
        quick_ccsd,
        ],
    system_prompt=SystemMessage(
        content=[
            {
                "type": "text",
                "text": PROMPT_CALCULATION_MAGE,
            }
        ]
    ),
)
