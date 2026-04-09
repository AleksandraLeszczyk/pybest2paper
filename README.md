## ⚛️ Quantum-Chemical-Agentic Platform

An end-to-end platform bridging the gap between non-deterministic Large Language Models and deterministic quantum chemistry engines.
This system automates the scientific workflow: from literature retrieval to live simulation and visual analysis.

### 🚀 Features

🤖 **Multi-Agent Orchestration** – Specialized AI agents (Researcher, Chemist, Visualizer) collaborate to break down and solve complex chemical problems.

🔌 **Custom MCP Server** – Standardized, secure tool execution bridging LLMs with scientific engines (like PyBEST) for real-time electronic structure calculations.

📚 **Scientific RAG** – High-precision Retrieval-Augmented Generation to ground LLM reasoning in verified datasets and chemical literature.

📊 **Auto-Visualization** – Automated generation of publication-ready molecular orbital diagrams and energy profiles.

🛰️ **Live Execution Frontend** – Sleek UI featuring a real-time tracking chatbox to monitor agent thought-chains and calculation progress.

### 🛠️ Tech Stack

|Domain | Technologies |
|-------|--------------|
|AI & Orchestration | Python, Multi-Agent Framework (LangChain), RAG, Chroma |
|Scientific Computing | Model Context Protocol (MCP), PyBEST |
|Frontend & UI | Gradio (Real-time progress) |
|Data Viz | Matplotlib, Seaborn |

## How to run

1. Set up environemnt variables. Choose LLM models that you like most for each task. 
This setup is mostly for simple testing, better models are recommended.

```
HF_TOKEN=
OPENAI_API_KEY=
DB_NAME=knowledge_db  # Name of directory for database with papers
CODE_DB_NAME=code_db  # Name of directory for database for code snippets
MODEL_EMBEDDING=all-MiniLM-L6-v2
MODEL_KNOWLEDGE_SUMMARY=gpt-4.1-nano
MODEL_CODE_WRITING=gpt-5.4-mini
MODEL_PRINCIPAL_INVESTIGATOR=gpt-5.4-mini
MODEL_VIZ_CREATOR=gpt-5.4-mini
MCP_SERVER_URL=http://your-mcp-server:8080/execute  
```

2. Set up knowledge database

Populate the directory knowledge_base/books and knowledge_base/articles with your collection and run:

```python src/agentic_valence/scripts/knowledge_db_setup.py```

3. Set up code database

Populate the directory code_db with your libraries (e.g. PyBEST, PySCF, psi4) and run:
```python src/agentic_valence/scripts/code_db_setup.py```


4. Install

MacOS and Linux
```
  uv venv --python 3.12 
  source .venv/bin/activate
  uv pip install .
  python src/agentic_valence/ui.py
```

5. Run user interface
```
  python src/agentic_valence/ui.py
```
