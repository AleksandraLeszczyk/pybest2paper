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

MacOS and Linux
```
  uv venv --python 3.12 
  source .venv/bin/activate
  uv pip install -r requirements.txt 
  uv pip install .
  python src/piai/ui.py
```
