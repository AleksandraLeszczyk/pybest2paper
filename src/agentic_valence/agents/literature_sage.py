import os

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage 
from langchain_core.documents import Document

from dotenv import load_dotenv


load_dotenv(override=True)

PROMPT_LITERATURE_SAGE = """
You are a knowledgeable scientist specialized in quantum chemistry with access to the datastore.
You are a critical thinker who has an eye for detail and do not tolerate errors or lying. If you are not sure, you don't answer.
You answer must be formatted as HTML.
You answer questions related to quantum chemistry based on the sources.
"""

MODEL = os.environ["MODEL_KNOWLEDGE_SUMMARY"]
DB_NAME = os.environ["DB_NAME"]

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
retriever = vectordb.as_retriever()
model_literature_sage = ChatOpenAI(temperature=0, model_name=MODEL)

@tool
def search(query: str) -> str:
    """Search for information."""
    return format_docs(retriever.invoke(query, k=10))

literature_sage = create_agent(
    model_literature_sage,
    tools=[search],
    system_prompt=SystemMessage(
        content=[
            {
                "type": "text",
                "text": PROMPT_LITERATURE_SAGE,
            }
        ]
    ),
)

def format_docs(context: list[str | Document]):
    result = ''
    for doc in context:
        result += (
            f"<span style='color: #ff7800;'>Source: {doc.metadata['source']}</span>\n\n"
        )
        result += doc.page_content + "\n\n"
    return result



