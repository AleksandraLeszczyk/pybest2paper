
import logging
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from app.knowledge_db_setup import fetch_documents, create_chunks, create_vector_database

logger = logging.getLogger()

def test_set_db(tmp_path):
    """Tests full workflow of setting up db and retrieving knowledge."""

    logger.info("Fetching documents from knowledge_base.")
    documents = fetch_documents(dir_path="tests")

    logger.info("Splitting documents into chunks.")
    chunks = create_chunks(documents)

    logger.info("Setting up database.")
    test_db_name = tmp_path / "knowledge_test_db"
    create_vector_database(chunks, test_db_name)

    logger.info("Trying to retrieve something from db.")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma(persist_directory=test_db_name, embedding_function=embeddings)
    retriever = vectordb.as_retriever()
    context = retriever.invoke("What is a difference between pCCD and RHF orbitals?", k=2)
    logger.info("Retrieved context: %s" % context)
    assert len(context)==2


