import logging
import os
import glob
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from dotenv import load_dotenv

logger = logging.getLogger()
load_dotenv(override=True)


def fetch_documents(dir_path: str | Path | None) -> list[Document]:
    """Fetch documents from knowledge_base directory.

    Returns:
        list[Document]: _description_
    """
    if not dir_path:
        dir_path = Path(__file__).parent.parent

    knowledge_base_path = str(Path(dir_path) / "knowledge_base")
    logger.info("Attempting to fetch data from directory %s" % dir_path)
    folders = glob.glob(str(Path(knowledge_base_path) / "*"))
    documents = []
    for folder in folders:
        doc_type = os.path.basename(folder)
        loader = DirectoryLoader(folder, glob="**/*.pdf", loader_cls=PyPDFLoader)
        folder_docs = loader.load()
        for doc in folder_docs:
            doc.metadata["doc_type"] = doc_type
            documents.append(doc)
    logger.info("Success.")
    return documents


def create_chunks(documents: list[Document]) -> list[Document]:
    """Split documents from the list into chunks.

    Args:
        documents (list[Document])

    Returns:
        list[Document]: list of chunks
    """
    logger.info("Splitting %i documents into chunks" % len(documents))
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=300)
    chunks = text_splitter.split_documents(documents)
    logger.info("Obtained %i chunks" % len(chunks))
    return chunks


def create_vector_database(chunks: list[Document], db_name: str|None = None) -> Chroma:
    """Create vector datasetore in location defined by DB_NAME environment variable.

    Args:
        chunks (list[Document]): list of chunks

    Returns:
        Chroma: vector database
    """
    if not db_name:
        db_name = os.environ.get("db_name", "knowledge_db")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    if os.path.exists(db_name):
        Chroma(persist_directory=db_name, embedding_function=embeddings).delete_collection()
        logger.info("Database detected in %s. Deleting old items." % db_name)
    
    logger.info("Creating new database in %s" % db_name)

    vectordb = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=db_name
    )

    collection = vectordb._collection
    count = collection.count()

    sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
    dimensions = len(sample_embedding)
    logger.info(f"Created database with %i %i-dimension vectors." % (count, dimensions))
    return vectordb


if __name__ == "__main__":
    load_dotenv(override=True)
    documents = fetch_documents()
    chunks = create_chunks(documents)
    create_vector_database(chunks)
    print("Ingestion complete")
