from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
from dotenv import load_dotenv

load_dotenv()

# Embeddings model
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Sample documents
SAMPLE_DOCS = [
    Document(
        page_content="LangChain is a framework for developing applications powered by language models.",
        metadata={"source": "langchain_docs", "topic": "overview"},
    ),
    Document(
        page_content="LangGraph is a library for building stateful, multi-actor applications with LLMs.",
        metadata={"source": "langgraph_docs", "topic": "overview"},
    ),
    Document(
        page_content="Vector stores are databases optimized for storing and searching embeddings.",
        metadata={"source": "vector_guide", "topic": "database"},
    ),
    Document(
        page_content="RAG combines retrieval with generation for more accurate LLM responses.",
        metadata={"source": "rag_guide", "topic": "architecture"},
    ),
    Document(
        page_content="Embeddings convert text into numerical vectors for semantic similarity.",
        metadata={"source": "embeddings_guide", "topic": "fundamentals"},
    ),
    Document(
        page_content="Chroma is an open-source embedding database for AI applications.",
        metadata={"source": "chroma_docs", "topic": "database"},
    ),
    Document(
        page_content="FAISS is a library for efficient similarity search developed by Facebook.",
        metadata={"source": "faiss_docs", "topic": "database"},
    ),
    Document(
        page_content="Pinecone is a managed vector database service for production workloads.",
        metadata={"source": "pinecone_docs", "topic": "database"},
    ),
]


def similarity_search_with_scores():
    # persistent directory (NOT temp)
    persist_dir = "chroma_db"

    vectorstore = Chroma.from_documents(
        documents=SAMPLE_DOCS,
        embedding=embeddings_model,
        persist_directory=persist_dir
    )

    query = "Explain vector stores."
    results_with_scores = vectorstore.similarity_search_with_score(query, k=3)

    print(f"Top 3 results with scores for query '{query}':")

    # for i, (doc, score) in enumerate(results_with_scores):
    #     similarity = 1 / (1 + score)  # convert distance → similarity
    #     print(
    #         f"Result {i+1}: {doc.page_content} "
    #         f"(Score: {similarity:.4f}, Source: {doc.metadata['source']})"
    #     )

    for i , (doc, score) in enumerate(results_with_scores):
        print(
            f"Result {i+1}: {doc.page_content} "
            f"(Score: {score:.4f}, Source: {doc.metadata['source']})"
        )

def metadata_filtering():
    with tempfile.TemporaryDirectory() as temp_dir:
        vectorstore = Chroma.from_documents(
            documents=SAMPLE_DOCS,
            embedding=embeddings_model,
            persist_directory=temp_dir
        )

        query = "What is LangChain?"
        filter_criteria = {"topic": "overview"}
        results = vectorstore.similarity_search(query, k=5, filter=filter_criteria)

        print(f"Results for query '{query}' with metadata filter {filter_criteria}:")
        for i, doc in enumerate(results):
            print(f"Result {i+1}: {doc.page_content} (Source: {doc.metadata['source']})")



if __name__ == "__main__":
    similarity_search_with_scores()