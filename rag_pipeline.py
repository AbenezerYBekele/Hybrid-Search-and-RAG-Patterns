from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Knowledge base
KNOWLEDGE_BASE = """LangChain is a framework for developing applications powered by language models.

Key Features:
- Supports various LLM providers (OpenAI, Anthropic, Groq, etc.)
- Built-in prompt templates and chains
- Memory and state management
- Integration with vector databases
- Agent capabilities for autonomous decision-making

Getting Started:
Install with: pip install langchain langchain-groq
Create your first chain in minutes.

Pricing: LangChain is open source and free to use."""


def create_kb():
    """Create a knowledge base by loading documents"""
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    doc = Document(
        page_content=KNOWLEDGE_BASE,
        metadata={"source": "knowledge_base"}
    )
    chunks = splitter.split_documents([doc])

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings_model,
        collection_name="knowledge_base"
    )
    return vector_store


def get_llm(temperature=0.3):
    """Return a working Groq LLM, trying models in order until one succeeds."""
    models = [
        "llama-3.3-70b-versatile",
        "llama3-70b-8192",
        "llama-3.1-8b-instant",
        "llama3-8b-8192",
    ]
    for model in models:
        try:
            llm = ChatGroq(model=model, temperature=temperature)
            llm.invoke("hi")  # quick availability check
            print(f"Using model: {model}\n")
            return llm
        except Exception:
            continue
    raise RuntimeError(
        "No working Groq model found. "
        "Check https://console.groq.com/docs/models for current models."
    )


def basic_rag():
    vector_store = create_kb()
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    llm = get_llm(temperature=0.3)

    prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context. If you don't know the answer, say you don't know.

Context:
{context}

Question: {question}

Answer:""")

    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    questions = [
        "What is LangChain?",
        "who created LangChain?",
        "What are the key features of LangChain?",
        "Who is the president of the United States?",
    ]

    print("Basic RAG Demo:\n")
    for q in questions:
        answer = rag_chain.invoke(q)
        print(f"Question: {q}")
        print(f"Answer: {answer}\n")


if __name__ == "__main__":
    basic_rag()