from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

documents = [
    Document(page_content="LangChain is a framework for developing applications powered by language models.", metadata={"source": "doc1"}),
    Document(page_content="It supports various LLM providers, built-in prompt templates and chains, memory and state management, integration with vector databases, and agent capabilities for autonomous decision-making.", metadata={"source": "doc2"}),
    Document(page_content="LangChain is open source and free to use.", metadata={"source": "doc3"}),
    Document(page_content='The authentication process requires valid credentials. Use 0Auth2 for secure API access.', metadata={'type': 'auth'}),
    Document(page_content='Router configuration guide: Access the admin panel at 192.168.1.1 to modify settings.', metadata={'type': 'config'}),
    Document(page_content='WCAG 2.1 compliance requires all images to have alt text and sufficient color contrast.', metadata={'type': 'compliance'}),
    Document(page_content='Product SKU-7742X is our flagship router. It supports gigabit speeds and advanced QoS features.', metadata={'type': 'product'}),
    Document(page_content='For network connectivity issues, first check the ethernet cable and router status lights.', metadata={'type': 'troubleshooting'}),
    Document(page_content='Error code E_CONN_REFUSED indicates the server rejected the connection. Check firewall settings.', metadata={'type': 'error'}),
]

print(f'Loaded {len(documents)} documents')

vectorstore = Chroma.from_documents(
    documents,
    embeddings,
    collection_name='hybrid_test'
)

vector_retriever = vectorstore.as_retriever(search_kwargs={'k': 3})
print('Vector retriever ready')

bm25_retriever = BM25Retriever.from_documents(documents, k=3)
print('BM25 retriever ready')

def hybrid_retrieve(query, retrievers, weights, k=3, rrf_k=60):
    """Reciprocal Rank Fusion (RRF) based hybrid retriever"""
    doc_scores = {}
    for retriever, weight in zip(retrievers, weights):
        results = retriever.invoke(query)
        for rank, doc in enumerate(results):
            key = doc.page_content
            rrf_score = weight * (1 / (rank + 1 + rrf_k))
            if key in doc_scores:
                doc_scores[key] = (doc_scores[key][0] + rrf_score, doc)
            else:
                doc_scores[key] = (rrf_score, doc)
    sorted_docs = sorted(doc_scores.values(), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in sorted_docs[:k]]

print('Hybrid retriever ready')

def test_query(query, name, retriever):
    results = retriever.invoke(query)
    print(f'\n{name} - Query: "{query}"')
    for i, doc in enumerate(results[:3]):
        content = getattr(doc, 'page_content', '') or ''
        preview = content[:80] + '...' if len(content) > 80 else content
        print(f'{i+1}. {preview}')
    return results

def test_hybrid_query(query, name, retrievers, weights):
    results = hybrid_retrieve(query, retrievers, weights)
    print(f'\n{name} - Query: "{query}"')
    for i, doc in enumerate(results[:3]):
        content = getattr(doc, 'page_content', '') or ''
        preview = content[:80] + '...' if len(content) > 80 else content
        print(f'{i+1}. {preview}')
    return results

TEST_QUERIES = [
    ('What is LangChain?', 'LangChain'),
    ('How to configure the router?', 'Router'),
    ('How to troubleshoot connectivity issues?', 'Troubleshooting'),
    ('What are the error codes and their meanings?', 'Error Codes'),
    ('SKU-7742X specifications', 'Product SKUs'),
    ('E_CONN_REFUSED error', 'Error Codes'),
]

print("\n" + "="*80)
print("HYBRID SEARCH RESULTS (BM25 + Vector)")
print("="*80)

for query, label in TEST_QUERIES:
    print("\n" + "=" * 60)
    test_hybrid_query(query, "HYBRID (BM25+Vector)", [bm25_retriever, vector_retriever], [0.5, 0.5])