from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import numpy as np

load_dotenv()

embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def basic_embedding():
    """Test basic embedding functionality."""
    text = "what is the capital of France?"
    embedding = embeddings_model.embed_query(text)
    print(f"Embedding for '{text}'")
    print(f"First 5 values: {embedding[:5]}")
    print(f"Vector dimension: {len(embedding)}")
    print(f"Vector norm: {np.linalg.norm(embedding):.4f}")





def batch_embedding():
    """Test batch embedding functionality."""
    texts = [
        "what is the capital of France?",
        "who is the president of the United States?",
        "what is the largest mammal?"
    ]
    embeddings = embeddings_model.embed_documents(texts)
    for i, (text, embedding) in enumerate(zip(texts, embeddings)):
        print(f"\nEmbedding for '{text}'")
        print(f"First 5 values: {embedding[:5]}")
        print(f"Vector dimension: {len(embedding)}")
        print(f"Vector norm: {np.linalg.norm(embedding):.4f}")


def similarity_search():
    
    docs= [
        "python is a programming language",
        "javaScript is used for web development",
        "the capital of France is Paris",
        "the largest mammal is the blue whale"
    ]
    query = "which city is the capital of France?"

    #embedding
    doc_vectors = embeddings_model.embed_documents(docs)
    query_vector = embeddings_model.embed_query(query)


    def cosine_similarity(vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    similarities = [cosine_similarity(query_vector, doc_vector) for doc_vector in doc_vectors]

    # rank docs
    ranked_docs = sorted(zip(docs, similarities), key=lambda x: x[1], reverse=True)

    print(f"\nSimilarity scores for query: '{query} \n'")
    print (" Ranked documents:")
    for doc, score in ranked_docs:
        print(f"  {score:.4f}: {doc}")




if __name__ == "__main__":
    ##similarity_search()
    batch_embedding()