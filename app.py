import chromadb
chroma_client = chromadb.Client()

collection_name = "test_collection"
collection = chroma_client.get_or_create_collection(name=collection_name)

document_id = [{ "id": "doc1", "text": "This is the first document."},
               { "id": "doc2", "text": "This is the second document."},
               { "id": "doc3", "text": "This is the third document."}
               ]

for doc in document_id:
    collection.upsert(
        ids=doc["id"],
        documents=doc["text"]
    )

query_texts = "This is the first document."

results = collection.query(
    query_texts=[query_texts],
    n_results=3
)

print(results)