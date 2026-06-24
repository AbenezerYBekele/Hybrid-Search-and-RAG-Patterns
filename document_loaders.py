import os
import tempfile
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import ( TextLoader,
    WebBaseLoader,
    PyPDFLoader,
    DirectoryLoader)

from dotenv import load_dotenv

load_dotenv()

def load_text_file():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
        temp_file.write(b"Sample text content for testing.")
        temp_file_path = temp_file.name
        #load text using textloade
    try:
        loader = TextLoader(temp_file_path)
        documents = loader.load()

        # print info
        print (f"loaded {len(documents)} document (s)")
        print (f"Content of the first document: {documents[0].page_content[:80]}...") 
        print (f"Metadata of the first document: {documents[0].metadata}")
        # Print loaded doc
        for doc in documents:
            print("Document content:", doc.page_content)
    finally:
        os.remove(temp_file_path)

if __name__ == "__main__":
    load_text_file()


def pdf_loader(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print (f"loaded {len(documents)} document (s)")
    for i, doc in enumerate(documents):
        print(f"Content of document {i+1}: {doc.page_content[:80]}...")
        print(f"Metadata of document {i+1}: {doc.metadata}")
    if __name__ == "__main__":
        pdf_loader("./Docs/Associate.pdf")