from dotenv import load_dotenv
load_dotenv()

from langchain_core import __version__ as core_version
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
import importlib.metadata

lg_version = importlib.metadata.version("langgraph")

print("langchain-core version:", core_version)
print("langgraph version:", lg_version)

def main():
    # Groq test (FREE)
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    print(llm.invoke("say 'setup complete!' in one word"))

    # Gemini test (FREE)
    gemini_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    print(gemini_llm.invoke("say 'setup complete!' in one word"))

print("setup complete!")

if __name__ == "__main__":
    main()
