import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from langchain_community.document_loaders import PyPDFLoader


loader = PyPDFLoader("data/2023_Canadian_federal_budget.pdf")
documents = loader.load()

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings

vectorstore = Chroma.from_documents(documents, embedding=OpenAIEmbeddings())

result = vectorstore.similarity_search("What did the author do growing up?")
print(result)