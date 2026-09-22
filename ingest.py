from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


# Load environment variables from .env
load_dotenv()


# 1. Load the PDF
loader = PyPDFLoader(
    "documents/RAG_Based_Moodle_Plugin_Example.pdf"
)

documents = loader.load()

print(f"Loaded {len(documents)} pages")


# 2. Split the document into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks")


# 3. Create embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)


# 4. Store the chunks and embeddings in Chroma
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_db"
)

print(f"Stored {len(chunks)} chunks in Chroma")
