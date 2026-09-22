import os
import streamlit as st

from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Load environment variables
load_dotenv()


# Page configuration
st.set_page_config(
    page_title="KnowledgeLens AI",
    page_icon="📚"
)



st.markdown("""
<style>

.main {
    background-color: #f8fafc;
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
}

h1 {
    font-size: 42px;
    font-weight: 700;
}

h2 {
    font-size: 28px;
    font-weight: 600;
}

[data-testid="stSidebar"] {
    background-color: #f1f5f9;
}

[data-testid="stFileUploader"] {
    border: 2px dashed #94a3b8;
    border-radius: 12px;
    padding: 10px;
}

.stButton > button {
    width: 100%;
    border-radius: 10px;
    height: 45px;
    font-weight: 600;
}

.document-card {
    padding: 20px;
    border-radius: 14px;
    background-color: white;
    border: 1px solid #e2e8f0;
    margin-bottom: 20px;
}

.status-card {
    padding: 15px;
    border-radius: 12px;
    background-color: #ecfdf5;
    border: 1px solid #a7f3d0;
    margin-top: 15px;
}

.source-card {
    padding: 10px 15px;
    border-radius: 8px;
    background-color: #f8fafc;
    border: 1px solid #e2e8f0;
    margin-top: 5px;
}

</style>
""", unsafe_allow_html=True)

# Header

st.markdown(
    """
    <h1>📚 KnowledgeLens AI</h1>
    <p style="font-size:18px;color:#64748b;">
        Your intelligent document assistant
    </p>
    """,
    unsafe_allow_html=True
)


# AI status

col1, col2 = st.columns([5, 1])

with col2:
    st.success("● AI Ready")

# Create embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)


# Create Chroma database
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)


# Sidebar
st.sidebar.header("📄 Upload Document")

uploaded_file = st.sidebar.file_uploader(
    "Choose a PDF file",
    type=["pdf"]
)


# Process uploaded PDF
if uploaded_file is not None:

    os.makedirs("documents", exist_ok=True)

    file_path = os.path.join(
        "documents",
        uploaded_file.name
    )

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.sidebar.success(
        f"Uploaded: {uploaded_file.name}"
    )

    if st.sidebar.button("Process Document"):

        with st.spinner("Processing document..."):

            # Load PDF
            loader = PyPDFLoader(file_path)

            documents = loader.load()

            # Split PDF into chunks
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            chunks = splitter.split_documents(
                documents
            )

            # Add document ID
            for chunk in chunks:
                chunk.metadata["document_id"] = (
                    uploaded_file.name
                )

            # Store chunks
            vectorstore.add_documents(
                chunks
            )

        # Save current document
        st.session_state["current_document"] = (
            uploaded_file.name
        )

        st.sidebar.success(
            f"Processed {len(chunks)} chunks!"
        )

# Get current document

current_document = st.session_state.get(
    "current_document"
)

# If document is processed

if current_document:

    st.success(
        f"📄 Current document: {current_document}"
    )

    # Create retriever

    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 4,
            "filter": {
                "document_id": current_document
            }
        }
    )

# Suggested questions

    st.markdown("### 💡 Try asking")

    q1, q2, q3 = st.columns(3)

    with q1:
        st.info("What is this document about?")

    with q2:
        st.info("What are the main topics?")

    with q3:
        st.info("Give me a summary?")

    # Chat input
    question = st.chat_input(
        "Ask a question about this PDF..."
    )

    # User asks a question
    if question:

        # Show question
        with st.chat_message("user"):
            st.write(question)

        # Search document
        with st.spinner("Searching the document..."):
            documents = retriever.invoke(question)

        # Create context
        context = "\n\n".join(
            document.page_content
            for document in documents
        )

        # RAG prompt
        prompt = ChatPromptTemplate.from_template(
            """
You are a helpful document assistant.

Answer the question using ONLY the
information from the uploaded PDF.

If the answer cannot be found in the PDF,
say:

"I couldn't find the answer in the uploaded document."

Do not make up information.

Context:
{context}

Question:
{question}

Answer:
"""
        )

        # Create prompt messages
        messages = prompt.format_messages(
            context=context,
            question=question
        )

        # Create LLM
        llm = ChatOpenAI(
            model="gpt-4.1-mini",
            temperature=0
        )

        # Generate answer
        with st.spinner(
            "Generating answer..."
        ):

            response = llm.invoke(
                messages
            )

        # Display answer
        with st.chat_message("assistant"):
            st.write(response.content)

        # Sources
        st.markdown("### 📚 Sources")

        pages = set()

        for document in documents:

            page = document.metadata.get(
                "page"
            )

            if page is not None:
                pages.add(page + 1)

        for page in sorted(pages):
            st.write(f"📄 Page {page}")

else:

    st.header("👋 Welcome to KnowledgeLens AI")

    st.write(
        "Upload a PDF and let AI help you understand, "
        "search and explore its contents."
    )

    st.subheader("✨ What you can do")

    st.markdown("""
    - 📄 Upload PDF documents
    - 🔍 Search document content
    - 💬 Ask questions in natural language
    - 📚 Get answers with sources
    """)
