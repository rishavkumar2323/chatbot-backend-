# --------------------------------------------------------------------------
# Streamlit Community Cloud ships an old system sqlite3 (< 3.35), but
# chromadb needs a newer one. This swaps in pysqlite3-binary's sqlite
# before chromadb (via langchain_chroma) gets imported. Safe to keep for
# local runs too — it silently no-ops if pysqlite3 isn't installed.
# --------------------------------------------------------------------------
try:
    __import__("pysqlite3")
    import sys as _sys
    _sys.modules["sqlite3"] = _sys.modules.pop("pysqlite3")
except ImportError:
    pass

import os
import shutil

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
st.set_page_config(page_title="Z1 Super App Assistant", page_icon="🤖", layout="wide")

DATA_DIR = "data"
PDF_PATH = os.path.join(DATA_DIR, "sets_of_miniapp.pdf")
PERSIST_DIR = "./chroma_db_bge"
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
# NOTE: switched from bge-base to bge-small to fit Streamlit Community
# Cloud's ~1GB RAM limit reliably. If you deploy somewhere with more RAM
# (Render, Railway, your own VPS), you can switch back to
# "BAAI/bge-base-en-v1.5" for slightly better retrieval quality.
GROQ_MODEL_NAME = "llama-3.3-70b-versatile"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 10

PROMPT_TEMPLATE = """
You are an AI Assistant for the Z1 Super App.

Your task is to answer ONLY from the provided context.

Rules:
1. Use only the given context.
2. Do not make up information.
3. If the answer is clearly available in the context, answer it.
If the answer is not available in the context, reply:
"I could not find this information in the provided documents."
4. If the question is unrelated to the Z1 Super App or its Mini Apps, reply:
   "This is not the correct question. Please ask a question related to the Z1 Super App or its Mini Apps."
5. Keep answers clear and well-structured.
6. Use bullet points whenever appropriate.

Examples:

Example 1
Question:
What is the First Look mini app?

Answer:
The First Look mini app allows users to preview upcoming episodes, trailers, or program segments before they are released to the public.
Key features include:
- The Vault
- ACR Redemption
- NDA Check-In
- Community Reaction Hub

Example 2
Question:
What are the stakeholder benefits of the First Look mini app?

Answer:
Stakeholder Benefits:

Z1 Users
Key Benefits
Exclusivity: Provides social currency and "bragging rights."
Utility: Gives a tangible, high-value way to spend earned ACRs.

Content Creators
Early Feedback: Offers a controlled environment to gauge
audience sentiment before a wide release.
Marketing Momentum: Creates a core group of "evangelists"
who generate organic hype.

Z1 Platform
Retention: Increases daily active usage as members check for
new "drops."
Economy Stability: Circulates and burns ACRs, maintaining
the health of the app's internal economy.

Advertisers/Sponsors
High-Intent Audience: Provides access to the most engaged
segments of the user base for targeted brand placements or
"Presented by" credits.

Example 3
Question:
What is the In-Episode Easter Egg Hunt?

Answer:
The In-Episode Easter Egg Hunt is a mini app that lets users search for hidden symbols or objects while watching newly released episodes. Users who find them first can earn Z1 jackpot rewards.

Example 4
Question:
Who is the CEO of Microsoft?

Answer:
Please ask a question related to the Z1 Super App or its Mini Apps.

Example 5
Question:
What is the subscription price of Z1 Premium?

Answer:
I could not find this information in the provided documents.

Context:
{context}

User Question:
{query}

Answer:
"""


# --------------------------------------------------------------------------
# CACHED RESOURCES
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model (first run only, can take a minute)...")
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)


@st.cache_resource(show_spinner="Building knowledge base from PDF...")
def build_vectorstore(pdf_path: str, force_rebuild: bool = False):
    embedding_model = get_embedding_model()

    if force_rebuild and os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)

    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        # Reuse an already-built vector store instead of re-embedding every restart
        return Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding_model)

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    split_docs = splitter.split_documents(documents)

    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embedding_model,
        persist_directory=PERSIST_DIR,
    )
    return vectorstore


def get_llm():
    # Never hardcode the key in source. Set it as an environment variable
    # or, on Streamlit Community Cloud, in the app's "Secrets" panel.
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets["GROQ_API_KEY"]
        except Exception:
            api_key = None

    if not api_key:
        st.error(
            "GROQ_API_KEY is not set. Add it as an environment variable "
            "(local run) or under Settings → Secrets (Streamlit Cloud) "
            "before using the app."
        )
        st.stop()

    return ChatGroq(model=GROQ_MODEL_NAME, api_key=api_key, temperature=0.3)


def answer_question(query: str, retriever, llm) -> str:
    docs = retriever.invoke(query)
    context = "\n\n".join(doc.page_content for doc in docs)
    filled_prompt = PROMPT_TEMPLATE.format(context=context, query=query)
    response = llm.invoke(filled_prompt)
    return response.content


# --------------------------------------------------------------------------
# SIDEBAR — knowledge base source
# --------------------------------------------------------------------------
st.sidebar.title("📚 Knowledge Base")

os.makedirs(DATA_DIR, exist_ok=True)

if not os.path.exists(PDF_PATH):
    uploaded_file = st.sidebar.file_uploader(
        "Upload the Z1 mini-apps PDF", type=["pdf"]
    )
    if uploaded_file is not None:
        with open(PDF_PATH, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.rerun()
else:
    st.sidebar.success("PDF loaded ✅")
    if st.sidebar.button("🔁 Rebuild knowledge base"):
        build_vectorstore(PDF_PATH, force_rebuild=True)
        st.rerun()

if st.sidebar.button("🗑️ Clear chat history"):
    st.session_state.messages = []
    st.rerun()

# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
st.title("🤖 Z1 Super App Assistant")
st.caption("Ask questions about the Z1 Super App and its mini apps.")

if not os.path.exists(PDF_PATH):
    st.info("👈 Upload the source PDF from the sidebar to get started.")
    st.stop()

vectorstore = build_vectorstore(PDF_PATH)
retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
llm = get_llm()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_query = st.chat_input("Ask about the Z1 Super App or its Mini Apps...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = answer_question(user_query, retriever, llm)
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
