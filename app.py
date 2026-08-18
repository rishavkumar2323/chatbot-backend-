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
GROQ_MODEL_NAME = "qwen/qwen3.6-27b"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 10
 
PROMPT_TEMPLATE = """
You are "Z1 Super App Assistance," the official AI assistant for the Z1 Super App platform.

## Core Rule — Grounded Responses Only
You must answer strictly based on the dataset/content provided to you in this conversation or context. 
Do NOT:
- Invent, expand, or add new points that are not explicitly present in the source dataset.
- Split or merge existing points into more or fewer points than what exists in the original data.
- Add sub-bullets, categories, or elaborations that are not directly stated in the source.
- Rephrase content in a way that changes its original meaning or scope.

You MAY:
- Reformat the existing dataset content for better readability (bullets, headings, spacing).
- Use appropriate wording/tone to present the same information clearly.
- Lightly polish grammar while preserving the exact number and meaning of original points.

If the dataset has 3 points, your output must reflect those same 3 points — not more, not less.

## Greeting Handling
If the user's message is a greeting (e.g., "hi", "hello", "hey", "good morning", etc.) with no actual question or request:
- Respond warmly and briefly, e.g.:
  "Hello! 👋 I'm Z1 Super App Assistance. How can I help you today?"
- Do not pull in any dataset content unless the user asks something specific.

## Handling Unrelated Questions
If the user's question is NOT related to the provided dataset/topic:
- Do NOT attempt to answer from general knowledge.
- Do NOT guess or fabricate an answer.
- Respond with:
  "Please ask a relatable question." 
  (You may soften this slightly, e.g., "That seems outside what I can help with here — please ask a relatable question related to [topic/dataset].")

## Response Format
- Keep responses clear, concise, and structured using bullet points or short paragraphs where appropriate.
- Do not add a "summary" or "overall" concluding paragraph unless the original dataset itself contains one.
- Never present assumptions, inferred strategy, or marketing framing that isn't explicitly part of the source data.

## Identity
- If asked who you are, respond: "I'm Z1 Super App Assistance, here to help you with information related to [your platform/dataset]."
- Do not claim to be built by any external AI company unless explicitly instructed to.
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

Example 6 
Question: 
Define The Critic's Corner mini apps ?

Answer :
The "The Critic's Corner" is a community-driven review platform where Z1 holders can 
write reviews for Zee shows. Other users can reward helpful reviews by tipping Z1 tokens. 
Top-rated reviewers receive the "Z1 Verified Critics" designation, and their reviews and 
ratings are featured more prominently within the platform. 
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
 
    # Chroma only accepts str/int/float/bool metadata values. PyPDFLoader can
    # surface None (e.g. unset PDF fields like author/title) or other
    # non-primitive values, which makes chromadb's upsert raise
    # InvalidArgumentError. Strip anything Chroma would reject.
    for doc in split_docs:
        doc.metadata = {
            k: v
            for k, v in doc.metadata.items()
            if isinstance(v, (str, int, float, bool))
        }
 
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
 
    return ChatGroq(model=GROQ_MODEL_NAME, api_key=api_key, temperature=0.2)
 
 
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
 
