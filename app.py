import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq


# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(title="Z1 RAG Chatbot API")


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# FILE PATHS
# =========================================================

PDF_PATH = "sets of miniapp.pdf"
CHROMA_PATH = "./chroma_db_bge"


# =========================================================
# EMBEDDING MODEL
# Same as your Colab code
# =========================================================

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5"
)


# =========================================================
# LOAD / CREATE CHROMA
# Same chunking as your Colab:
# chunk_size = 800
# chunk_overlap = 150
# =========================================================

if os.path.exists(CHROMA_PATH):

    print("Loading existing ChromaDB...")

    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_model
    )

else:

    print("Creating ChromaDB from PDF...")

    loader = PyPDFLoader(PDF_PATH)

    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    split_docs = text_splitter.split_documents(documents)

    print("Total chunks:", len(split_docs))

    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embedding_model,
        persist_directory=CHROMA_PATH
    )

    print("ChromaDB created successfully.")


# =========================================================
# RETRIEVER
# Same as your Colab code: k = 10
# =========================================================

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 10}
)


# =========================================================
# GROQ LLM
# Same model and temperature as your Colab
# API key comes from Render environment variable
# =========================================================

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.environ["GROQ_API_KEY"],
    temperature=0.3
)


# =========================================================
# REQUEST MODEL
# =========================================================

class ChatRequest(BaseModel):
    query: str


# =========================================================
# RAG FUNCTION
# =========================================================

def ask_rag(query):

    # Retrieve top 10 relevant documents
    docs = retriever.invoke(query)

    # Create context exactly like your Colab
    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    # =====================================================
    # YOUR ORIGINAL PROMPT
    # =====================================================

    prompt = """
You are an AI Assistant for the Z1 Super App.

Your task is to answer ONLY from the provided context.

Rules:
1. Use only the given context.
2. Do not make up information.
3. If the answer is clearly available in the context.
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

Example 3
question :
What is the core feature of First Look mini app?

Answer
Core Features & User Journey
• The Vault: A high-security digital screening room where upcoming content is hosted
for a limited window (e.g., 24–48 hours before the global premiere).
• ACR Redemption: A seamless "Pay-to-Unlock" mechanism where users spend
their accumulated ACRs to gain entry to specific previews.
• NDA Check-In: A lightweight, gamified "Digital Agreement" that reinforces the
exclusivity of the content, encouraging users not to leak spoilers.
• Community Reaction Hub: A dedicated space for "First Look" viewers to discuss
their theories and reactions, creating a buzz that spills over into the broader app
once the content goes live.

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
the health of the app’s internal economy.

Advertisers/Sponsors High-Intent Audience: Provides access to the most engaged
segments of the user base for targeted brand placements or
"Presented by" credits.

Example 3
Question:
What is the In-Episode Easter Egg Hunt?

Answer:
The In-Episode Easter Egg Hunt is a mini app that lets users search for hidden symbols or objects while watching newly released episodes. Users who find them first can earn Z1 jackpot rewards.

 activities
- Completing challenges

Example 5
Question:
Who is the CEO of Microsoft?

Answer:
Please ask a question related to the Z1 Super App or its Mini Apps.

Example 6
Question:
What is the subscription price of Z1 Premium?

Answer:
I could not find this information in the provided documents.

Example 7
Question:
What is the operational highlight of First Look mini app?

Answer
Security: Watermarking technology can be integrated to ensure that if content is
leaked, the source can be identified, protecting the intellectual property of content
partners.

• Scalability: The architecture allows for everything from 30-second "sneak peeks" to
full-length pilot episodes, making it adaptable for various content genres.

Context:
{context}

User Question:
{query}

Answer:
"""

    # Fill context and user query
    prompt = prompt.format(
        context=context,
        query=query
    )

    # Call LLM
    response = llm.invoke(prompt)

    return response.content


# =========================================================
# CHAT API
# This is what your HTML frontend will call
# =========================================================

@app.post("/api/chat")
def chat(request: ChatRequest):

    answer = ask_rag(request.query)

    return {
        "answer": answer
    }


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/")
def home():

    return {
        "status": "running",
        "message": "Z1 RAG Backend is working"
    }
