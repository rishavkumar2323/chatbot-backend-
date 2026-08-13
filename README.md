# Z1 Super App Assistant — Deployment Guide

## ⚠️ Zaroori: pehle ye karo
Aapke original `final.py` mein Groq API key hardcoded thi (`gsk_myLeN...`).
Wo key ab expose ho chuki hai, isliye:
1. https://console.groq.com pe jaake wo key **revoke/regenerate** karo.
2. Nayi key kabhi bhi code mein mat likho — hamesha environment variable ya
   secrets manager se lo (is app mein already aisa hi setup hai).

## Files
- `app.py` — Streamlit chatbot (PDF load → chunk → embed → Chroma → Groq LLM)
- `requirements.txt` — dependencies
- `.gitignore` — vectorstore/PDF/secrets ko git se bahar rakhta hai

## Local run
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

export GROQ_API_KEY="your_new_key_here"   # Windows: set GROQ_API_KEY=...
streamlit run app.py
```
Pehli baar app kholne par sidebar se apni PDF upload karo (ya `data/sets_of_miniapp.pdf`
naam se pehle se rakh do). Vectorstore `chroma_db_bge/` mein persist ho jayega,
taaki restart pe dobara embed na karna pade.

## Deploy — Streamlit Community Cloud
1. Ye files ek GitHub repo mein push karo (PDF **mat** push karo, use uploader se add karo).
2. https://share.streamlit.io pe jaake "New app" → repo select karo → `app.py` as entry point.
3. App settings → **Secrets** mein ye daalo:
   ```toml
   GROQ_API_KEY = "your_new_key_here"
   ```
4. Deploy karo. First build thoda slow hoga (torch + embedding model download hote hain).

### Streamlit Cloud ke liye already fix kiya hua hai
- **sqlite3 error** (`chromadb requires sqlite3 >= 3.35.0`): Streamlit Cloud ka
  system sqlite purana hai. `app.py` ke top pe `pysqlite3-binary` se swap karta
  hai, aur `requirements.txt` mein wo package add hai. Isse `RuntimeError` wala
  common Chroma deploy error nahi aayega.
- **Build size/timeout**: `requirements.txt` mein `--extra-index-url
  https://download.pytorch.org/whl/cpu` daala hai taaki CPU-only torch
  (~200MB) install ho, GPU/CUDA wala bhaari torch (~2GB+) nahi — warna build
  fail/timeout ho sakta hai.

## ⚠️ Resource caution
Free tier ~**1GB RAM hard limit** hai (verified — Streamlit apps par ye limit
hit hone par "gone over its resource limits" error deta hai). Isse reliably
avoid karne ke liye embedding model already **`BAAI/bge-small-en-v1.5`** pe
switch kar diya gaya hai (bge-base ke bajaye) — RAM footprint kaafi kam hai.
Agar zyada RAM wali jagah (Render/Railway/apna VPS) pe deploy karo, to
`app.py` mein `EMBED_MODEL_NAME` ko wapas `"BAAI/bge-base-en-v1.5"` kar sakte ho
better retrieval quality ke liye.

Agar phir bhi resource-limit error aaye:
- Chat history bahut lambi na hone do (naya session start karo).
- `TOP_K` (retriever ka `k`) 10 se ghata ke 4–5 karo.
- Sabse reliable fix: hosted embeddings API pe switch (local torch model
  hatana) — bata dena, ye version bana dunga.

## Kya maine verify kiya, kya nahi (honest note)
- ✅ `app.py` syntax-valid hai, `requirements.txt` ke saare packages resolve
  hote hain (maine `pip install --dry-run` se test kiya).
- ✅ sqlite3 fix code mein sahi jagah pe hai.
- ⚠️ CPU-only torch wala trick (`--extra-index-url .../whl/cpu`) maine
  **poori tarah verify nahi kar paya** — mera sandbox `download.pytorch.org`
  tak nahi pahunch pata (network-restricted environment), isliye pakka nahi
  keh sakta ki ye Streamlit Cloud pe CUDA packages ko skip karega ya nahi.
  Ye harmless hai (worst case sirf bada torch download hoga), isliye rakha
  hai, lekin ye "guaranteed fix" nahi hai.
- ⚠️ Actual deploy success — GROQ key sahi hona, PDF sahi load hona, aur real
  RAM usage — ye sirf Streamlit Cloud pe actual deploy karke hi 100% confirm
  hoga. Main isse yahan run nahi kar sakta.

## Deploy — Hugging Face Spaces (alternative)
1. Naya Space banao, SDK = Streamlit.
2. `app.py`, `requirements.txt` upload karo.
3. Space Settings → **Repository secrets** mein `GROQ_API_KEY` add karo.
4. HF Spaces ka sqlite generally newer hota hai, lekin `pysqlite3-binary` fix
   waise bhi harmless hai (no-op agar zaroorat na ho).
