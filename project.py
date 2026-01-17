import streamlit as st
import tempfile, base64, os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
 
# used for backgrounds
st.set_page_config(page_title="PDF Chatbot", page_icon="📄", layout="wide") 

def set_bg(image_file="bg.jpg"):
   
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    bg_path = os.path.join(BASE_DIR, image_file)

    if not os.path.exists(bg_path):
        st.warning("⚠️ Background image not found! Put bg.jpg in same folder as project.py")
        return

    with open(bg_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}

        /* Hide sidebar */
        section[data-testid="stSidebar"] {{
            display: none;
        }}

        
        .blob {{
            position: fixed;
            width: 420px;
            height: 420px;
            filter: blur(70px);
            opacity: 0.45;
            z-index: 0;
            border-radius: 999px;
            animation: float 8s ease-in-out infinite;
        }}

        .blob.one {{
            top: -130px;
            left: -130px;
            background: radial-gradient(circle, #00c6ff, #0072ff);
        }}

        .blob.two {{
            bottom: -150px;
            right: -150px;
            background: radial-gradient(circle, #ff7eb3, #ff758c);
            animation-delay: 1.5s;
        }}

        @keyframes float {{
            0%,100% {{ transform: translateY(0px); }}
            50% {{ transform: translateY(35px); }}
        }}

        
        .glass {{
            position: relative;
            z-index: 1;
            background: rgba(0,0,0,0.55);
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 18px;
            padding: 18px;
            backdrop-filter: blur(14px);
            box-shadow:
                0 18px 45px rgba(0,0,0,0.55),
                inset 0 1px 0 rgba(255,255,255,0.06);
            transform: perspective(1200px) rotateX(0deg);
            transition: 0.25s ease;
        }}

        
        .glass:hover {{
            transform: perspective(1200px) translateY(-6px) rotateX(2deg);
            box-shadow:
                0 25px 70px rgba(0,0,0,0.65),
                inset 0 1px 0 rgba(255,255,255,0.08);
        }}

        
        div.stButton > button {{
            border-radius: 14px;
            padding: 10px 18px;
            font-weight: 800;
            background: linear-gradient(135deg, rgba(0,255,180,0.25), rgba(0,180,255,0.18));
            border: 1px solid rgba(255,255,255,0.14);
            box-shadow: 0 10px 25px rgba(0,0,0,0.35);
            transition: 0.2s ease;
        }}

        div.stButton > button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 18px 40px rgba(0,0,0,0.5);
        }}

        
        .user {{
            background: rgba(255,255,255,0.14);
            padding: 12px 14px;
            border-radius: 16px;
            margin: 8px 0;
            border: 1px solid rgba(255,255,255,0.10);
            box-shadow: 0 10px 25px rgba(0,0,0,0.22);
        }}
        .bot {{
            background: rgba(46, 204, 113, 0.18);
            padding: 12px 14px;
            border-radius: 16px;
            margin: 8px 0;
            border: 1px solid rgba(46, 204, 113, 0.25);
            box-shadow: 0 10px 25px rgba(0,0,0,0.22);
        }}

        footer {{visibility:hidden;}}
        </style>

        <!--  3D floating blobs -->
        <div class="blob one"></div>
        <div class="blob two"></div>
        """,
        unsafe_allow_html=True
    )

set_bg("bg.jpg")  

# initializing the model
# Ensure API key is present (accept GOOGLE_API_KEY or GEMINI_API_KEY)
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if api_key:
    api_key = api_key.strip().strip('"').strip("'")

if not api_key:
    st.error("Gemini API key not found — please set `GOOGLE_API_KEY` or `GEMINI_API_KEY`.")
    st.markdown(
        """
        **How to set the key:**
        - Locally: add to `.env` (no surrounding quotes) e.g. `GOOGLE_API_KEY=ya29...`
        - Streamlit Cloud: go to *Manage app → Settings → Secrets* and add `GOOGLE_API_KEY`.
        - As a fallback (less secure): export `GOOGLE_API_KEY` in the environment or pass `api_key` when constructing `GoogleGenerativeAI`.
        """
    )
    st.stop()

try:
    model = GoogleGenerativeAI(model="gemini-2.5-flash", api_key=api_key)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", api_key=api_key)
except Exception as e:
    import traceback
    tb = traceback.format_exc()
    st.error("Model initialization failed — see traceback below.")
    st.text_area("Model init traceback", tb, height=300)
    print(tb)
    st.stop()

# fixing the parameter
TOP_K = 4
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_SUMMARY_CHUNKS = 20

# header
st.markdown(
    """
    <div class="glass">
        <h1 style="margin:0;">📄✨ PDF Chatbot</h1>
        <p style="margin-top:6px;opacity:0.85;">
        📤 Upload PDF → 🧠 Summarize → 💬 Ask Questions (Gemini + FAISS)
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("")

@st.cache_resource(show_spinner=False)
def build_vectorstore(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)

    vs = FAISS.from_documents(chunks, embeddings)
    return vs, chunks, len(docs)

# main layout
left, right = st.columns([1.15, 1])

with left:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("📤 Upload Section")
    upload_file = st.file_uploader("📎 Upload your PDF here", type=["pdf"])
    st.markdown("</div>", unsafe_allow_html=True)

    if upload_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(upload_file.read())
            pdf_path = tmp.name

        st.success("✅ PDF Uploaded Successfully 🎉")

        with st.spinner("⚡ Indexing PDF... Creating Embeddings + FAISS"):
            vectorstore, chunks, total_pages = build_vectorstore(pdf_path)

        st.info(f"📄 Pages: {total_pages}")

with right:
    if upload_file is not None:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.subheader("✨ Features")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("📌 Summarize PDF ✨", use_container_width=True):
                st.info("🧠 Summarizing... please wait ✅")

                pdf_text = "\n\n".join([c.page_content for c in chunks[:MAX_SUMMARY_CHUNKS]])

                summary_prompt = f"""
summrize this pdf in:
1) short paragraph
2) 8-10 bullet points
3) Key topics list

PDF Text:
{pdf_text}
"""
                summary = model.invoke(summary_prompt)
                st.markdown('<div class="bot"><b>📌 Summary:</b></div>', unsafe_allow_html=True)
                st.write(summary)

        with col2:
            if st.button("🧹 Clear Chat", use_container_width=True):
                st.session_state["chat"] = []

        st.write("---")

        st.subheader("💬 Ask Questions from PDF 🤖")
        query = st.text_input("📝 Type your question here...")

        if "chat" not in st.session_state:
            st.session_state["chat"] = []

        if query:
            retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
            docs = retriever.invoke(query)

            context = "\n\n".join([d.page_content for d in docs])

            prompt = PromptTemplate(
                template="""
You are a helpful assistant. Answer using the given context.

Context:
{context}

Question: {question}

Answer:
""",
                input_variables=["context", "question"],
            )

            chain = prompt | model | StrOutputParser()

            with st.spinner("🤔 Thinking..."):
                response = chain.invoke({"context": context, "question": query})

            st.session_state["chat"].append(("user", query))
            st.session_state["chat"].append(("bot", response))

        for role, msg in st.session_state["chat"]:
            if role == "user":
                st.markdown(f'<div class="user"><b>🧑 You:</b><br>{msg}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="bot"><b>🤖 Bot:</b><br>{msg}</div>', unsafe_allow_html=True)

        with st.expander("🔍 Show Retrieved Chunks 📄"):
            if "docs" in locals():
                for i, d in enumerate(docs):
                    st.write(f"--- 📌 Chunk {i+1} ---")
                    st.write(d.page_content[:1500])

        st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.markdown(
            """
            <div class="glass">
                <h3 style="margin:0;">👋 Hey there!</h3>
                <p style="margin-top:8px;opacity:0.85;">
                    📎 Upload a PDF first, then I can summarize it and answer your questions 😄
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
