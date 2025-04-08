import os
import fitz
import base64
import docx2txt
import requests
import streamlit as st
from concurrent.futures import ThreadPoolExecutor
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import CTransformers

st.set_page_config(page_title="💬 AskDocs AI", layout="wide")

# ============================== Utilities ==============================
def img_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def download_model_from_hf(model_url, save_path):
    st.info("Model downloading from Hugging Face...")
    response = requests.get(model_url, stream=True)
    if response.status_code == 200:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        st.success("Model downloaded successfully.")
    else:
        st.error("Failed to download the model from Hugging Face.")
        return None

# ============================ Background ===============================
image_path = "images\\freepik__adjust__9850.jpeg"
if os.path.exists(image_path):
    img_base64 = img_to_base64(image_path)
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 0, 0.5)), url('data:image/jpeg;base64,{img_base64}');
            background-size: cover;
            background-position: center;
        }}
        .title-container {{
            text-align: center;
            color: white;
            font-size: 4em;
            margin-top: 350px;
        }}
        </style>
        """, unsafe_allow_html=True
    )
else:
    st.error("Background image not found!")

# ========================== Session Setup ==============================
st.title("💬 AskDocs AI")
st.subheader("*From documents to decisions — powered by AI, secured locally.*")

st.session_state.setdefault('vectorstore', None)
st.session_state.setdefault('documents_processed', False)
st.session_state.setdefault('llm', None)
st.session_state.setdefault('ready_to_ask', False)

# ======================= Document Processing ===========================
def extract_text(file):
    ext = os.path.splitext(file.name)[1].lower()
    if ext == '.pdf':
        return "".join([p.get_text() for p in fitz.open(stream=file.read(), filetype="pdf")])
    elif ext == '.docx':
        return docx2txt.process(file)
    elif ext == '.txt':
        return file.getvalue().decode('utf-8')
    else:
        st.error(f"Unsupported file type: {ext}")
        return None

def process_documents(files):
    with ThreadPoolExecutor() as executor:
        texts = list(executor.map(extract_text, files))
    return "\n\n".join([t for t in texts if t])

# ============================= LLM Loader ==============================
@st.cache_resource
def load_llm():
    model_path = "models\\llama-2-7b-chat.Q4_K_M.gguf"
    model_url = "https://huggingface.co/gokulgowtham01/AskDocs_GEN-AI/upload/main/llama-2-7b-chat.Q4_K_M.gguf"

    if not os.path.exists(model_path):
        download_model_from_hf(model_url, model_path)

    if os.path.exists(model_path):
        return CTransformers(
            model=model_path,
            model_type="llama",
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.15,
            config={'context_length': 2048, 'gpu_layers': 0}
        )
    return None

    if st.session_state.llm is None:
        with st.spinner("Loading model..."):
            st.session_state.llm = load_llm()
            if st.session_state.llm:
                st.success("Engine's in Perfect Condition! Let's Gooooooooo...")

# =========================== Sidebar Upload ============================
with st.sidebar:
    st.header("*Upload your Document!*")
    uploaded_files = st.file_uploader("*Upload PDF, DOCX, or TXT files*", type=["pdf", "docx", "txt"], accept_multiple_files=True)

    if uploaded_files and st.button("Start The Fun!"):
        with st.spinner("Processing documents..."):
            all_text = process_documents(uploaded_files)
            if all_text:
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                chunks = splitter.split_text(all_text)

                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                st.session_state.vectorstore = FAISS.from_texts(chunks, embeddings)
                st.session_state.vectorstore.save_local("faiss_index")

                st.session_state.documents_processed = True
                st.session_state.ready_to_ask = False
                st.success(f"Successfully processed {len(uploaded_files)} documents!")
            else:
                st.error("No text extracted.")

# =========================== Load Cached DB ============================
if not st.session_state.documents_processed and os.path.exists("faiss_index"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    st.session_state.vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    st.session_state.documents_processed = True

# =========================== Ask Questions =============================
if st.session_state.documents_processed:
    st.header("Ask Questions")
    question = st.text_input("*Enter your query about the documents*")
    if question and st.button("Ask Now!"):
        if st.session_state.llm is None:
            st.warning("LLM not loaded yet.")
        elif st.session_state.vectorstore:
            with st.spinner("Hmmmm...Thinking... brewing up your answer!"):
                retriever = st.session_state.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
                qa_chain = RetrievalQA.from_chain_type(
                    llm=st.session_state.llm,
                    chain_type="stuff",
                    retriever=retriever,
                    return_source_documents=True
                )
                result = qa_chain({"query": question})

                st.subheader("Answer")
                st.write(result["result"])

                st.subheader("Sources")
                for i, doc in enumerate(result["source_documents"]):
                    with st.expander(f"Source {i + 1}"):
                        st.write(doc.page_content)
else:
    st.info("Upload your documents to unlock instant, intelligent answers!")

# =========================== Help & Footer =============================
with st.sidebar:
    st.markdown("---")
    st.markdown("""
    ### How to use:
    1. Upload PDF, DOCX, or TXT documents
    2. Click 'Start The Fun!'
    3. Ask questions about your documents
    
    ### About the model:
    This application uses a llama-2-7b-chat.Q4_K_M Model to process your documents to give precise answers.\n
    This is a fun startup idea and this application is in beta.\n
    Stay Tuned for more updates and fun!\n
    Stay Updated and Catch you in the upcoming patch!\n
    Bah-Byee!!!
    
    """)
