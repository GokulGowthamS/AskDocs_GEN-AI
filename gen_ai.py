import os
import fitz
import base64
import docx2txt
import streamlit as st
from concurrent.futures import ThreadPoolExecutor
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import CTransformers

st.set_page_config(page_title="💬 AskDocs AI", layout="wide")

def img_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        img_data = img_file.read()
        return base64.b64encode(img_data).decode()

image_path = "F:\\Guvi Projects\\GenAI (Final Project)\\images\\freepik__adjust__9850.jpeg"

try:
    img_base64 = img_to_base64(image_path)

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 0, 0.7)), url('data:image/jpeg;base64,{img_base64}');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            padding: 0;
        }}
        .title-container {{
            text-align: center;
            color: white;
            font-size: 4em;
            margin-top: 350px;
            margin-bottom: 20px;
            
        }}
        
        </style>
        """, unsafe_allow_html=True
    )

except FileNotFoundError:
    st.error("Image not found at the specified path.")

st.title("💬 AskDocs AI")

st.subheader("*From documents to decisions — powered by AI, secured locally.*")

if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'documents_processed' not in st.session_state:
    st.session_state.documents_processed = False
if 'llm' not in st.session_state:
    st.session_state.llm = None
if 'ready_to_ask' not in st.session_state:
    st.session_state.ready_to_ask = False

def extract_text(file):
    file_extension = os.path.splitext(file.name)[1].lower()

    if file_extension == '.pdf':
        with fitz.open(stream=file.read(), filetype="pdf") as doc:
            return "".join([page.get_text() for page in doc])

    elif file_extension == '.docx':
        return docx2txt.process(file)

    elif file_extension == '.txt':
        return file.getvalue().decode('utf-8')

    else:
        st.error(f"Unsupported file format: {file_extension}")
        return None
    
def process_documents(files):
    with ThreadPoolExecutor() as executor:
        texts = list(executor.map(extract_text, files))
    return "\n\n".join([text for text in texts if text])

@st.cache_resource
def load_llm():
    model_path = "F:\\Guvi Projects\\GenAI (Final Project)\\models\\llama-2-7b-chat.Q4_K_M.gguf"

    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        st.info("Please download the model and place it in the 'models' directory")
        return None

    llm = CTransformers(
        model=model_path,
        model_type="llama",
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.15,
        config={
            'context_length': 2048,
            'gpu_layers': 0
        }
    )

    return llm

if st.session_state.llm is None:
    st.markdown("### Loading LLM model...")
    with st.spinner("Please wait while the local model loads..."):
        st.session_state.llm = load_llm()
        if st.session_state.llm:
            st.success("Engine's in Perfect Condition! Let's Gooooooooo...")

with st.sidebar:
    st.header("*Upload your Document!*")
    uploaded_files = st.file_uploader("Upload PDF, DOCX, or TXT files", type=["pdf", "docx", "txt"], accept_multiple_files=True)

    if uploaded_files:
        process_btn = st.button("Start The Fun!")
        if process_btn:
            with st.spinner("Processing documents..."):
                all_text = process_documents(uploaded_files)

                if all_text:
                    with st.spinner("Splitting text into chunks..."):
                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                        chunks = text_splitter.split_text(all_text)

                    with st.spinner("Creating embeddings and building vector store..."):
                        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

                        st.session_state.vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
                        st.session_state.vectorstore.save_local("faiss_index")
                        st.session_state.documents_processed = True
                        st.session_state.ready_to_ask = False
                        st.success(f"Successfully processed {len(uploaded_files)} documents")
                else:
                    st.error("No text was extracted from the documents.")

if not st.session_state.documents_processed and os.path.exists("faiss_index"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    st.session_state.vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    st.session_state.documents_processed = True

if st.session_state.documents_processed:
    st.header("Ask Questions")
    question = st.text_input("*Enter your query about the documents*")
    start_btn = st.button("Ask Now!")

    if question and start_btn:
        if st.session_state.llm is None:
            st.warning("Please load the LLM model first.")
        elif st.session_state.vectorstore:
            with st.spinner("Hmmmm...Thinking hard... Brewing up your answer! Grab a coffee...while I work my magic!"):
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
                    with st.expander(f"Source {i+1}"):
                        st.write(doc.page_content)
else:
    st.info("Upload your documents to unlock instant, intelligent answers!")

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
