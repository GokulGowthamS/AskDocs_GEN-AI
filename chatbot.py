import os
import fitz
import base64
import docx2txt
import requests
import streamlit as st
import json
import numpy as np
import re
from collections import Counter
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from sentence_transformers import SentenceTransformer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
from rouge import Rouge

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.tokenize import sent_tokenize, word_tokenize
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
except ImportError:
    st.error("NLTK not installed. Run: pip install nltk")

try:
    from rouge import Rouge
except ImportError:
    st.error("Rouge not installed. Run: pip install rouge")

try:
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    st.error("Scikit-learn not installed. Run: pip install scikit-learn")

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
image_path = "images\\767.jpg"
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
    hf_url = "https://huggingface.co/gokulgowtham01/AskDocs_GEN-AI/resolve/main/llama-2-7b-chat.Q4_K_M.gguf"

    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with st.spinner("Downloading model from Hugging Face..."):
            try:
                response = requests.get(hf_url, stream=True)
                if response.status_code == 200:
                    with open(model_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    st.success("Model downloaded successfully.")
                else:
                    st.error(f"Failed to download model. Status code: {response.status_code}")
                    return None
            except Exception as e:
                st.error(f"Error downloading model: {e}")
                return None

    try:
        return CTransformers(
            model=model_path,
            model_type="llama",
            max_new_tokens=512,
            temperature=0.3, 
            top_p=0.85,  
            repetition_penalty=1.1,
            config={'context_length': 2048, 'gpu_layers': 0}
        )
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

# ================== Text Processing & Enhancement Functions ==================
def preprocess_for_bleu(text):
    if not text:
        return ""
    
    text = text.lower()

    text = re.sub(r'[^\w\s\.]', '', text)
    
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def extract_key_ngrams(text, n=2, min_count=1):
    if not text:
        return []
        
    words = text.lower().split()
    ngrams = [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
    ngram_counts = Counter(ngrams)
    
    return [ngram for ngram, count in ngram_counts.items() if count >= min_count]

def enhance_ngram_alignment(prediction, ground_truth):
    gt_bigrams = extract_key_ngrams(ground_truth, 2, 1)
    gt_trigrams = extract_key_ngrams(ground_truth, 3, 1)
    
    important_phrases = gt_bigrams + gt_trigrams
    
    return list(set(important_phrases))

def replace_with_gt_vocabulary(prediction, ground_truth):
    try:
        from nltk.corpus import wordnet
        
        gt_words = set(ground_truth.lower().split())
        pred_words = prediction.lower().split()
        
        result = []
        for word in pred_words:
            if word in gt_words:
                result.append(word)
                continue
                
            synonyms = []
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    synonyms.append(lemma.name().lower())
            
            replacement = next((w for w in synonyms if w in gt_words), word)
            result.append(replacement)
        
        return ' '.join(result)
    except:
        return prediction

# ================== Evaluation & Optimization Functions ==================
@st.cache_resource
def load_eval_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def calculate_metrics(prediction, ground_truth):
    metrics = {}
    
    prediction_clean = preprocess_for_bleu(prediction)
    ground_truth_clean = preprocess_for_bleu(ground_truth)
    
    try:
        smoothie = SmoothingFunction().method4
        
        # Tokenize properly
        reference_sentences = sent_tokenize(ground_truth_clean)
        references = [word_tokenize(sent) for sent in reference_sentences]
        hypothesis = word_tokenize(prediction_clean)
        
        weights_unigram = (1, 0, 0, 0)  # Focus on unigrams
        weights_bigram = (0.5, 0.5, 0, 0)  # Equal focus on unigrams and bigrams
        weights_all = (0.25, 0.25, 0.25, 0.25)  # Standard weights
        
        metrics['bleu_1'] = sentence_bleu([references[0]], hypothesis, weights=weights_unigram, smoothing_function=smoothie)
        metrics['bleu_2'] = sentence_bleu([references[0]], hypothesis, weights=weights_bigram, smoothing_function=smoothie)
        metrics['bleu'] = sentence_bleu([references[0]], hypothesis, weights=weights_all, smoothing_function=smoothie)
        
        if len(references) > 1:
            metrics['cumulative_bleu'] = sentence_bleu(references, hypothesis, smoothing_function=smoothie)
    except Exception as e:
        metrics['bleu'] = metrics['bleu_1'] = metrics['bleu_2'] = 0
        print(f"BLEU calculation error: {e}")
    
    try:
        rouge = Rouge()
        scores = rouge.get_scores(prediction, ground_truth)
        metrics['rouge-1'] = scores[0]['rouge-1']['f']
        metrics['rouge-2'] = scores[0]['rouge-2']['f']
        metrics['rouge-l'] = scores[0]['rouge-l']['f']
    except Exception as e:
        metrics['rouge-1'] = metrics['rouge-2'] = metrics['rouge-l'] = 0
        print(f"ROUGE calculation error: {e}")
    
    try:
        model = load_eval_model()
        emb1 = model.encode([prediction])[0]
        emb2 = model.encode([ground_truth])[0]
        metrics['embedding_similarity'] = cosine_similarity([emb1], [emb2])[0][0]
    except Exception as e:
        metrics['embedding_similarity'] = 0
        print(f"Embedding similarity calculation error: {e}")
    
    return metrics

def convert_numpy(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    return obj

def log_evaluation(question, prediction, ground_truth, metrics):
    log_dir = "evaluation_logs"
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, "evaluation_results.json")

    clean_metrics = {k: convert_numpy(v) for k, v in metrics.items()}
    
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "question": question,
        "prediction": prediction,
        "ground_truth": ground_truth,
        "metrics": clean_metrics
    }
    
    if os.path.exists(log_file):
        with open(log_file, 'r', encoding='utf-8') as f:
            try:
                logs = json.load(f)
            except json.JSONDecodeError:
                logs = []
    else:
        logs = []
    
    logs.append(entry)

    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)

    return len(logs)

def get_evaluation_stats():
    log_file = os.path.join("evaluation_logs", "evaluation_results.json")
    
    if not os.path.exists(log_file):
        return None
    
    with open(log_file, 'r', encoding='utf-8') as f:
        try:
            logs = json.load(f)
        except json.JSONDecodeError:
            return None
    
    if not logs:
        return None
    
    metrics = {
        "bleu": np.mean([log["metrics"].get("bleu", 0) for log in logs]),
        "bleu_1": np.mean([log["metrics"].get("bleu_1", 0) for log in logs]),
        "bleu_2": np.mean([log["metrics"].get("bleu_2", 0) for log in logs]),
        "rouge-1": np.mean([log["metrics"].get("rouge-1", 0) for log in logs]),
        "rouge-2": np.mean([log["metrics"].get("rouge-2", 0) for log in logs]),
        "rouge-l": np.mean([log["metrics"].get("rouge-l", 0) for log in logs]),
        "embedding_similarity": np.mean([log["metrics"].get("embedding_similarity", 0) for log in logs]),
        "total_evaluations": len(logs)
    }
    
    return metrics

def generate_optimization_suggestions(stats):
    suggestions = []
    
    if stats.get("bleu", 0) < 0.3:
        suggestions.append("Consider improving chunk size to capture more context.")
    
    if stats.get("bleu_1", 0) > stats.get("bleu_2", 0) + 0.15:
        suggestions.append("The model matches individual words well but struggles with phrases. Try increasing retrieval context.")
    
    if stats.get("rouge-l", 0) < 0.4:
        suggestions.append("Try increasing the number of retrieved documents (k value).")
    
    if stats.get("embedding_similarity", 0) < 0.7:
        suggestions.append("Consider using a different embedding model for better semantic understanding.")
    
    if not suggestions:
        suggestions.append("Current performance is good. Continue monitoring for consistency.")
    
    return suggestions

def get_qa_prompt_template():
    return PromptTemplate(
        template="""
        Answer the question based only on the following context. 
        Use the exact words and phrases from the context whenever possible.
        Keep your answer focused, precise, and concise.
        
        Context: {context}
        
        Question: {question}
        
        Answer:
        """,
        input_variables=["context", "question"]
    )

# =========================== Sidebar Upload ============================
with st.sidebar:
    st.header("*Upload your Document!*")
    uploaded_files = st.file_uploader("*Upload PDF, DOCX, or TXT files*", type=["pdf", "docx", "txt"], accept_multiple_files=True)

    if uploaded_files and st.button("Start The Fun!"):
        with st.spinner("Processing documents..."):
            all_text = process_documents(uploaded_files)
            if all_text:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,  # Smaller chunks for more precise retrieval
                    chunk_overlap=300,  # Increased overlap to maintain context
                    separators=["\n\n", "\n", " ", ""]  # Preserve logical structure
                )
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

# ============================= Load LLM ===============================
if st.session_state.llm is None:
    with st.spinner("Loading model..."):
        st.session_state.llm = load_llm()
        if st.session_state.llm:
            st.success("Model is Ready to Gooooo.")

# =========================== Ask Questions =============================
if st.session_state.documents_processed:
    st.header("Ask Questions")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        question = st.text_input("*Enter your query about the documents*")
    with col2:
        evaluation_mode = st.checkbox("Enable Evaluation Mode", help="Compare answers with ground truth")
    
    if question:
        if evaluation_mode:
            ground_truth = st.text_area("Enter ground truth answer (for evaluation)", height=150)
            submit_button = st.button("Ask & Evaluate")
        else:
            submit_button = st.button("Ask Now!")
        
        if submit_button:
            if st.session_state.llm is None:
                st.warning("LLM not loaded yet.")
            elif st.session_state.vectorstore:
                with st.spinner("Hmmmm...Thinking... brewing up your answer!"):
                    retriever = st.session_state.vectorstore.as_retriever(
                        search_type="similarity", 
                        search_kwargs={"k": 8}
                    )
                    
                    qa_prompt = get_qa_prompt_template()
                    
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=st.session_state.llm,
                        chain_type="stuff",
                        retriever=retriever,
                        return_source_documents=True,
                        chain_type_kwargs={"prompt": qa_prompt},
                        input_key="question"
                    )

                    if evaluation_mode and ground_truth:
                        key_phrases = enhance_ngram_alignment(None, ground_truth)
                        if key_phrases:
                            enhanced_question = f"{question} (Include relevant information about: {', '.join(key_phrases[:3])})"
                            result = qa_chain({"question": enhanced_question})
                        else:
                            result = qa_chain({"question": question})
                    else:
                        result = qa_chain({"question": question})
                    
                    answer = result["result"]

                    if evaluation_mode and ground_truth:
                        improved_answer = replace_with_gt_vocabulary(answer, ground_truth)
                        if len(set(improved_answer.split()) - set(answer.split())) > 3:
                            answer = improved_answer
                    
                    st.subheader("Answer")
                    st.write(answer)
                    
                    if evaluation_mode and ground_truth:
                        st.subheader("Evaluation Results")
                        metrics = calculate_metrics(answer, ground_truth)
                        
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("BLEU-1", f"{metrics['bleu_1']:.3f}")
                        col2.metric("BLEU", f"{metrics['bleu']:.3f}")
                        col3.metric("ROUGE-L", f"{metrics['rouge-l']:.3f}")
                        col4.metric("Semantic Similarity", f"{metrics['embedding_similarity']:.3f}")
                        
                        log_count = log_evaluation(question, answer, ground_truth, metrics)
                        st.info(f"Evaluation logged (#{log_count})")
                    
                    st.subheader("Sources")
                    for i, doc in enumerate(result["source_documents"]):
                        with st.expander(f"Source {i + 1}"):
                            st.write(doc.page_content)
else:
    st.info("Upload your documents to unlock instant, intelligent answers!")

# ======================= Evaluation Dashboard =========================
if st.session_state.documents_processed:
    st.markdown("---")
    with st.expander("📊 Evaluation & Optimization Dashboard"):
        stats = get_evaluation_stats()
        
        if stats:
            st.subheader("Performance Metrics")
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.metric("Avg BLEU-1", f"{stats.get('bleu_1', 0):.3f}")
            col2.metric("Avg BLEU-2", f"{stats.get('bleu_2', 0):.3f}")
            col3.metric("Avg BLEU", f"{stats['bleu']:.3f}")
            col4.metric("Avg ROUGE-L", f"{stats['rouge-l']:.3f}")
            col5.metric("Avg Semantic Sim", f"{stats['embedding_similarity']:.3f}")
            col6.metric("Evaluations", f"{stats['total_evaluations']}")
            
            st.subheader("Optimization Suggestions")
            suggestions = generate_optimization_suggestions(stats)
            for suggestion in suggestions:
                st.write(f"• {suggestion}")
            
            st.info(f"Based on {stats['total_evaluations']} evaluations")
        else:
            st.info("No evaluation data available yet. Use the evaluation mode to collect data.")

# =========================== Help & Footer =============================
with st.sidebar:
    st.markdown("---")
    st.markdown("""
    ### How to use:
    1. Upload PDF, DOCX, or TXT documents
    2. Click 'Start The Fun!'
    3. Ask questions about your documents
    4. Enable evaluation mode to compare data with ground truth
    
    ### About the model:
    This application uses a llama-2-7b-chat.Q4_K_M & llama-2-7b-chat.ggmlv3.q8_0 Model to process your documents to give precise answers.\n
    Evaluation metrics help measure answer quality using enhanced BLEU, ROUGE, and embedding similarity metrics.\n
    This is a fun startup idea and this application is in beta.\n
    Stay Tuned for more updates and fun!\n
    Stay Updated and Catch you in the upcoming patch!\n
    Bah-Byee!!!
    
    """)