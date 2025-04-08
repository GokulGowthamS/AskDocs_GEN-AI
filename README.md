# 💬 AskDocs AI

> *From documents to decisions — powered by AI, secured locally.*

## 🔍 What is AskDocs AI?

**AskDocs AI** is a early stage startup comprising a powerful and private document question-answering app powered by **LLaMA 2** and **LangChain**. Upload your PDF, DOCX, or TXT files, and ask intelligent questions. All data is processed locally to maintain maximum privacy.

---

## 🎯 Features

- 📄 Supports PDF, DOCX, and TXT files
- 🧠 Local LLaMA 2 model inference (no cloud dependencies)
- 🔍 Intelligent search & similarity-based retrieval
- 📦 Built with Streamlit for an interactive UI
- 🖼️ Custom dark-themed background support
- 🧵 Efficient multi-threaded document parsing

---

## 🚀 Getting Started

### 📦 Prerequisites

- Python 3.9+
- Download the LLaMA 2 model from [HERE!](https://huggingface.co/gokulgowtham01/AskDocs_GEN-AI/upload/main/llama-2-7b-chat.Q4_K_M.gguf)  

### 📁 Folder Structure  
  
```  
project_root/
├── gen_ai.py                           # Streamlit application  
├── images/
│   └── freepik__adjust__9850.jpeg
│   └── 767.jpg
├── faiss_index/                        # Generated vector store  (auto-created)
├── evaluation_logs/                    # Generated json file  (auto-created)  
├── documents/
│   ├── empty.txt
│   ├── sample_doc.docx
│   ├── Advanced_Facts_Octopus.pdf  
│   └── sample_txt.txt
└── requirements.txt  
```

---

## 🧰 Installation

```bash
# Clone the repo
git clone https://github.com/yourusername/AskDocs_GEN-AI.git
cd AskDocs_GEN-AI

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

```  
## 🧠 Usage  

```bash
# Run
streamlit run app.py
```

1.  Upload one or more PDF, DOCX, or TXT documents.  
2.  Click Start The Fun!  
3.  Ask your question in natural language.  
4.  Get answers with cited document sources.

## 📦 Dependencies  

- streamlit  
- langchain  
- huggingface_hub  
- faiss-cpu  
- sentence-transformers  
- python-docx  
- PyMuPDF (fitz)  
- docx2txt  
- ctransformers

## 📎 Notes  

- You need to download and manually place the LLaMA model.
- Ensure GPU support is configured if needed (set gpu_layers accordingly).
- Image background is customizable — make sure the image path is valid.





