import streamlit as st
import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq
import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
INDEX_FILE = "index.faiss"
METADATA_FILE = "metadata.json"
CHAT_HISTORY_FILE = "chat_history.json"
MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50

# Page config
st.set_page_config(page_title="AI RAG App", layout="wide")

# Custom CSS for styling
st.markdown("""
<style>
    /* Styling elements for a modern clean UI */
    .stChatFloatingInputContainer {
        padding-bottom: 20px;
    }
    .uploaded-doc {
        border-left: 3px solid #00f2fe;
        padding-left: 10px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

def load_chat_history():
    """Load chat history from local JSON file."""
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                pass
    return []

def save_chat_history(history):
    """Save chat history to local JSON file."""
    with open(CHAT_HISTORY_FILE, "w") as f:
        json.dump(history, f)

# Initialize Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_chat_history()

@st.cache_resource
def load_embedding_model():
    """Load the SentenceTransformer model and cache it."""
    return SentenceTransformer(MODEL_NAME)

@st.cache_resource
def get_groq_client():
    """Initialize Groq client."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets["GROQ_API_KEY"]
        except Exception:
            pass
            
    if not api_key:
        st.error("GROQ_API_KEY not found. Please add it to your .env file locally, or to Streamlit Secrets when deploying.")
        st.stop()
    return Groq(api_key=api_key)

def init_or_load_faiss():
    """Load existing FAISS index or create a new one."""
    embed_dim = 384 # Output dimension of all-MiniLM-L6-v2
    if os.path.exists(INDEX_FILE) and os.path.exists(METADATA_FILE):
        index = faiss.read_index(INDEX_FILE)
        with open(METADATA_FILE, "r") as f:
            metadata = json.load(f)
    else:
        index = faiss.IndexFlatL2(embed_dim)
        metadata = []
    return index, metadata

def save_faiss(index, metadata):
    """Save FAISS index and metadata to disk."""
    faiss.write_index(index, INDEX_FILE)
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f)

def remove_source_from_faiss(source_to_remove, index, metadata):
    """Remove all chunks associated with a specific source from FAISS and metadata."""
    indices_to_keep = [i for i, meta in enumerate(metadata) if meta.get("source") != source_to_remove]
    
    if len(indices_to_keep) == 0:
        embed_dim = 384
        new_index = faiss.IndexFlatL2(embed_dim)
        new_metadata = []
    else:
        embeddings_to_keep = np.array([index.reconstruct(i) for i in indices_to_keep]).astype("float32")
        embed_dim = 384
        new_index = faiss.IndexFlatL2(embed_dim)
        new_index.add(embeddings_to_keep)
        new_metadata = [metadata[i] for i in indices_to_keep]
        
    save_faiss(new_index, new_metadata)

def extract_text(file):
    """Extract text from PDF or TXT file."""
    text = ""
    if file.name.endswith(".pdf"):
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    elif file.name.endswith(".txt"):
        text = file.getvalue().decode("utf-8")
    return text

def chunk_text(text):
    """Split text into manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_text(text)

def embed_texts(texts, model):
    """Generate embeddings for a list of texts."""
    embeddings = model.encode(texts, show_progress_bar=False)
    return np.array(embeddings).astype("float32")

# Main UI setup
st.title("AI RAG Document Assistant")
st.markdown("Upload documents and ask questions based on their content.")

# Sidebar for controls
with st.sidebar:
    st.header("Document Upload")
    uploaded_files = st.file_uploader("Upload PDFs or TXTs", type=["pdf", "txt"], accept_multiple_files=True)
    
    if st.button("Process Documents"):
        if uploaded_files:
            with st.spinner("Processing documents..."):
                model = load_embedding_model()
                index, metadata = init_or_load_faiss()
                
                new_chunks = []
                new_metadata = []
                
                # Progress bar for files
                progress_bar = st.progress(0)
                for i, file in enumerate(uploaded_files):
                    text = extract_text(file)
                    if not text.strip():
                        continue
                        
                    chunks = chunk_text(text)
                    new_chunks.extend(chunks)
                    # Use a unique identifier for the chunk based on existing metadata length
                    start_idx = len(metadata) + len(new_metadata)
                    new_metadata.extend([{"source": file.name, "chunk_id": start_idx + j, "text": chunk} for j, chunk in enumerate(chunks)])
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                if new_chunks:
                    st.info("Generating embeddings...")
                    embeddings = embed_texts(new_chunks, model)
                    index.add(embeddings)
                    metadata.extend(new_metadata)
                    save_faiss(index, metadata)
                    st.success(f"Added {len(new_chunks)} chunks to the knowledge base!")
                else:
                    st.warning("No readable text found in documents.")
        else:
            st.warning("Please upload files first.")
            
    st.divider()
    
    # Show stats
    index, metadata = init_or_load_faiss()
    st.metric("Total Chunks in DB", len(metadata))
    
    if metadata:
        st.write("**Chunk Distribution:**")
        source_counts = {}
        for meta in metadata:
            src = meta.get("source", "Unknown")
            source_counts[src] = source_counts.get(src, 0) + 1
            
        for source, count in source_counts.items():
            col1, col2 = st.columns([4, 1])
            with col1:
                st.caption(f"📄 {source}: {count} chunks")
            with col2:
                if st.button("X", key=f"del_{source}", help=f"Remove {source}"):
                    remove_source_from_faiss(source, index, metadata)
                    st.rerun()
    
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        if os.path.exists(CHAT_HISTORY_FILE):
            os.remove(CHAT_HISTORY_FILE)
        st.rerun()
        
    if st.button("Clear Knowledge Base"):
        if os.path.exists(INDEX_FILE):
            os.remove(INDEX_FILE)
        if os.path.exists(METADATA_FILE):
            os.remove(METADATA_FILE)
        st.success("Knowledge base cleared!")
        st.rerun()

# Chat Interface
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "context" in message and message["context"]:
            with st.expander("Show Retrieved Context"):
                for ctx in message["context"]:
                    st.markdown(f"**Source:** {ctx['source']}")
                    st.info(ctx['text'])
                    st.divider()

user_query = st.chat_input("Ask a question about your documents...")

if user_query:
    # Display user query
    with st.chat_message("user"):
        st.markdown(user_query)
    
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    save_chat_history(st.session_state.chat_history)
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # Retrieval
        model = load_embedding_model()
        index, metadata = init_or_load_faiss()
        
        retrieved_context = []
        context_text = ""
        
        if index.ntotal > 0:
            query_embedding = embed_texts([user_query], model)
            distances, indices = index.search(query_embedding, k=3)
            
            for idx in indices[0]:
                if idx != -1 and idx < len(metadata):
                    meta = metadata[idx]
                    retrieved_context.append(meta)
                    context_text += f"\n--- Source: {meta['source']} ---\n{meta['text']}\n"
        else:
            context_text = "No documents available in the knowledge base."
            
        # Generation
        groq_client = get_groq_client()
        system_prompt = f"""You are a helpful AI assistant. Use the following context to answer the user's question. 
If the context doesn't contain the answer, say "I don't have enough information from the documents to answer that."

FORMATTING INSTRUCTIONS:
- Keep your answer clear, concise, and well-formatted using Markdown.
- If the user asks for a quiz, list, or multiple-choice questions, ensure EACH option (e.g., A, B, C, D) starts on a NEW line.
- Use bold text for headings or emphasis.
- Use bullet points or numbered lists where appropriate.

CONTEXT:
{context_text}
"""
        try:
            with st.spinner("Thinking..."):
                chat_completion = groq_client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_query}
                    ],
                    model="llama-3.1-8b-instant", # Using Groq's fast llama3.1 model
                    temperature=0.5,
                    max_tokens=1024,
                )
                
            full_response = chat_completion.choices[0].message.content
            message_placeholder.markdown(full_response)
            
            if retrieved_context:
                with st.expander("Show Retrieved Context"):
                    for ctx in retrieved_context:
                        st.markdown(f"**Source:** {ctx['source']}")
                        st.info(ctx['text'])
                        st.divider()
            
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": full_response,
                "context": retrieved_context
            })
            save_chat_history(st.session_state.chat_history)
            
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            message_placeholder.error(error_msg)
            st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
            save_chat_history(st.session_state.chat_history)
