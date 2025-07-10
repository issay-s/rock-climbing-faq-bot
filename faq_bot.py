import os
import faiss
import numpy as np 
import tiktoken
import time
import pickle
from typing import List, Tuple, Optional
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI 
import gradio as gr

# Load environment variables
load_dotenv()

# Use new OpenAI client initialization
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMBED_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 300  # in tokens
CHUNK_OVERLAP = 50  
TOP_K = 3
INDEX_FILE = "rag_index.pkl"  

def get_embedding(text: str, max_retries: int = 3) -> Optional[List[float]]:
    """
    Convert text to embedding vector using OpenAI's embedding API.
    Includes retry logic for API failures.
    
    Args:
        text: The text to embed
        max_retries: Maximum number of retry attempts
    
    Returns:
        List of floats representing the embedding, or None if all retries fail
    """
    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(model=EMBED_MODEL, input=[text])
            return response.data[0].embedding
        except Exception as e:
            print(f"Embedding attempt {attempt + 1} failed: {e}")
        if attempt == max_retries - 1:
            print(f"Failed to get embedding after {max_retries} attempts")
            return None
            time.sleep(2 ** attempt)  # Exponential backoff
    return None

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split text into overlapping chunks based on token count.
    Overlapping chunks help preserve context across boundaries.
    
    Args:
        text: The text to chunk
        chunk_size: Maximum tokens per chunk
        overlap: Number of tokens to overlap between chunks
    
    Returns:
        List of text chunks
    """
    enc = tiktoken.encoding_for_model("gpt-4")
    tokens = enc.encode(text)
    chunks = []
    
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk = tokens[i:i + chunk_size]
        if len(chunk) > 0: 
            chunks.append(enc.decode(chunk))
    
    return chunks

def load_documents(folder_path: str) -> List[Tuple[str, str]]:
    """
    Load all text and markdown files from a directory recursively.
    
    Args:
        folder_path: Path to the directory containing documents
    
    Returns:
        List of tuples containing (file_path, file_content)
    """
    docs = []
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"Warning: Folder '{folder_path}' does not exist")
        return docs
    
    for filepath in folder.rglob("*.*"):
        if filepath.suffix in [".txt", ".md"]:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if content.strip():  
                        docs.append((str(filepath), content))
            except Exception as e:
                print(f"Error reading file {filepath}: {e}")
                continue
    
    print(f"Loaded {len(docs)} documents")
    return docs

def build_index(documents: List[Tuple[str, str]]) -> Tuple[faiss.Index, List[str], List[str]]:
    """
    Build a FAISS index from documents by chunking text and creating embeddings.
    
    Args:
        documents: List of (file_path, content) tuples
    
    Returns:
        Tuple of (faiss_index, text_chunks, metadata)
    """
    index = faiss.IndexFlatL2(1536)
    texts, metadata = [], [] 
    
    total_chunks = 0
    for path, doc in documents:
        try:
            chunks = chunk_text(doc)
            print(f"Processing {len(chunks)} chunks from {Path(path).name}")
            
            for chunk in chunks:
                emb = get_embedding(chunk)
                if emb is not None: 
                    index.add(np.array([emb], dtype='float32'))
                    texts.append(chunk)
                    metadata.append(path)
                    total_chunks += 1
                else:
                    print(f"Skipping chunk due to embedding failure")
        except Exception as e:
            print(f"Error processing document {path}: {e}")
            continue
    
    print(f"Built index with {total_chunks} total chunks")
    return index, texts, metadata

def search_index(query: str, index: faiss.Index, texts: List[str], metadata: List[str]) -> List[str]:
    """
    Search the FAISS index for chunks most similar to the query.
    
    Args:
        query: The search query
        index: FAISS index
        texts: List of text chunks
        metadata: List of source file paths
    
    Returns:
        List of relevant text chunks
    """
    emb = get_embedding(query)
    if emb is None:
        print("Failed to embed query, returning empty results")
        return []
    
    try:
        D, I = index.search(np.array([emb], dtype='float32'), TOP_K)
        
        results = []
        for i in I[0]:
            if i < len(texts):  # Ensure index is valid
                results.append(texts[i])
        
        return results
    except Exception as e:
        print(f"Search failed: {e}")
        return []

def ask_gpt(question: str, context: str) -> str:
    """
    Generate an answer using GPT based on the provided context and question.
    
    Args:
        question: The user's question
        context: Retrieved relevant context from documents
    
    Returns:
        GPT's response as a string
    """
    system_prompt = """You are a helpful AI assistant answering questions based on internal documentation. 
    Use the provided context to answer questions accurately. If the context doesn't contain enough information 
    to answer the question, say so clearly."""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
    ]
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"GPT API call failed: {e}")
        return "I'm sorry, I couldn't generate a response due to an API error."

def save_index(index: faiss.Index, texts: List[str], metadata: List[str], filename: str = INDEX_FILE):
    """
    Save the FAISS index and associated data to disk for persistence.
    
    Args:
        index: FAISS index to save
        texts: List of text chunks
        metadata: List of source file paths
        filename: File to save to
    """
    try:
        with open(filename, 'wb') as f:
            pickle.dump((index, texts, metadata), f)
        print(f"Index saved to {filename}")
    except Exception as e:
        print(f"Failed to save index: {e}")

def load_index(filename: str = INDEX_FILE) -> Optional[Tuple[faiss.Index, List[str], List[str]]]:
    """
    Load a previously saved FAISS index from disk.
    
    Args:
        filename: File to load from
    
    Returns:
        Tuple of (faiss_index, text_chunks, metadata) or None if loading fails
    """
    try:
        if Path(filename).exists():
            with open(filename, 'rb') as f:
                index, texts, metadata = pickle.load(f)
            print(f"Index loaded from {filename}")
            return index, texts, metadata
        else:
            print(f"No existing index found at {filename}")
            return None
    except Exception as e:
        print(f"Failed to load index: {e}")
        return None
    
def run_query(query: str, index, texts, metadata) -> str:
    if not query.strip():
        return "Please enter a question."
    
    context_chunks = search_index(query, index, texts, metadata)
    
    if not context_chunks:
        return "No relevant information found in the documents."
    
    combined_context = "\n---\n".join(context_chunks)
    answer = ask_gpt(query, combined_context)
    
    return answer + f"\n\n(Used {len(context_chunks)} context chunks)"

# Global variables for Gradio interface
global_index = None
global_texts = None
global_metadata = None

def chat_response(message, history):
    """Gradio chat function"""
    global global_index, global_texts, global_metadata
    
    if global_index is None:
        return "System not initialized. Please restart the application."
    
    response = run_query(message, global_index, global_texts, global_metadata)
    return response

# -------- Main Interaction Loop --------
if __name__ == "__main__":
    print("RAG Chatbot starting up...")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        exit(1)
    
    print("Checking for existing index...")
    loaded_data = load_index()
    
    if loaded_data and input("Use existing data? (type \"yes\" for yes) ") == "yes":
        index, texts, metadata = loaded_data
        print(f"Using existing index with {len(texts)} chunks")
    else:
        print("Building new index...")
        documents = load_documents("context")
        
        if not documents:
            print("No documents found in 'context' directory")
            exit(1)
        
        index, texts, metadata = build_index(documents)
        
        save_index(index, texts, metadata)
    
    # Set global variables for Gradio
    global_index = index
    global_texts = texts
    global_metadata = metadata
    
    print("\nChatbot ready! Type your questions or 'exit' to quit.")
    print("=" * 50)

    # Check if we want to run Gradio interface
    use_gradio = input("Use Gradio web interface? (type \"yes\" for yes, anything else for terminal): ").strip().lower()
    
    if use_gradio == "yes":
        # Create simple Gradio interface
        iface = gr.ChatInterface(
            fn=chat_response,
            title="Texas Rock Climbing FAQ Chatbot",
            type="messages"
        )
        
        print("Starting Gradio interface...")
        iface.launch(server_port=7860, server_name="0.0.0.0")
    else:
        # Original terminal interface
        while True:
            query = input("\nAsk a question (or type 'exit'): ").strip()
            
            if query.lower() == 'exit':
                print("Goodbye!")
                break

            print("Processing...")
            response = run_query(query, index, texts, metadata)
            print(f"\nAnswer:\n{response}")



