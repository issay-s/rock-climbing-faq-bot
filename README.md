**Overview**  
The chatbot uses a Python-based Retrieval-Augmented Generation (RAG) system to generate contextual answers based on club documentation. It embeds and indexes relevant documents, and uses OpenAI's API to provide accurate, context-aware responses.  
  
**Tech Stack**  
OpenAI API — Generates responses using GPT models  
FAISS — Fast similarity search over embedded document chunks  
tiktoken — Handles token-aware chunking of documents  
Gradio — Frontend UI for chatting with the bot  
Amazon EC2 — Cloud instance to host the application  
  
**Deployment Details**  
The Gradio app is launched on an Amazon EC2 instance.  
Required Python packages are installed on the instance.  
The EC2 instance is configured to allow inbound traffic on port 7860 from 0.0.0.0/0, making the chatbot accessible from any device.  
Environment variables (like the OpenAI API key) are secured using environment variables.  
  
**Features**  
Users can ask natural language questions about TRC (e.g., "How do I become a member?").  
The system retrieves and embeds club documentation into a FAISS index for fast, semantic lookup.  
The chatbot answers using only the most relevant information from the indexed content.  
Option to use either a terminal interface or a Gradio web interface.   
  
**TODO**:  
* Read context only once at the beginning.   
* Move remote server connection to HTTPS  
* [FIXED✅] ~~Move frontend to "chat-style"~~  
* Add more context for GPT to use  
* Deploy using IAC  
