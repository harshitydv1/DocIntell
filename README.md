# DocIntel: My Personal AI Document Assistant

Hey! Welcome to **DocIntel**. I built this project because I needed a fast, reliable way to "chat" with my PDFs and text files without having to pay for heavy cloud databases or wait forever for responses.

Basically, this is a Retrieval-Augmented Generation (RAG) app. You upload your documents, and you can instantly ask questions about them. What makes this cool is that the database runs entirely locally on your machine, while the actual brain (the AI) runs on Groq for blazing-fast conversational responses.

## How it works under the hood

When you upload a file, the app reads the text and chops it up into smaller, meaningful pieces (we call these "chunks"). I set it to around 400 tokens per chunk with a little bit of overlap so it captures just the right amount of context.

Next, it takes those chunks and converts them into numbers (embeddings) using a lightweight model running directly on your computer. These embeddings are stored in a local, file-based database called **FAISS**.

When you type a question, the app searches FAISS for the 3 most relevant chunks of text. It then grabs those chunks and sends them along with your question to **Llama 3** (via the Groq API). Groq processes it incredibly fast and spits out an accurate answer based *only* on your documents. 

## The Tech Stack I Used

* **Streamlit**: I used this for the entire frontend and backend. It makes building a chat UI with Python super easy.
* **FAISS**: Facebook's vector database. It's totally free, runs locally on your hard drive, and is insanely fast for searching text embeddings.
* **Sentence Transformers**: Specifically the `all-MiniLM-L6-v2` model. It's small enough to run locally without a massive GPU, but smart enough to understand semantic meaning.
* **Groq (Llama-3.1-8b)**: This handles the actual text generation. Groq's API is known for its crazy speed, which makes the chat interface feel incredibly responsive.
* **Langchain**: I used their `RecursiveCharacterTextSplitter` to handle the document chunking so it doesn't accidentally chop sentences in half.
* **PyPDF2**: Just a simple library to pull raw text out of the PDFs you upload.

## Some Cool Features

- **No lost data**: Your indexed documents and chat history are saved locally. If you accidentally refresh the page, you won't lose your chat or have to re-upload everything.
- **Manage your files**: You can see exactly how many text chunks belong to each file right in the sidebar. There's also a handy 'X' button to delete a specific file's data from the database if you don't need it anymore.
- **Trust but verify**: Every time the AI answers a question, you can click "Show Retrieved Context" to see the exact paragraphs it used to come up with its answer.

## How to run it yourself

If you want to spin this up locally, it's pretty straightforward.

First, set up a virtual environment so you don't mess up your global Python packages:
```bash
python3 -m venv venv
source venv/bin/activate
```

Install the required libraries:
```bash
pip install -r requirements.txt
```

You'll need a free Groq API key. Create a file named `.env` in the main folder and add your key like this:
```env
GROQ_API_KEY=your_groq_api_key_here
```

Finally, just run the app:
```bash
streamlit run app.py
```

That's it! Let me know if you run into any issues.
