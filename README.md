
# Deepseek PDF Chatbot

A application that enables users to upload PDF documents and ask questions about their content using LangChain and Ollama. The system utilizes embeddings and vector storage for efficient document retrieval and provides concise, context-aware answers.

## Features

- PDF document upload and processing
- Text chunking and embedding generation
- Semantic search for relevant context retrieval
- Question answering using the Deepseek language model
- Streamlit-based user interface

## Prerequisites

- Python 3.9+
- Ollama installed and running locally
- The Deepseek model downloaded in Ollama
- LangChain for the document processing pipeline
- Ollama for the local language model hosting
- Streamlit for the web interface framework
- PdfPlumber for PDF file handling

## Installation

1. Clone the repository:

    ```bash
    gh repo clone shwetha-sundar/deepseek-rag
    cd deepseek-ollama
    ```

2. Create and activate a virtual environment:

    ```bash
    python -m venv pyenv
    .\pyenv\Scripts\activate # Non-windows - source .venv/bin/activate
    ```

3. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

## Project Structure

```text
deepseek-rag/
├── main.py                # Main application file (implementing RAG)
├── chat.py                # Streamlit chat application file
├── document_store/        # Directory for uploaded PDFs
├── chroma_store/          # Directory for persistent vector DB
├── requirements.txt       # Project dependencies
├── .gitignore             # Git ignore file
└── README.md              # Project documentation
```

## Usage

1. Start the Ollama service and ensure the Deepseek model is available:

    ```bash
    ollama run deepseek-r1:1.5b
    ollama run nomic-embed-text
    ```

2. Run the Streamlit application:

    ```bash
    streamlit run chat.py
    ```

3. Access the application in your web browser at `http://localhost:8501`

4. Upload a PDF document using the file uploader

5. Ask questions about the document content using the chat input

## Configuration

Key parameters to be aware of:

- Size of text chunks - 500
- Overlap between chunks - 100
- Ollama model to use - "deepseek-r1:1.5b"
- Embedding model to use - "nomic-embed-text"
- Similarity search search results - 10
