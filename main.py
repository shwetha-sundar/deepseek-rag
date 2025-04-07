from langchain_chroma import Chroma
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain.schema.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import pandas as pd

PROMPT_TEMPLATE = """You are a skilled assistant. Refer to the provided context to respond to the query.
If uncertain, acknowledge your lack of knowledge. 
Keep your answers brief and factual, using no more than fifty words.
    
Context: {document_context}

Query: {user_query}

Answer:"""

PDF_STORAGE_PATH = 'document_store/'
EMBEDDING_MODEL = OllamaEmbeddings(model="nomic-embed-text")
LANGUAGE_MODEL = OllamaLLM(model="deepseek-r1:1.5b")
CHROMA_PERSIST_DIR = "chroma_store"

DOCUMENT_VECTOR_DB = Chroma(
    embedding_function=EMBEDDING_MODEL,
    persist_directory=CHROMA_PERSIST_DIR,
    collection_name="document_embeddings"
)

def upload_pdfs(files):
    os.makedirs(PDF_STORAGE_PATH, exist_ok=True)
    
    saved_paths = []

    for file in files:
        file_path = os.path.join(PDF_STORAGE_PATH, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        saved_paths.append(file_path)

    return saved_paths

def create_vector_store(file_paths):
    all_chunked_docs = []

    # Loop through each file and process
    for file_path in file_paths:
        loader = PDFPlumberLoader(file_path)
        documents = loader.load()
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            add_start_index=True
        )
        chunked_docs = text_splitter.split_documents(documents)
        all_chunked_docs.extend(chunked_docs)

    # Add all chunked documents to Chroma vector store
    return add_to_chroma(all_chunked_docs)

def retrieve_docs(query, k=10):
    print(DOCUMENT_VECTOR_DB.similarity_search(query, k=k))
    return DOCUMENT_VECTOR_DB.similarity_search(query, k=k)

def generate_answer(user_query, context_documents):
    context_text = "\n\n".join([doc.page_content for doc in context_documents])
    conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    response_chain = conversation_prompt | LANGUAGE_MODEL
    return response_chain.invoke({"user_query": user_query, "document_context": context_text})

def add_to_chroma(chunks: list[Document]):
    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = DOCUMENT_VECTOR_DB.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        DOCUMENT_VECTOR_DB.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print("✅ No new documents to add")

def clear_database():
    all_ids = DOCUMENT_VECTOR_DB._collection.get(include=[])["ids"]
        
    # Delete all using the list of IDs
    if all_ids:
        DOCUMENT_VECTOR_DB._collection.delete(ids=all_ids)
        print("❌ All documents deleted from the database")
    else:
        print("🗑️ No documents to delete")

def display_embeddings():

    collection_data = DOCUMENT_VECTOR_DB._collection.get(include=["embeddings", "documents"])

    # Extract embeddings, documents, and metadata
    embeddings = collection_data["embeddings"]
    documents = collection_data["documents"]

    embeddings_display = [str(embedding[:5]) for embedding in embeddings]  # Display only the first 5 elements of each embedding for readability

    # Create a DataFrame to display the results
    df = pd.DataFrame({
        "Embedding (first 5 values)": embeddings_display,
        "Text chunk": documents
    })

    return df

def calculate_chunk_ids(chunks):

    # This will create IDs like "document_store/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks