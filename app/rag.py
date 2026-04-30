import os
import chromadb
from chromadb.utils import embedding_functions
import pymupdf
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class ProspectusRAG:
    def __init__(self):
        # Setup ChromaDB
        self.db_path = "outputs/vector_db"
        os.makedirs(self.db_path, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.db_path)
        
        # Using Default (Local) Embedding Function for zero-config demo reliability
        # This runs on-device and doesn't require an API key or deployment
        self.collection = self.client.get_or_create_collection(
            name="prospectuses"
        )

    def ingest_pdf(self, file_path: str, fund_name: str):
        """Extracts text from PDF, chunks it, and stores in vector DB."""
        doc = pymupdf.open(file_path)
        chunks = []
        metadatas = []
        ids = []
        
        chunk_size = 1000 # characters
        overlap = 200
        
        text = ""
        for page in doc:
            text += page.get_text()
            
        # Basic chunking
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            chunks.append(chunk)
            metadatas.append({"fund": fund_name, "source": Path(file_path).name})
            ids.append(f"{fund_name}_{i}")
            
        if chunks:
            self.collection.upsert(
                documents=chunks,
                metadatas=metadatas,
                ids=ids
            )
        return len(chunks)

    def query(self, query_text: str, fund_name: str = None):
        """Queries the vector DB for relevant context."""
        where_filter = {"fund": fund_name} if fund_name else None
        
        results = self.collection.query(
            query_texts=[query_text],
            n_results=5,
            where=where_filter
        )
        
        return results["documents"][0] if results["documents"] else []

def query_prospectus_semantics(query: str, fund_name: str = None) -> str:
    """External wrapper for the AI to call."""
    from openai import AzureOpenAI
    rag = ProspectusRAG()
    
    # 1. Retrieve context
    context_chunks = rag.query(query, fund_name)
    if not context_chunks:
        return "No relevant information found in the prospectuses."
    
    context_text = "\n---\n".join(context_chunks)
    
    # 2. Synthesize with GPT-4o
    client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version="2024-08-01-preview",
        api_key=os.getenv("AZURE_OPENAI_API_KEY")
    )
    
    prompt = f"""
    You are a Financial Document Expert. Answer the following question based ONLY on the provided context from the fund prospectus.
    If the answer isn't in the context, say you don't know.
    
    CONTEXT:
    {context_text}
    
    QUESTION: {query}
    
    Provide a professional, detailed answer with specific citations if available.
    """
    
    response = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    return response.choices[0].message.content
