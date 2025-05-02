import os
import numpy as np
from typing import List, Dict, Tuple
import glob
from PyPDF2 import PdfReader
from azure.ai.inference import EmbeddingsClient
from azure.core.credentials import AzureKeyCredential

# Setup Azure AI client
endpoint = "https://models.inference.ai.azure.com"
model_name = "embed-v-4-0"
token = os.environ["GITHUB_TOKEN"]

client = EmbeddingsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(token)
)

def compute_embedding(texts: List[str]):
    """Generate embeddings for a list of texts"""
    response = client.embed(
        input=texts,
        model=model_name
    )
    
    # Extract embeddings into a list of numpy arrays for easy processing
    embeddings = [np.array(item.embedding) for item in response.data]
    return embeddings, response.usage

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)

def embed_documents(documents: Dict[str, str]) -> Dict[str, np.ndarray]:
    """Embed a dictionary of documents"""
    doc_ids = list(documents.keys())
    doc_texts = list(documents.values())
    
    embeddings, usage = compute_embedding(doc_texts)
    print(f"Document embedding usage: {usage}")
    
    # Create dictionary mapping document IDs to their embeddings
    doc_embeddings = {doc_id: embedding for doc_id, embedding in zip(doc_ids, embeddings)}
    return doc_embeddings

def search_similar_documents(query: str, doc_embeddings: Dict[str, np.ndarray], top_n: int = 3) -> List[Tuple[str, float]]:
    """Find documents most similar to query"""
    # Get query embedding
    query_embedding, _ = compute_embedding([query])
    query_vector = query_embedding[0]
    
    # Calculate similarity scores
    similarities = []
    for doc_id, doc_vector in doc_embeddings.items():
        similarity = cosine_similarity(query_vector, doc_vector)
        similarities.append((doc_id, similarity))
    
    # Sort by similarity (highest first)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Return top N results
    return similarities[:top_n]

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text content from a PDF file"""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf = PdfReader(file)
            for page in pdf.pages:
                text += page.extract_text() + " "
        return text.strip()
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""

def load_pdf_documents(directory_path: str) -> Dict[str, str]:
    """Load PDF files from a directory and extract their text"""
    documents = {}
    pdf_files = glob.glob(os.path.join(directory_path, "*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {directory_path}")
        return documents
    
    print(f"Found {len(pdf_files)} PDF files.")
    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        print(f"Processing {filename}...")
        text = extract_text_from_pdf(pdf_path)
        if text:
            documents[filename] = text
            print(f"  Extracted {len(text)} characters")
        else:
            print(f"  Could not extract text from {filename}")
    
    return documents

# Example usage with PDF files
if __name__ == "__main__":
    # Directory containing PDF files
    pdf_directory = r"pdf_docs"
    
    # Create the directory if it doesn't exist
    os.makedirs(pdf_directory, exist_ok=True)
    
    print(f"Checking for PDF documents in {pdf_directory}")
    documents = load_pdf_documents(pdf_directory)
    
    if not documents:
        print("No valid PDF documents found. Using sample text documents instead.")
        # Sample documents as fallback
        documents = {
            "doc1.pdf": "Artificial intelligence is transforming various industries",
            "doc2.pdf": "Machine learning algorithms require large datasets",
            "doc3.pdf": "Natural language processing helps computers understand human language",
            "doc4.pdf": "Computer vision enables machines to interpret visual information",
            "doc5.pdf": "Deep learning models are inspired by the human brain"
        }
    
    # Embed documents
    print("\nEmbedding documents...")
    doc_embeddings = embed_documents(documents)
    
    # Search for similar documents
    query = input("\nEnter your search query: ")
    if not query:
        query = "How do computers understand text?"
        print(f"Using default query: '{query}'")
    
    print(f"Searching for documents similar to: '{query}'")
    
    results = search_similar_documents(query, doc_embeddings)
    
    # Display results
    print("\nTop matching documents:")
    for doc_id, similarity in results:
        print(f"Document: {doc_id}")
        print(f"Similarity: {similarity:.4f}")
        print(f"Content preview: {documents[doc_id][:200]}...")
        print("-" * 80)