from fastapi import FastAPI
from pydantic import BaseModel
import faiss
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from llama_index.llms.ollama import Ollama

app = FastAPI()

# Function to read the original text chunks from a file
def read_file_chunks(file_name):
    with open(file_name, 'r') as file:
        data = file.read()
    # Extract chunks enclosed in triple double quotes
    chunks = re.findall(r'"""(.*?)"""', data, re.DOTALL)
    return [chunk.strip() for chunk in chunks if chunk.strip()]

# Load the original text chunks
filename = 'chunks.txt'  # Your original text file
original_chunks = read_file_chunks(filename)

# Load the Faiss index
index = faiss.read_index("faiss_index.bin")

# Load the SentenceTransformer model for embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize the Ollama model
llm = Ollama(model="llama3.2:1b", request_timeout=60.0)

# Function to perform similarity search and get the closest matching chunk
def get_closest_chunk(query_text):
    query_embedding = embedding_model.encode(query_text).reshape(1, -1).astype("float32")  # Encode the query

    # Perform the similarity search in Faiss
    k = 1  # Number of nearest neighbors to retrieve
    distances, indices = index.search(query_embedding, k)

    if indices[0][0] != -1 and indices[0][0] < len(original_chunks):
        chunk_index = indices[0][0]
        return original_chunks[chunk_index], distances[0][0]  # Return the chunk and its distance
    else:
        return None, None

# Function to summarize the chunk using Ollama
def summarize_chunk(chunk_text):
    # Generate the summary using Ollama
    response = llm.complete(f"Elaborate the issue in detail and provide the resolution steps based on the given data: {chunk_text}")
    # Access the text directly from the response object
    summary_text = response.text  # Access the 'text' attribute directly
    return summary_text

# Pydantic model for request body
class QueryRequest(BaseModel):
    query_text: str

@app.post("/search")
async def search_query(request: QueryRequest):
    closest_chunk, distance = get_closest_chunk(request.query_text)
    
    if closest_chunk:
        summary = summarize_chunk(closest_chunk)
        # Construct a user-friendly response
        return {
            "summary": summary,  # Clean summary text only
            "relevance_score": round(float(distance), 2)  # Optional: Adding a rounded relevance score
        }
    else:
        return {
            "error": "No matching chunk found."
        }
