from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain import PromptTemplate
from langchain.llms import CTransformers
import os
from waitress import serve

# Load environment variables from .env file
load_dotenv()

# Get API keys from environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
LLAMA_API_KEY = os.getenv("LLAMA_API_KEY")

# Ensure API keys are loaded
if not PINECONE_API_KEY or not LLAMA_API_KEY:
    raise ValueError("Missing API keys! Check your .env file.")

# Set API keys as environment variables
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["LLAMA_API_KEY"] = LLAMA_API_KEY

# Initialize Flask app
app = Flask(__name__)

# Load embeddings
embeddings = download_hugging_face_embeddings()

# Pinecone index name
index_name = "medicalbot"

# Initialize Pinecone vector store
docsearch = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)

# Create retriever with optimized `k`
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 1})  # Reduced from k=3

# Define Prompt Template
prompt_template = """
You are a helpful AI assistant. Answer the following question based on the provided context:

Context: {context}
Question: {question}

Provide a concise and accurate response.
"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs = {"prompt": PROMPT}

# Load Llama model locally
model_path = "./llama/llama-2-7b-chat.ggmlv3.q4_0.bin"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

llm = CTransformers(
    model=model_path,
    model_type="llama",
    config={'max_new_tokens': 512, 'temperature': 0.8}  # Reduced max tokens from 1024 to 512
)

# Define Retrieval-QA Chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET"])
def chat():
    msg = request.args.get("msg", "").strip()
    if not msg:
        return "Error: Empty input."

    # Count tokens before sending query
    token_count = llm.get_num_tokens(msg)
    print(f"User Query Token Count: {token_count}")

    # Enforce token limit
    max_input_tokens = 400
    if token_count > max_input_tokens:
        return "Error: Query is too long. Please shorten your input."

    # Get AI response
    try:
        result = qa.invoke({"query": msg})  # Using `invoke()` instead of `__call__()`
        response_text = result.get("result", "Sorry, I couldn't find an answer.")
    except Exception as e:
        response_text = f"Error processing request: {str(e)}"

    print(f"User: {msg}")
    print(f"AI Response: {response_text}")

    return str(response_text)

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=8080, debug=True)
    # serve(app, host="127.0.0.1", port=8080)
