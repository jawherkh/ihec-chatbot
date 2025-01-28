import json
import logging
import os
import sys
import requests
from flask import Flask, request, jsonify
from llama_index.core import Document, VectorStoreIndex, Settings, StorageContext
from llama_index.llms.lmstudio import LMStudio
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import load_index_from_storage
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.llms import LLM
from llama_index.core import QueryBundle
from flask_cors import CORS
import uuid
import sqlite3
from llama_index.core import PropertyGraphIndex
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import base64


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
app.secret_key = os.environ.get("FLASK_SECRET_KEY", os.urandom(24))


key = '2legit2overfit1234567890'

# Fixed IV (16 characters for AES-128)
iv = 'BBBBBBBBBBBBBBBB'.encode('utf-8')

# Function to encrypt data
def encrypt(data, key, iv):
    data = pad(data.encode(), 16)  # Pad the data to a multiple of 16 bytes
    cipher = AES.new(key.encode('utf-8'), AES.MODE_CBC, iv)
    encrypted_data = cipher.encrypt(data)
    return base64.b64encode(encrypted_data).decode('utf-8')  # Return as a string

# Function to decrypt data
def decrypt(enc, key, iv):
    enc = base64.b64decode(enc)  # Decode the base64-encoded input
    cipher = AES.new(key.encode('utf-8'), AES.MODE_CBC, iv)
    decrypted_data = cipher.decrypt(enc)
    return unpad(decrypted_data, 16).decode('utf-8')  # Convert bytes to string

def create_session_id():
    return str(uuid.uuid4())


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables to store our components
query_engine = None
llm = None


def initialize_components():
    """Initialize all necessary components for the chatbot"""
    global query_engine, llm

    try:
        # Initialize LLM configuration
        llm = LMStudio(
            model_name="phi-3.1-mini-128k-instruct",
            base_url="http://172.20.10.4:1234/v1",
            temperature=0.5,
            timeout=120.0,
            model_config={
                'protected_namespaces': (),
                'request_timeout': 120.0
            }
        )
        
        # Load and process documents
        documents = load_documents()
        
        # Initialize embedding model
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        
        # Configure settings
        Settings.llm = llm
        Settings.embed_model = embed_model
        
        # Initialize index and query engine
        query_engine = setup_index_and_query_engine(documents)

        logger.info("All components initialized successfully")
        return True

    except Exception as e:
        logger.error(f"Failed to initialize components: {str(e)}")
        return False


def load_documents():
    """Load and process documents from JSON"""
    try:
        with open("dataset.json", "r") as f:
            data = json.load(f)
            
        if "dataset" not in data:
            raise ValueError("JSON file is missing required 'dataset' key")
            
        documents = []
        for entry in data["dataset"]:
            if all(key in entry for key in ["category", "question", "answer"]):
                document = Document(
                    text=entry["answer"],
                    metadata={
                        "category": entry["category"],
                        "question": entry["question"],
                        "entry_id": entry.get("id", "N/A")
                    }
                )
                documents.append(document)
                
        if not documents:
            raise ValueError("No valid documents created")
            
        return documents
            
    except Exception as e:
        logger.error(f"Error loading documents: {str(e)}")
        raise


def setup_index_and_query_engine(documents):
    """Set up the index and query engine"""
    try:
        storage_dir = "./storage"
        
        if not os.path.exists(storage_dir):
            storage_context = StorageContext.from_defaults()
            index = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                show_progress=True,
                similarity_top_k=2
            )
            storage_context.persist(persist_dir=storage_dir)
        else:
            storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
            index = load_index_from_storage(storage_context=storage_context)
            
        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=2
        )
        
        return GuardrailQueryEngine(
            retriever=retriever,
            llm=llm
        )
        
    except Exception as e:
        logger.error(f"Error setting up index and query engine: {str(e)}")
        raise


@app.route('/chat', methods=['POST'])
def chat():
    try:
        if not query_engine:
            return jsonify({"error": "System not initialized"}), 500
        data = decrypt(request.json.get('message', ''), key, iv)
        if not data:
            return jsonify({"error": "Missing question in request"}), 400
        print(data)
        # Generate session ID
        session_id = create_session_id()
        
        logger.info(f"Processing query: '{data}'")
        
        # Use the query engine to retrieve information
        response = query_engine.custom_query(data)
        response = encrypt(response, key, iv)
        
        return jsonify({
            "session_id": session_id,
            "response": str(response)
        })
        
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        return jsonify({
            "error": "Une erreur s'est produite lors du traitement de votre demande."
        }), 500


class GuardrailQueryEngine(CustomQueryEngine):
    def __init__(self, retriever: VectorIndexRetriever, llm: LLM):
        # Initialize the parent class with the required components
        super().__init__(
            retriever=retriever,
            llm=llm,
        )
        # Store the components as instance attributes
        self._retriever = retriever
        self._llm = llm

    @property
    def retriever(self) -> VectorIndexRetriever:
        return self._retriever

    @property
    def llm(self) -> LLM:
        return self._llm

    def custom_query(self, query_str: str):
        logger.info(f"Processing custom query with query: '{query_str}'")

        try:
            # Retrieve relevant context from the vector store
            nodes = self.retriever.retrieve(QueryBundle(query_str))
            logger.info(f"Retrieved {len(nodes)} nodes for the query.")

            # Guardrail: Reject unrelated queries
            if not nodes or nodes[0].score < 0.7:

                system_prompt = system_prompt = """

You are an AI developed to serve as an information source dedicated exclusively to IHEC Carthage. Keep your response short and concise. You provide accurate, relevant, and detailed answers to questions related to the university's programs, services, events, history, and any other related topics. Avoid discussing subjects unrelated to IHEC Carthage. If a question is not relevant to the university, politely inform the user and guide them back to topics about IHEC Carthage.
"""

                return self.llm.complete(system_prompt,max_tokens=100)
            
            # Generate response strictly from the context retrieved from the vector store
            context = "\n".join([n.text for n in nodes])
            logger.info(f"Generating response based on context: {context}")

            # Generate the response using the LLM based on the context
            response = self.llm.complete(
                f"answer with the same language as the question. Contexte: {context}\n\nQuestion: {query_str}\nRéponse: "
            )

            logger.info(f"Response generated: {response.text}")
            return response.text
                
        except Exception as e:
            logger.error(f"Query processing failed: {str(e)}")
            return "Une erreur s'est produite lors du traitement de votre demande."


if __name__ == "__main__":
    if initialize_components():
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        logger.error("Failed to initialize components. Exiting.")
        sys.exit(1)