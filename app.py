import os
import textwrap
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
from dotenv import load_dotenv
import logging

# Load environment variables from .env file
load_dotenv()

# Setup environment variable for HuggingFace API token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv('HUGGINGFACEHUB_API_TOKEN')

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define wrap_text_preserve_newlines function
def wrap_text_preserve_newlines(text, width=110):
    lines = text.split('\n')
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
    wrapped_text = '\n'.join(wrapped_lines)
    return wrapped_text

try:
    # Load and preprocess the document
    logger.info("Loading and preprocessing the document...")
    loader = TextLoader("data.txt")
    document = loader.load()

    # Text Splitting
    text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    docs = text_splitter.split_documents(document)
    
    # Embedding
    logger.info("Creating embeddings and building the FAISS database...")
    embeddings = HuggingFaceEmbeddings()
    db = FAISS.from_documents(docs, embeddings)
    
    # Load the LLM for Q&A
    logger.info("Loading the LLM for question answering...")
    llm = HuggingFaceEndpoint(repo_id="google/flan-t5-xxl", temperature=1.0, max_length=1024)
    chain = load_qa_chain(llm, chain_type="refine")

except Exception as e:
    logger.error("Error during initialization: %s", e)
    raise e

@app.route('/ask', methods=['POST'])
def ask_ai():
    try:
        data = request.json
        question = data.get('userInput')
        if not question:
            return jsonify({'error': 'Invalid input: No question provided'}), 400
        
        logger.info("Received question: %s", question)
        docsResult = db.similarity_search(question)
        answer = chain.run(input_documents=docsResult, question=question)
        wrapped_answer = wrap_text_preserve_newlines(answer)
        
        logger.info("Answer generated: %s", wrapped_answer)
        return jsonify({'answer': wrapped_answer})
    
    except Exception as e:
        logger.error("Error during processing request: %s", e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting Flask server...")
    app.run(debug=True)
