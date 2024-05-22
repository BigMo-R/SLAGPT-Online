import os
import textwrap
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import HuggingFaceEndpoint
from dotenv import load_dotenv
import logging

# Load API environment variable from .env file
load_dotenv()

# Setup environment variable for HuggingFace API token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv('HUGGINGFACEHUB_API_TOKEN')

# Initialize Flask app
app = Flask(__name__)
CORS(app) # Allows communication from different parts of the web

# Setup logging / Tracks errors
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define wrap_text_preserve_newlines function
def wrap_text_preserve_newlines(text, width=110):
    lines = text.split('\n')
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
    wrapped_text = '\n'.join(wrapped_lines)
    return wrapped_text

# Calculates maximum number of tokens based on input length
def get_max_new_tokens(input_text, max_total_tokens=1024):
    input_tokens = len(input_text.split())
    return max(0, max_total_tokens - input_tokens)

# Processes documents given
def load_document(path):
    loader = TextLoader(path)
    document = loader.load()
    if not document:
        raise ValueError(f"Loaded document from {path} is empty")
    return document

# Splits document into chunks for easier processing / for neat AI responses
def process_documents(document, chunk_size=1500, chunk_overlap=200):
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(document)
    if not docs:
        raise ValueError("Text splitting resulted in no documents")
    return docs

# Embeds documents
def create_embeddings(doc_texts):
    embeddings = HuggingFaceEmbeddings()
    doc_embeddings = embeddings.embed_documents(doc_texts)
    if not doc_embeddings:
        raise ValueError("Embedding creation resulted in no embeddings")
    return doc_embeddings

# Faiss index for similarity search / context for text
def build_faiss_index(docs):
    logger.info(f"Building FAISS index with docs: {docs}")
    embeddings = HuggingFaceEmbeddings()
    db = FAISS.from_documents(docs, embeddings)
    return db

# Load and preprocess documents
try:
    # Loading multiple documents
    logger.info("Loading and preprocessing documents...")
    doc1 = load_document("data2/data1.txt")
    doc2 = load_document("data2/data2.txt")

    # Process documents
    docs1 = process_documents(doc1)
    docs2 = process_documents(doc2)

    # Combine documents for embedding
    combined_docs = docs1 + docs2
    doc_texts = [doc.page_content for doc in combined_docs]
    doc_embeddings = create_embeddings(doc_texts)

    # Build FAISS index
    db = build_faiss_index(combined_docs)

except Exception as e:
    logger.error("Error during initialization: %s", e)
    raise e

# When the ask button is clicked, the following is commanded and seeks to generate an answer
@app.route('/ask', methods=['POST'])
def ask_ai():
    try:
        data = request.json
        question = data.get('userInput')
        if not question:
            return jsonify({'error': 'Invalid input: No question provided'}), 400
        
        logger.info("Received question: %s", question)
        docsResult = db.similarity_search(question)
        
        # Calculate max_new_tokens based on the input length
        max_new_tokens = get_max_new_tokens(question)
        
        # Initialize the LLM with the calculated max_length
        llm = HuggingFaceEndpoint(
            repo_id="google/flan-t5-xxl",
            temperature=0.7,
            max_length=max_new_tokens
        )
        chain = load_qa_chain(llm, chain_type="refine")
        
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
