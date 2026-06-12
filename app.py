import os
import json
from flask import Flask, request, Response, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# ─── CONFIGURATION & INITIALIZATION ──────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_AVAILABLE = False

if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
       
        GEMINI_AVAILABLE = True
        print("[INFO] Gemini API successfully configured.")
    except Exception as e:
        print(f"[WARNING] Failed to configure Gemini API: {e}")
else:
    print("[WARNING] GEMINI_API_KEY not found in environment. Running in Fallback Mode.")

# In-memory session store: { session_id: [ {"role": "user/assistant", "content": "..."} ] }
conversation_sessions = {}

# ─── LOAD KNOWLEDGE BASE & SETUP RAG (TF-IDF) ────────────────────────────────
KB_PATH = os.path.join(os.path.dirname(__file__), "knowledge_base/clr_knowledge_base.json")
knowledge_base = []
kb_documents = []

if os.path.exists(KB_PATH):
    try:
        with open(KB_PATH, "r", encoding="utf-8") as f:
            knowledge_base = json.load(f)
            # Combine title and content to make search more robust
            kb_documents = [f"{doc['title']}: {doc['content']}" for doc in knowledge_base]
        print(f"[INFO] Loaded {len(knowledge_base)} documents from knowledge base.")
    except Exception as e:
        print(f"[ERROR] Failed to load knowledge base: {e}")
else:
    print("[WARNING] clr_knowledge_base.json not found! RAG features will be disabled.")

# Initialize Vectorizer if documents exist
vectorizer = None
tfidf_matrix = None
if kb_documents:
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(kb_documents)


def retrieve_relevant_docs(query, top_k=3):
    """Retrieves the most semantically relevant documents using TF-IDF cosine similarity."""
    if not vectorizer or not kb_documents:
        return []
    
    try:
        query_vec = vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Filter out completely irrelevant docs (similarity score of 0)
        results = [knowledge_base[idx] for idx in top_indices if similarities[idx] > 0]
        return results
    except Exception as e:
        print(f"[ERROR] RAG Retrieval failed: {e}")
        return []


# ─── CORE GENERATION FUNCTIONS ───────────────────────────────────────────────
def build_rag_prompt(user_message, retrieved_docs):
    """Constructs a prompt injecting library context."""
    context_str = ""
    for doc in retrieved_docs:
        context_str += f"[{doc['category']}] {doc['title']}:\n{doc['content']}\n\n"
        
    prompt = (
        "You are the CLR Chatbot, an official virtual assistant for the Covenant University Centre for Learning Resources.\n"
        "Use the following pieces of retrieved library context to answer the user's question accurately.\n"
        "If the answer cannot be found in the context, use your general knowledge but prioritize library guidelines.\n"
        "Be polite, helpful, and concise.\n\n"
        f"--- LIBRARY CONTEXT ---\n{context_str}---\n\n"
        f"User Question: {user_message}"
    )
    return prompt


def generate_gemini_stream(user_message, retrieved_docs, history):
    """Streams responses from Gemini Flash."""
    prompt = build_rag_prompt(user_message, retrieved_docs)
    
    # Format chat history for Gemini's structured API format
    formatted_contents = []
    for turn in history:
        # Map roles correctly to what Gemini expects ('user' or 'model')
        role = "user" if turn["role"] == "user" else "model"
        formatted_contents.append({"role": role, "parts": [turn["content"]]})
        
    # Append current turn prompt
    formatted_contents.append({"role": "user", "parts": [prompt]})
    
    model = genai.GenerativeModel(model_name="gemini-2.5-flash")
    response_stream = model.generate_content(formatted_contents, stream=True)
    
    for chunk in response_stream:
        if chunk.text:
            yield chunk.text


def generate_fallback_response(user_message, retrieved_docs):
    """Static fallback mechanism when Gemini API is offline or unconfigured."""
    if retrieved_docs:
        main_match = retrieved_docs[0]
        return (
            f"I am currently operating in offline mode. Based on our library database regarding "
            f"*{main_match['title']}*:\n\n{main_match['content']}\n\n"
            f"For more help, contact clr@covenantuniversity.edu.ng."
        )
    return (
        "I'm sorry, I'm currently running in offline mode and couldn't locate a precise match "
        "in my local library database for your question. Please try asking using different keywords, "
        "or contact the library desk at library.covenantuniversity.edu.ng."
    )


# ─── API ROUTES ──────────────────────────────────────────────────────────────
@app.route("/api/chat", methods=["POST"])
def handle_chat():
    """Streaming chat endpoint using Server-Sent Events (SSE)."""
    data = request.get_json() or {}
    user_message = data.get("message", "").strip()
    session_id = data.get("session_id", "default")

    if not user_message:
        return jsonify({"error": "Message cannot be empty."}), 400

    # Initialize session history tracking
    if session_id not in conversation_sessions:
        conversation_sessions[session_id] = []

    history = conversation_sessions[session_id]

    # Step 1: Run Document Retrieval
    retrieved_docs = retrieve_relevant_docs(user_message, top_k=3)

    def event_stream():
        full_response = ""
        try:
            if GEMINI_AVAILABLE:
                for chunk in generate_gemini_stream(user_message, retrieved_docs, history):
                    full_response += chunk
                    yield f"data: {chunk}\n\n"
            else:
                fallback = generate_fallback_response(user_message, retrieved_docs)
                full_response = fallback
                yield f"data: {fallback}\n\n"

        except Exception as e:
            print(f"[ERROR] Streaming generation failed: {e}")
            fallback = generate_fallback_response(user_message, retrieved_docs)
            full_response = fallback
            yield f"data: {fallback}\n\n"

        finally:
            # Commit conversational turns to history after stream wraps up
            history.append({"role": "user", "content": user_message})
            history.append({"role": "assistant", "content": full_response})
            
            # Keep history under control (last 20 turns)
            if len(history) > 20:
                conversation_sessions[session_id] = history[-20:]
            yield "data: [DONE]\n\n"

    return Response(event_stream(), mimetype="text/event-stream")


@app.route("/api/reset", methods=["POST"])
def handle_reset():
    """Clear conversation history for a given session."""
    data = request.get_json() or {}
    session_id = data.get("session_id", "default")
    if session_id in conversation_sessions:
        del conversation_sessions[session_id]
    return jsonify({"status": "ok", "message": f"Session '{session_id}' cleared."})


if __name__ == "__main__":
    # Ensure port matches standard local deployments
    app.run(host="0.0.0.0", port=5000, debug=True)