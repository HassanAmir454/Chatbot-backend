from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)

# API Key setup
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise Exception("GEMINI_API_KEY not set")

client = genai.Client(api_key=API_KEY)

# Session storage (in-memory)
sessions = {}
MAX_CONTEXT = 10  # Keep last N messages

def add_to_session(session_id, role, message):
    if session_id not in sessions:
        sessions[session_id] = []
    sessions[session_id].append({
        "role": role,
        "message": message,
        "timestamp": datetime.utcnow().isoformat()
    })
    # Trim context
    if len(sessions[session_id]) > MAX_CONTEXT:
        sessions[session_id] = sessions[session_id][-MAX_CONTEXT:]

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "").strip()
    session_id = data.get("session_id", "default_session")

    if not user_message:
        return jsonify({"reply": "Please send a valid message."}), 400

    # Add user message to session
    add_to_session(session_id, "user", user_message)

    # Prepare conversation context for LLM
    conversation_history = "\n".join(
        [f"{msg['role']}: {msg['message']}" for msg in sessions[session_id]]
    )

    # System prompt for safety & guidance
    system_prompt = (
        "You are a helpful assistant. "
        "Respond clearly and safely to the user. "
        "Do NOT include any intent summaries, classifications, or metadata. "
        "Avoid unsafe, harmful, or offensive content."
    )

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"{system_prompt}\n\nConversation:\n{conversation_history}\nAssistant:"
        )

        bot_reply = response.text.strip()

        # Add bot reply to session
        add_to_session(session_id, "assistant", bot_reply)

        # Return only the reply text
        return jsonify({"reply": bot_reply})

    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(error_msg)
        return jsonify({"reply": error_msg}), 500

if __name__ == "__main__":
    PORT = 5002
    app.run(port=PORT, debug=True)
