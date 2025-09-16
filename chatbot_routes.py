from flask import Blueprint, request, jsonify, render_template
import requests
from datetime import datetime
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Blueprint for chatbot
chatbot_bp = Blueprint("chatbot", __name__)

# Groq API configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # <-- replace with your key
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

@chatbot_bp.route("/chatbot")
def chatbot_page():
    return render_template("chatbot.html")

@chatbot_bp.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        user_message = data.get("message", "")

        if not user_message:
            return jsonify({"error": "No message provided"}), 400

        # Groq payload
        payload = {
            "model": "llama-3.1-8b-instant",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a supportive mental health chatbot. Respond in short, clear, and friendly sentences."
                },
                {"role": "user", "content": user_message}
            ],
            "max_tokens": 300,
            "temperature": 0.7
        }

        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }

        response = requests.post(GROQ_API_URL, headers=headers, json=payload)

        if response.status_code == 200:
            response_data = response.json()
            bot_message = response_data["choices"][0]["message"]["content"]
            return jsonify({
                "response": bot_message,
                "timestamp": datetime.now().isoformat()
            })
        else:
            return jsonify({"error": "Failed to get response from AI"}), 500

    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return jsonify({"error": "Internal server error"}), 500
