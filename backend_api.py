import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

from groq import Groq
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- LOAD ENV ----------
load_dotenv()

# ---------- APP ----------
app = Flask(__name__)
CORS(app)

# ---------- GROQ ----------
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ---------- LOAD SOP ----------
with open("sop.txt", "r") as f:
    sop_data = f.read()

sop_sections = [s.strip() for s in sop_data.split('---') if s.strip()]

# ---------- LOAD MODEL ----------
print("Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
sop_embeddings = model.encode(sop_sections)
print("Model ready ✅")


@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.json
        query = data.get("query", "").strip()

        if not query:
            return jsonify({
                "answer": "Please enter a valid query.",
                "emergency": False
            })

        # 🚨 Emergency detection
        emergency_keywords = ["fire", "explosion", "leak", "spill"]
        is_emergency = any(word in query.lower() for word in emergency_keywords)

        # 🔍 Search
        query_embedding = model.encode([query])
        scores = cosine_similarity(query_embedding, sop_embeddings)[0]

        best_index = scores.argmax()
        best_score = scores[best_index]

        # ❌ No match → better UX
        if best_score < 0.4:
            return jsonify({
                "answer": "⚠️ No SOP found. Try keywords like 'fire', 'chemical spill', etc.",
                "emergency": False
            })

        relevant_sop = sop_sections[best_index]

        # 🤖 LLM
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": f"""You are a lab safety SOP assistant.

Use ONLY the SOP below:

{relevant_sop}

Rules:
- Show title
- Step-by-step instructions
- Include warning
- No extra info
"""
                },
                {"role": "user", "content": query}
            ]
        )

        return jsonify({
            "answer": response.choices[0].message.content,
            "emergency": is_emergency
        })

    except Exception as e:
        return jsonify({
            "answer": f"Server error: {str(e)}",
            "emergency": False
        })


if __name__ == "__main__":
    app.run(debug=True)