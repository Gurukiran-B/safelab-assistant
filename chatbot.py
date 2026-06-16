import os
from groq import Groq
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load API key
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Load SOP data
with open("sop.txt", "r") as f:
    sop_data = f.read()

# Split SOP into sections
sop_sections = [s.strip() for s in sop_data.split('---') if s.strip()]

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")
sop_embeddings = model.encode(sop_sections)

print("SOP Safety Assistant (type 'exit' to quit)\n")

while True:
    user_input = input("You: ")

    if user_input.lower() in ["exit", "quit"]:
        print("Ending chat...")
        break

    # Convert query to vector
    query_embedding = model.encode([user_input])
    scores = cosine_similarity(query_embedding, sop_embeddings)[0]

    best_index = scores.argmax()
    best_score = scores[best_index]

    if best_score < 0.4:
        print("\nBot: I don't know")
        continue

    relevant_sop = sop_sections[best_index]

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
            {"role": "user", "content": user_input}
        ]
    )

    print("\nBot:", response.choices[0].message.content)