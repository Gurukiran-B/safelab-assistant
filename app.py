import os
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- LOAD API ----------
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ---------- UI FIRST ----------
st.set_page_config(page_title="SOP Safety Assistant")
st.title("🚨 SOP Safety Assistant")

query = st.text_input("Enter your query:")

# ---------- CACHE MODEL ----------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# ---------- CACHE SOP ----------
@st.cache_data
def load_sop():
    with open("sop.txt", "r") as f:
        data = f.read()
    sections = [s.strip() for s in data.split('---') if s.strip()]
    return sections

# Load once
model = load_model()
sop_sections = load_sop()

@st.cache_data
def get_embeddings(sections):
    return model.encode(sections)

sop_embeddings = get_embeddings(sop_sections)

# ---------- PROCESS ----------
if query:
    # 🚨 Emergency detection
    emergency_keywords = ["fire", "explosion", "leak", "spill"]
    is_emergency = any(word in query.lower() for word in emergency_keywords)

    if is_emergency:
        st.markdown("<h3 style='color:red;'>🚨 EMERGENCY MODE</h3>", unsafe_allow_html=True)
    else:
        st.markdown("<h3 style='color:blue;'>Normal Mode</h3>", unsafe_allow_html=True)

    # Search
    query_embedding = model.encode([query])
    scores = cosine_similarity(query_embedding, sop_embeddings)[0]

    best_index = scores.argmax()
    best_score = scores[best_index]

    if best_score < 0.4:
        st.error("I don't know")
    else:
        relevant_sop = sop_sections[best_index]

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": f"""You are a lab safety SOP assistant.

Use ONLY the SOP below:

{relevant_sop}

Give step-by-step instructions with warning.
"""
                },
                {"role": "user", "content": query}
            ]
        )

        if is_emergency:
            st.error(response.choices[0].message.content)
        else:
            st.success(response.choices[0].message.content)