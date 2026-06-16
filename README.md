# 🚨 SOP Safety Assistant (SafeLab Assistant)

A RAG-based AI assistant designed to provide rapid, step-by-step Standard Operating Procedure (SOP) instructions for laboratory environments. It uses local embeddings to search through SOPs and leverages the Groq API (Llama 3) to generate clear, concise instructions.

## 🌟 Features

- **Retrieval-Augmented Generation (RAG)**: Searches local `sop.txt` using `sentence-transformers`.
- **Fast Inference**: Powered by the Groq API (`llama-3.1-8b-instant`).
- **Emergency Detection**: Automatically triggers Emergency Mode for critical queries (e.g., "fire", "spill", "leak", "explosion").
- **Multiple Interfaces**:
  - **Streamlit App**: An interactive web dashboard.
  - **Custom Web UI (HTML/JS + Flask)**: Includes Voice Input (Speech-to-Text) and Audio Output (Text-to-Speech).
  - **CLI Chatbot**: A simple terminal-based conversational interface.

## 🛠️ Tech Stack

- **Backend / Logic**: Python, Flask, Streamlit
- **Embeddings**: `sentence-transformers` (`all-MiniLM-L6-v2`) & `scikit-learn` (Cosine Similarity)
- **LLM**: Groq API (`llama-3.1-8b-instant`)
- **Frontend**: HTML5, TailwindCSS (via CDN), JavaScript (Web Speech API)

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.8+
- Get a [Groq API Key](https://console.groq.com/keys)

### 2. Setup

Install the required dependencies (you can create a virtual environment first):

```bash
pip install streamlit flask flask-cors python-dotenv groq sentence-transformers scikit-learn
```

Create a `.env` file in the root directory and add your Groq API key:

```env
GROQ_API_KEY=your_api_key_here
```

### 3. Running the Applications

You can run the assistant using any of the following methods:

#### Option A: Streamlit UI
Run the Streamlit application for a clean, built-in dashboard:
```bash
streamlit run app.py
```

#### Option B: Flask API + Custom Web UI (Voice Enabled)
1. Start the backend API:
```bash
python backend_api.py
```
2. Open `index.html` in your web browser. This interface supports microphone input and speaks the responses out loud.

#### Option C: Command Line Interface
Run the simple CLI chatbot in your terminal:
```bash
python chatbot.py
```

## 📄 Managing SOPs
To update the knowledge base, edit the `sop.txt` file. Separate different SOP sections using `---`. The application will automatically embed the updated text on its next run.
