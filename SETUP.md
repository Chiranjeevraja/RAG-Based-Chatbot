# YouTube RAG Chatbot — Setup Guide

## Prerequisites

- Python 3.10+
- Node.js 18+
- [uv](https://github.com/astral-sh/uv) *(recommended Python package manager)*
- OpenAI API key
- Qdrant Cloud account + API key *(free tier available)*
- YouTube Data API v3 key *(for comments — transcript works without it)*

### Install uv (if not already installed)

```bash
pip install uv
```

---

## 1. Clone / open the project

```bash
cd RAG-Based-Chatbot
```

## 2. Create your `.env` file

```bash
cp .env.example .env
```

Edit `.env` and fill in:

```env
OPENAI_API_KEY=sk-...
QDRANT_URL=https://your-cluster-id.us-east4-0.gcp.cloud.qdrant.io
QDRANT_API_KEY=your-qdrant-api-key
YOUTUBE_API_KEY=AIza...
```

### Getting a Qdrant Cloud API key

1. Sign up free at [cloud.qdrant.io](https://cloud.qdrant.io/)
2. Create a cluster (free tier is enough)
3. Copy the **Cluster URL** and **API Key** from the cluster dashboard
4. Paste both into `.env`

### Getting a YouTube Data API key

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a project → **APIs & Services** → **Enable APIs**
3. Search for **YouTube Data API v3** → Enable it
4. **Credentials** → **Create Credentials** → **API Key**
5. Paste it into `.env`

> **Note:** If you don't have a YouTube API key, the app still works — it will index the transcript only (no comments).

---

## 3. Backend setup

```bash
cd backend

# Create virtual environment with uv
uv venv

# Activate it
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt

# Start the server
python main.py
```

Backend runs at **http://localhost:8000**
API docs available at **http://localhost:8000/docs**

---

## 4. Frontend setup

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at **http://localhost:5173**

---

## 5. Quick start (Windows)

Double-click `start.bat` — it starts both servers automatically.

> **Note:** The batch file uses plain `pip`. If you prefer uv, set up the venv manually using Step 3 above, then run `python main.py` from the activated venv.

---

## How it works

```
YouTube URL
    │
    ├─► youtube-transcript-api  →  Transcript text
    │
    └─► YouTube Data API v3     →  Top comments
                │
                ▼
         Text Chunking
         (500 words, 100 overlap)
                │
                ▼
      OpenAI Embeddings
      (text-embedding-3-small)
                │
                ▼
        Qdrant Cloud
                │
                ▼
    User asks a question
                │
                ▼
    Vector similarity search
    → Top-6 relevant chunks
                │
                ▼
    GPT-4o-mini answers
    with streamed response
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Health check |
| `POST` | `/api/process` | Index a YouTube video |
| `POST` | `/api/chat` | Ask a question (non-streaming) |
| `POST` | `/api/chat/stream` | Ask a question (streaming) |
| `GET` | `/api/videos` | List indexed videos |
| `DELETE` | `/api/videos/{id}` | Remove a video's chunks |
