import os
import json
import requests
import numpy as np
import faiss

# Config
GEMINI_API_KEY = "AIzaSyAshsfuDMTbF5CS9tXy_iwoHf3f4iXCOTE"
EMBED_MODEL = "models/embedding-001"
EMBED_URL = f"https://generativelanguage.googleapis.com/v1beta/{EMBED_MODEL}:embedContent"
EMBED_DIM = 768  # Gemini gives 768-dimensional vectors

class ContextEngine:
    def __init__(self):
        self.index = faiss.IndexFlatL2(EMBED_DIM)
        self.texts = []  # Keep track of original entries

    def embed_text(self, text):
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": GEMINI_API_KEY
        }
        body = {
            "model": EMBED_MODEL,
            "content": {
                "parts": [{"text": text}]
            }
        }

        response = requests.post(EMBED_URL, headers=headers, data=json.dumps(body))
        response.raise_for_status()
        vector = response.json()["embedding"]["values"]
        return np.array(vector, dtype=np.float32)

    def add_context(self, text):
        vector = self.embed_text(text)
        self.index.add(np.array([vector]))
        self.texts.append(text)
        print(f"✅ Stored: {text}")

    def retrieve(self, query, top_k=1):
        query_vector = self.embed_text(query)
        D, I = self.index.search(np.array([query_vector]), top_k)
        results = [(self.texts[i], D[0][idx]) for idx, i in enumerate(I[0])]
        return results
