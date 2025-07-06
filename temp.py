import google.generativeai as genai
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

# --- Configuration ---
# Replace with your actual Google API key
GOOGLE_API_KEY = "AIzaSyAshsfuDMTbF5CS9tXy_iwoHf3f4iXCOTE" 
# --- End Configuration ---

try:
    genai.configure(api_key=GOOGLE_API_KEY)
except AttributeError:
    print("Please set your GOOGLE_API_KEY in the script.")
    exit()

# 1. Sample Text Data
texts = [
    "The quick brown fox jumps over the lazy dog.",
    "A stitch in time saves nine.",
    "Actions speak louder than words.",
    "All that glitters is not gold.",
    "The early bird catches the worm."
]

# 2. Generate Embeddings using Gemini
print("Generating embeddings with Gemini...")
try:
    result = genai.embed_content(
        model="models/embedding-001",
        content=texts,
        task_type="retrieval_document"
    )
    embeddings = np.array(result['embedding'])
    print(f"Successfully generated {len(embeddings)} embeddings.")
except Exception as e:
    print(f"An error occurred during embedding generation: {e}")
    exit()

# 3. Reduce Dimensionality with t-SNE
print("Reducing dimensionality with t-SNE...")
tsne = TSNE(n_components=2, random_state=42, perplexity=min(len(texts) - 1, 30))
embeddings_2d = tsne.fit_transform(embeddings)
print("Dimensionality reduction complete.")

# 4. Plot the Embeddings
print("Plotting the embeddings...")
plt.figure(figsize=(10, 8))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7)

# Add labels to the points
for i, text in enumerate(texts):
    plt.annotate(text, (embeddings_2d[i, 0], embeddings_2d[i, 1]), fontsize=9)

plt.title("2D Visualization of Gemini Vector Embeddings")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.grid(True)
plt.show()
print("Plot displayed. Close the plot window to exit.")
