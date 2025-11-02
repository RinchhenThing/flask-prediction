from flask import Flask, request, jsonify
import torch
import numpy as np
import re
from gensim.models import KeyedVectors
from model import GRUClassifier

app = Flask(__name__)

# ------------------ Load Word2Vec and GRU model ------------------
print("🧠 Loading Word2Vec...")
w2v = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin.gz", binary=True)
EMBED_DIM = w2v.vector_size

model = GRUClassifier(EMBED_DIM)
model.load_state_dict(torch.load("gru_classifier.pth", map_location="cpu"))
model.eval()
print("✅ Model ready.")

# ------------------ Preprocessing functions ------------------
TOKEN_PATTERN = re.compile(r"[A-Za-z']+")
def tokenize(s): return [w.lower() for w in TOKEN_PATTERN.findall(s)]

def text_to_sequence(text, keyed_vectors, max_len=100):
    tokens = tokenize(text)[:max_len]
    vectors = []
    for token in tokens:
        if token in keyed_vectors:
            vectors.append(keyed_vectors[token])
        else:
            vectors.append(np.zeros(keyed_vectors.vector_size, dtype=np.float32))
    if vectors:
        return torch.tensor(np.array(vectors), dtype=torch.float32)
    else:
        return torch.zeros((1, keyed_vectors.vector_size), dtype=torch.float32)

def predict_fake_real(text):
    seq = text_to_sequence(text, w2v, 100)
    seq = seq.unsqueeze(0)
    with torch.no_grad():
        output = model(seq)
        prob = torch.sigmoid(output).item()
        label = "REAL" if prob >= 0.5 else "FAKE"
    return label, prob

# ------------------ Flask Routes ------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Fake News Detector API is running."})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' in request."}), 400

    text = data["text"]
    label, prob = predict_fake_real(text)
    return jsonify({
        "prediction": label,
        "confidence": round(prob, 3)
    })

# ------------------ Run ------------------
if __name__ == "__main__":
    # Listen on all interfaces so curl or browser can reach it
    app.run(host="0.0.0.0", port=5000, debug=False)
