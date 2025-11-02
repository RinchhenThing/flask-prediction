import torch
import numpy as np
import re
from gensim.models import KeyedVectors
from model import GRUClassifier

# -----------------------------
# 1. Load model and embeddings
# -----------------------------
print("🧠 Loading Word2Vec...")
w2v = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin.gz", binary=True)
EMBED_DIM = w2v.vector_size

print("✅ Word2Vec loaded successfully!")
model = GRUClassifier(EMBED_DIM)
model.load_state_dict(torch.load("gru_classifier.pth", map_location="cpu"))
model.eval()
print("✅ GRU model loaded and ready for inference.")

# -----------------------------
# 2. Tokenizer and preprocessing
# -----------------------------
TOKEN_PATTERN = re.compile(r"[A-Za-z']+")

def tokenize(s):
    return [w.lower() for w in TOKEN_PATTERN.findall(s)]

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

# -----------------------------
# 3. Prediction function
# -----------------------------
def predict_fake_real(text):
    seq = text_to_sequence(text, w2v, 100)
    seq = seq.unsqueeze(0)  # add batch dimension
    with torch.no_grad():
        output = model(seq)
        prob = torch.sigmoid(output).item()
        label = "REAL" if prob >= 0.5 else "FAKE"
    return label, prob

# -----------------------------
# 4. Interactive mode (optional)
# -----------------------------
if __name__ == "__main__":
    print("\n💬 Fake News Detector — type or paste text below.")
    print("Type 'q' to quit.\n")

    while True:
        text = input("Enter news text: ")
        if text.lower() == "q":
            break
        label, prob = predict_fake_real(text)
        print(f"Prediction: {label} (confidence: {prob:.2f})\n")
