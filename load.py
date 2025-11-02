import torch
from model import GRUClassifier

# the embedding dimension used in training
EMBED_DIM = 300  # from the original script: GoogleNews 300D

# 1. Initialize model with same parameters used in training
model = GRUClassifier(embed_dim=EMBED_DIM, hidden_dim=128, num_layers=1, bidirectional=True)

# 2. Load trained weights
model.load_state_dict(torch.load("gru_classifier.pth", map_location="cpu"))
model.eval()

print("✅ Model loaded successfully!")

# You can now test it with some fake input to check if it runs
sample_input = torch.randn(1, 100, EMBED_DIM)  # (batch_size, seq_len, embed_dim)
output = model(sample_input)
print("Output tensor:", output)
