import torch
import torch.nn as nn

class GRUClassifier(nn.Module):
    def __init__(self, embed_dim, hidden_dim=128, num_layers=1, bidirectional=True):
        super().__init__()
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), 1)

    def forward(self, x):
        gru_out, h_n = self.gru(x)
        if self.gru.bidirectional:
            h_n = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        else:
            h_n = h_n[-1,:,:]
        h_n = self.dropout(h_n)
        logits = self.fc(h_n).squeeze(1)
        return logits
