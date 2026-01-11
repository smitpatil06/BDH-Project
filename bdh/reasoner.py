import torch
import torch.nn as nn

class BDHReasoner(nn.Module):
    def __init__(self, embedding_dim=768, hidden_dim=512):
        super().__init__()
        input_dim = embedding_dim * 2 # memory + claim
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2), # REGULARIZATION: Prevents overfitting
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            
            nn.Linear(hidden_dim // 2, 1)
            # Sigmoid removed: BCEWithLogitsLoss applies it internally for stability
        )

    def forward(self, narrative_mem, claim_emb):
        if narrative_mem.shape[0] != claim_emb.shape[0]:
            narrative_mem = narrative_mem.expand(claim_emb.shape[0], -1)
            
        combined = torch.cat([narrative_mem, claim_emb], dim=-1)
        return self.network(combined)