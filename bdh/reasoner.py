import torch
import torch.nn as nn

class BDHReasoner(nn.Module):
    def __init__(self, embedding_dim=768, hidden_dim=512):
        super().__init__()
        # Input is concatenated: narrative_mem + claim_emb (768 + 768 = 1536)
        input_dim = embedding_dim * 2
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2), # Prevents overfitting
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid() # Output probability 0-1
        )

    def forward(self, narrative_mem, claim_emb):
        # Ensure batch dimensions match
        if narrative_mem.shape[0] != claim_emb.shape[0]:
            narrative_mem = narrative_mem.expand(claim_emb.shape[0], -1)
            
        # Concatenate memory and claim for joint reasoning
        combined = torch.cat([narrative_mem, claim_emb], dim=-1)
        return self.network(combined)