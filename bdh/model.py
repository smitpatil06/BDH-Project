import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class BDHMemory(nn.Module):
    def __init__(self, embedding_dim=768):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.encoder = AutoModel.from_pretrained("distilbert-base-uncased")
        self.embedding_dim = embedding_dim
        
        # UPGRADE: GRUCell learns what to remember vs what to forget
        self.memory_rnn = nn.GRUCell(embedding_dim, embedding_dim)
        self.state = None

    def reset(self, device="cpu"):
        """Initialize memory on the correct device for a new novel."""
        self.state = torch.zeros(1, self.embedding_dim).to(device)

    def store(self, text):
        """Gated update of narrative state."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.encoder.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.encoder(**inputs)
            chunk_emb = outputs.last_hidden_state[:, 0, :]
        
        if self.state.device != chunk_emb.device:
            self.state = self.state.to(chunk_emb.device)
            
        # Update state via learned gates
        self.state = self.memory_rnn(chunk_emb, self.state)

    def get_memory(self):
        return self.state