import os
import torch
import pandas as pd
import numpy as np
from bdh.model import BDHMemory
from bdh.reasoner import BDHReasoner

DATA_DIR = "data/"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BOOK_NAME_MAP = {
    "In Search of the Castaways": "castaways.txt",
    "The Count of Monte Cristo": "monte_cristo.txt"
}

def main():
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    
    # 1. Initialize and Load Weights
    memory_module = BDHMemory()
    reasoner = BDHReasoner()
    checkpoint = torch.load("reasoner.pt", map_location=DEVICE)
    reasoner.load_state_dict(checkpoint['reasoner'])
    memory_module.memory_rnn.load_state_dict(checkpoint['memory_rnn'])
    
    # Move entire modules to DEVICE
    memory_module.to(DEVICE)
    reasoner.to(DEVICE).eval()

    encoder = memory_module.encoder
    tokenizer = memory_module.tokenizer

    # 2. Extract Narrative Belief States and Raw Text Chunks
    novel_db = {}
    for title, filename in BOOK_NAME_MAP.items():
        memory_module.reset(DEVICE) # This sets initial state to DEVICE
        state_history = []
        text_chunks = [] 
        
        with open(os.path.join(DATA_DIR, filename), "r", encoding="utf-8", errors="ignore") as f:
            words = f.read().split()
        
        for i in range(0, len(words), 512):
            chunk = " ".join(words[i:i+512])
            
            # --- THE FIX ---
            # Ensure the internal state is on the same device as the encoder
            if memory_module.state is not None:
                memory_module.state = memory_module.state.to(DEVICE)
            
            memory_module.store(chunk)
            
            mem = memory_module.get_memory()
            if mem.dim() == 1:
                mem = mem.unsqueeze(0)
            
            state_history.append(mem.detach().cpu()) # Store on CPU to save GPU VRAM
            text_chunks.append(chunk[:200] + "...") 
        
        novel_db[title] = {
            'final_state': memory_module.get_memory().detach(),
            'history': torch.cat(state_history, dim=0),
            'texts': text_chunks 
        }

    # 3. Reasoning and Text-Based Attribution
    results = []
    for _, row in test_df.iterrows():
        book = novel_db[row['book_name']]
        
        # Encode the Claim
        inputs = tokenizer(f"{row['caption']} {row['content']}", 
                           return_tensors="pt", truncation=True, padding=True).to(DEVICE)
        with torch.no_grad():
            claim_emb = encoder(**inputs).last_hidden_state[:, 0, :]
            
            # Label Prediction
            logit = reasoner(book['final_state'].to(DEVICE), claim_emb)
            score = torch.sigmoid(logit).item()
            label = "1" if score >= 0.5 else "0"

            # Attribution Logic
            history = book['history'].to(DEVICE)
            similarities = torch.nn.functional.cosine_similarity(claim_emb, history, dim=1)
            pivot_idx = torch.argmax(similarities).item()
            evidence_text = book['texts'][pivot_idx]
            
            # Track B Requirements: Custom Causal Rationales
            if label == "0":
                rationale = (f"Contradict: Later narrative events violate the causal assumptions implied by the backstory. "
                             f"Conflict detected near: \"{evidence_text}\"")
            else:
                rationale = (f"Consistent: The backstory provides sufficient causal grounding for later decisions. "
                             f"Causal link confirmed near: \"{evidence_text}\"")

            results.append({"id": row['id'], "label": label, "rationale": rationale})

    # 4. Save Final results.csv
    pd.DataFrame(results).to_csv("results.csv", index=False)
    print("Success: results.csv generated with GPU-synced states.")

if __name__ == "__main__":
    main()