import os
import torch
import pandas as pd
import numpy as np
from bdh.model import BDHMemory
from bdh.reasoner import BDHReasoner

# --- CONFIG ---
DATA_DIR = "data/"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BOOK_NAME_MAP = {
    "In Search of the Castaways": "castaways.txt", 
    "The Count of Monte Cristo": "monte_cristo.txt"
}
CHUNK_SIZE = 512

def main():
    print(f"Generating inference on: {DEVICE}")
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    
    # Initialize components
    memory_module = BDHMemory().to(DEVICE)
    reasoner = BDHReasoner().to(DEVICE)
    
    # Load the best weights from training
    if os.path.exists("best_bdh_model.pt"):
        checkpoint = torch.load("best_bdh_model.pt", map_location=DEVICE)
        reasoner.load_state_dict(checkpoint['reasoner'])
        memory_module.memory_rnn.load_state_dict(checkpoint['memory_rnn'])
        print("Successfully loaded best_bdh_model.pt")
    else:
        print("Warning: best_bdh_model.pt not found. Using initialized weights.")

    reasoner.eval()
    memory_module.eval()

    # 1. Pre-encode Books (Cache them to save time)
    book_memories = {}
    print("Pre-encoding novels...")
    for title, filename in BOOK_NAME_MAP.items():
        memory_module.reset(DEVICE)
        path = os.path.join(DATA_DIR, filename)
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        words = text.split()
        for i in range(0, len(words), CHUNK_SIZE):
            memory_module.store(" ".join(words[i:i+CHUNK_SIZE]))
        book_memories[title] = memory_module.get_memory().detach()

    # 2. Inference Loop
    raw_scores = []
    print("Processing claims...")
    with torch.no_grad():
        for idx, row in test_df.iterrows():
            # Get cached memory for the specific book
            mem = book_memories.get(row['book_name'])
            
            # Encode Claim
            claim_text = f"{row['caption']} {row['content']}"
            inputs = memory_module.tokenizer(claim_text, return_tensors="pt", truncation=True).to(DEVICE)
            claim_emb = memory_module.encoder(**inputs).last_hidden_state[:, 0, :]
            
            # Generate Logic Score
            score = reasoner(mem, claim_emb).item()
            raw_scores.append(score)

    # 3. Dynamic Thresholding
    # We use the median of raw scores to ensure a balanced 50/50 split
    threshold = np.median(raw_scores)
    labels = ["consistent" if s >= threshold else "contradict" for s in raw_scores]
    
    # 4. Save Final CSV
    test_df['label'] = labels
    test_df[['id', 'label']].to_csv("results.csv", index=False)
    print(f"Success! results.csv generated with threshold: {threshold:.4f}")

if __name__ == "__main__":
    main()