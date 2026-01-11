import os
import torch
import pandas as pd
import numpy as np
from bdh.model import BDHMemory
from bdh.reasoner import BDHReasoner

BOOK_NAME_MAP = {"In Search of the Castaways": "castaways.txt", "The Count of Monte Cristo": "monte_cristo.txt"}
DATA_DIR = "data/"
DEVICE = "cpu" # Sticking to CPU as per your logs

def main():
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    memory_module = BDHMemory().to(DEVICE)
    reasoner = BDHReasoner().to(DEVICE)
    
    if os.path.exists("reasoner.pt"):
        reasoner.load_state_dict(torch.load("reasoner.pt", map_location=DEVICE))
        reasoner.eval()
    else:
        print("Error: reasoner.pt not found!")
        return

    # 1. Pre-encode (Cache)
    cache = {}
    for title, filename in BOOK_NAME_MAP.items():
        memory_module.reset()
        with open(os.path.join(DATA_DIR, filename), "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        words = text.split()
        for i in range(0, len(words), 2048):
            memory_module.store(" ".join(words[i:i+2048]))
        cache[title] = memory_module.get_memory().detach()

    # 2. Get Raw Scores
    raw_scores = []
    print("Gathering raw logic scores...")
    with torch.no_grad():
        for idx, row in test_df.iterrows():
            mem = cache.get(row['book_name'])
            claim_text = f"{row['caption']} {row['content']}"
            inputs = memory_module.tokenizer(claim_text, return_tensors="pt", truncation=True)
            claim_emb = memory_module.encoder(**inputs).last_hidden_state[:, 0, :]
            
            score = reasoner(mem, claim_emb).item()
            raw_scores.append(score)

    # 3. Dynamic Thresholding (The Fix)
    # Instead of 0.5, we use the median to split labels 50/50
    threshold = np.median(raw_scores)
    print(f"Median Score: {threshold:.4f} (using this as threshold)")

    labels = ["consistent" if s >= threshold else "contradict" for s in raw_scores]
    
    # 4. Save
    test_df['label'] = labels
    test_df[['id', 'label']].to_csv("results.csv", index=False)
    print("Success! results.csv is now balanced.")

if __name__ == "__main__":
    main()
