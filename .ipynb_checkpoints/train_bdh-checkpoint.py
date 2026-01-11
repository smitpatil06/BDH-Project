import os
import torch
import pandas as pd
from tqdm import tqdm
from bdh.model import BDHMemory
from bdh.reasoner import BDHReasoner

# --- CONFIG ---
BOOK_NAME_MAP = {
    "In Search of the Castaways": "castaways.txt",
    "The Count of Monte Cristo": "monte_cristo.txt"
}
LABEL_MAP = {"consistent": 1.0, "contradict": 0.0}
DATA_DIR = "data/"
CHUNK_SIZE = 512
EPOCHS = 2
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_text(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def chunk_text(text, size=512):
    words = text.split()
    for i in range(0, len(words), size):
        yield " ".join(words[i:i+size])

def main():
    print(f"Using device: {DEVICE}")
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    
    # Initialize BDH Components
    memory_module = BDHMemory().to(DEVICE)
    reasoner = BDHReasoner().to(DEVICE)
    
    optimizer = torch.optim.Adam(reasoner.parameters(), lr=LR)
    loss_fn = torch.nn.BCELoss()

    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1} ---")
        for idx, row in df.iterrows():
            # 1. Load Novel
            book_file = BOOK_NAME_MAP.get(row['book_name'])
            if not book_file: continue
            
            novel_text = load_text(os.path.join(DATA_DIR, book_file))
            
            # 2. Build Memory (The BDH part)
            memory_module.reset()
            for chunk in chunk_text(novel_text, CHUNK_SIZE):
                memory_module.store(chunk)
            
            narrative_mem = memory_module.get_memory().to(DEVICE)

            # 3. Encode Claim (Caption + Content)
            claim_text = f"{row['caption']} {row['content']}"
            claim_inputs = memory_module.tokenizer(claim_text, return_tensors="pt", truncation=True).to(DEVICE)
            with torch.no_grad():
                claim_emb = memory_module.encoder(**claim_inputs).last_hidden_state[:, 0, :]

            # 4. Train Reasoner
            target = torch.tensor([[LABEL_MAP[row['label']]]], device=DEVICE)
            prediction = reasoner(narrative_mem, claim_emb)
            
            loss = loss_fn(prediction, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if idx % 5 == 0:
                print(f"Sample {idx} | Loss: {loss.item():.4f} | Pred: {prediction.item():.4f}")

    torch.save(reasoner.state_dict(), "reasoner.pt")
    print("\nTraining Finished. saved as reasoner.pt")

if __name__ == "__main__":
    main()
