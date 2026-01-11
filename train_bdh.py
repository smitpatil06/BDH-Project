import os
import sys
import torch
import pandas as pd
from tqdm import tqdm
from bdh.model import BDHMemory  # Fixed Import
from bdh.reasoner import BDHReasoner  # Fixed Import
from sklearn.model_selection import train_test_split

# --- CONFIG ---
BOOK_NAME_MAP = {"In Search of the Castaways": "castaways.txt", "The Count of Monte Cristo": "monte_cristo.txt"}
LABEL_MAP = {"consistent": 1.0, "contradict": 0.0}
DATA_DIR = "data/"
CHUNK_SIZE = 512
EPOCHS = 10 
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class Logger(object):
    def __init__(self, filename="train_log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w")
    def write(self, message):
        self.terminal.write(message); self.log.write(message); self.log.flush()
    def flush(self): pass

def main():
    sys.stdout = Logger("train_log.txt")
    print(f"Using device: {DEVICE}")
    
    # Internal Split: Use 85% for training, 15% for validation tracking
    full_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    train_df, val_df = train_test_split(full_df, test_size=0.15, random_state=42)
    
    memory_module = BDHMemory().to(DEVICE)
    reasoner = BDHReasoner().to(DEVICE)
    
    optimizer = torch.optim.Adam(
        list(reasoner.parameters()) + list(memory_module.memory_rnn.parameters()), 
        lr=LR
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
    
    # MIXED PRECISION: Updated for numerical stability (BCEWithLogitsLoss)
    scaler = torch.amp.GradScaler('cuda')
    loss_fn = torch.nn.BCEWithLogitsLoss()

    for epoch in range(EPOCHS):
        reasoner.train()
        total_loss = 0
        progress_bar = tqdm(train_df.iterrows(), total=len(train_df), desc=f"Epoch {epoch+1}")
        
        for idx, row in progress_bar:
            memory_module.reset(DEVICE)
            novel_text = open(os.path.join(DATA_DIR, BOOK_NAME_MAP[row['book_name']]), "r", encoding="utf-8", errors="ignore").read()
            words = novel_text.split()
            for i in range(0, len(words), CHUNK_SIZE):
                memory_module.store(" ".join(words[i:i+CHUNK_SIZE]))
            
            # Encode Claim
            claim_text = f"{row['caption']} {row['content']}"
            inputs = memory_module.tokenizer(claim_text, return_tensors="pt", truncation=True).to(DEVICE)
            with torch.no_grad():
                claim_emb = memory_module.encoder(**inputs).last_hidden_state[:, 0, :]

            # Forward + Backward with AMP
            target = torch.tensor([[LABEL_MAP[row['label']]]], device=DEVICE)
            with torch.amp.autocast('cuda'):
                prediction = reasoner(memory_module.get_memory(), claim_emb)
                loss = loss_fn(prediction, target)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(train_df)
        scheduler.step(avg_loss)
        print(f"Epoch {epoch+1} Summary | Loss: {avg_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

    torch.save({
        'reasoner': reasoner.state_dict(),
        'memory_rnn': memory_module.memory_rnn.state_dict()
    }, "reasoner.pt")
    print("Training Complete. Weights saved to reasoner.pt")

if __name__ == "__main__":
    main()