import os
import sys
import torch
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
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
EPOCHS = 15 
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class Logger(object):
    """Duplicates console output to a text file."""
    def __init__(self, filename="train_log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass

def run_validation(val_df, gold_df, memory_module, reasoner):
    reasoner.eval()
    preds, actuals = [], []
    
    # Validation subset for speed
    sample_df = val_df.sample(n=min(30, len(val_df)))
    for _, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Validating", leave=False):
        gold_match = gold_df[gold_df['id'] == row['id']]
        if gold_match.empty: continue
        
        memory_module.reset(DEVICE)
        book_path = os.path.join(DATA_DIR, BOOK_NAME_MAP[row['book_name']])
        with open(book_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        
        words = text.split()
        for i in range(0, len(words), CHUNK_SIZE):
            memory_module.store(" ".join(words[i:i+CHUNK_SIZE]))
        
        claim_text = f"{row['caption']} {row['content']}"
        inputs = memory_module.tokenizer(claim_text, return_tensors="pt", truncation=True).to(DEVICE)
        claim_emb = memory_module.encoder(**inputs).last_hidden_state[:, 0, :]
        
        score = reasoner(memory_module.get_memory(), claim_emb).item()
        preds.append(1.0 if score >= 0.5 else 0.0)
        actuals.append(LABEL_MAP[gold_match.iloc[0]['label']])
            
    return accuracy_score(actuals, preds), f1_score(actuals, preds)

def main():
    sys.stdout = Logger("train_log.txt") # Start logging to file
    print(f"Using device: {DEVICE}")
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    gold_df = pd.read_csv("gold.csv") #
    
    memory_module = BDHMemory().to(DEVICE)
    reasoner = BDHReasoner().to(DEVICE)
    
    optimizer = torch.optim.Adam(
        list(reasoner.parameters()) + list(memory_module.memory_rnn.parameters()), 
        lr=LR
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2
    )
    
    loss_fn = torch.nn.BCELoss()
    best_f1 = 0.0

    for epoch in range(EPOCHS):
        reasoner.train()
        epoch_loss = 0
        
        progress_bar = tqdm(train_df.iterrows(), total=len(train_df), desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for idx, row in progress_bar:
            memory_module.reset(DEVICE)
            book_file = BOOK_NAME_MAP.get(row['book_name'])
            with open(os.path.join(DATA_DIR, book_file), "r", encoding="utf-8", errors="ignore") as f:
                novel_text = f.read()
            
            words = novel_text.split()
            for i in range(0, len(words), CHUNK_SIZE):
                memory_module.store(" ".join(words[i:i+CHUNK_SIZE]))
            
            claim_text = f"{row['caption']} {row['content']}"
            claim_inputs = memory_module.tokenizer(claim_text, return_tensors="pt", truncation=True).to(DEVICE)
            with torch.no_grad():
                claim_emb = memory_module.encoder(**claim_inputs).last_hidden_state[:, 0, :]
            
            target = torch.tensor([[LABEL_MAP[row['label']]]], device=DEVICE)
            prediction = reasoner(memory_module.get_memory(), claim_emb)
            
            loss = loss_fn(prediction, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        # Finalize metrics for the epoch
        val_acc, val_f1 = run_validation(train_df, gold_df, memory_module, reasoner)
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        
        log_msg = f"\nSummary - Loss: {epoch_loss/len(train_df):.4f} | Val Acc: {val_acc:.2f} | Val F1: {val_f1:.2f} | LR: {current_lr:.6f}\n"
        print(log_msg)

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                'reasoner': reasoner.state_dict(),
                'memory_rnn': memory_module.memory_rnn.state_dict()
            }, "best_bdh_model.pt")
            print("Checkpoint Saved!\n")

if __name__ == "__main__":
    main()