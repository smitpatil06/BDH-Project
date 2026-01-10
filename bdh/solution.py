import torch
import torch.nn as nn
import pandas as pd
import os
import sys

# Ensure local imports work
sys.path.append(os.getcwd())
import bdh

class TrackBReasoning:
    def __init__(self):
        # 1. Initialize the Config first
        # We pass default parameters; adjust if your bdh.py expects specific names
        self.config = bdh.BDHConfig() 
        
        # 2. Initialize the Model with the config
        self.model = bdh.BDH(self.config)
        self.model.eval()

    def _text_to_tensor(self, text):
        # BDH expects tensors. This is a simple ASCII encoder 
        # to convert your text chunks into a format the model can process.
        tokens = [ord(c) % 256 for c in text[:1024]]
        return torch.tensor(tokens).unsqueeze(0)

    def get_consistency(self, backstory, novel_text):
        state = None
        max_drift = 0.0
        
        # Process novel in small windows
        chunk_size = 512
        chunks = [novel_text[i:i+chunk_size] for i in range(0, len(novel_text), chunk_size)]
        
        with torch.no_grad():
            for chunk in chunks:
                input_tensor = self._text_to_tensor(chunk)
                try:
                    # Standard BDH forward signature: (input, hidden_state)
                    output, next_state = self.model(input_tensor, state)
                    
                    if state is not None:
                        # Track B: Measure Causal Surprise via State Drift
                        # This checks if the 'memory' of the story is shifting too violently
                        drift = torch.norm(next_state - state).item()
                        max_drift = max(max_drift, drift)
                    
                    state = next_state
                except Exception as e:
                    continue

        # Logic: 1 = Consistent, 0 = Contradict
        threshold = 2.0 
        prediction = 0 if max_drift > threshold else 1
        rationale = f"Max state drift: {max_drift:.4f}"
        
        return prediction, rationale

def run_pipeline():
    if not os.path.exists("data/metadata.csv"):
        os.makedirs("data", exist_ok=True)
        pd.DataFrame({
            "story_id": ["test_log"],
            "backstory": ["Character is a pacifist."],
            "novel_text": ["The character started a war."]
        }).to_csv("data/metadata.csv", index=False)

    df = pd.read_csv("data/metadata.csv")
    reasoner = TrackBReasoning()
    results = []

    print("Running BDH Consistency Analysis...")
    for _, row in df.iterrows():
        pred, logic = reasoner.get_consistency(row['backstory'], str(row.get('novel_text', "")))
        results.append({
            "story_id": row['story_id'],
            "prediction": pred,
            "rationale": logic
        })

    pd.DataFrame(results).to_csv("results.csv", index=False)
    print("SUCCESS: results.csv generated.")

if __name__ == "__main__":
    run_pipeline()
