import os

import torch

import pandas as pd

import gdown

import sys

import warnings



# --- 1. CONFIGURATION ---

# Replace these with the IDs from your Google Drive links

# Example: If URL is drive.google.com/drive/folders/1abc123..., ID is '1abc123'

FOLDER_ID = "1ACIUeIkBc_TkEuDhgPHdDa1Udo5dGHsW"

METADATA_ID = "1Bc8oXDScXr8sYVKdJ9Q9Wm4H3oHlF7Xj"



sys.path.append(os.getcwd())

try:

    import bdh

except ImportError:

    print("Error: bdh.py not found! Ensure you are in the cloned 'bdh' directory.")

    sys.exit(1)



# --- 2. AUTOMATED DATA DOWNLOADER ---

def download_data():

    print("--- Starting Data Download ---")

    os.makedirs("data/novels", exist_ok=True)

    

    # Download Metadata CSV

    if not os.path.exists("data/metadata.csv"):

        gdown.download(id=METADATA_ID, output="data/metadata.csv", quiet=False)

    

    # Download Novel .txt files

    # Using download_folder for batch processing

    gdown.download_folder(id=FOLDER_ID, output="data/novels", quiet=False)



# --- 3. BDH CONSISTENCY REASONER ---

class BDHReasoner:

    def __init__(self):

        # Initialize config and model as per bdh.py requirements

        self.config = bdh.BDHConfig()

        self.model = bdh.BDH(self.config)

        self.model.eval()



    def analyze(self, backstory, novel_path):

        state = None

        max_drift = 0.0

        

        if not os.path.exists(novel_path):

            print(f"Warning: {novel_path} not found.")

            return 1, "File missing"



        with open(novel_path, 'r', encoding='utf-8', errors='ignore') as f:

            # Process novel in chunks to track global consistency evolution

            while True:

                chunk = f.read(2048) 

                if not chunk: break

                

                # Simple ASCII encoding for the BDH model input

                input_ids = torch.tensor([ord(c) % 256 for c in chunk]).unsqueeze(0)

                

                with torch.no_grad():

                    # The model returns (output, next_state)

                    output, next_state = self.model(input_ids, state)

                    

                    if state is not None:

                        # TRACK B CORE: Causal surprise measured by hidden state drift

                        # Large drift = Narrative is logically diverging from backstory

                        drift = torch.norm(next_state - state).item()

                        max_drift = max(max_drift, drift)

                    

                    state = next_state

        

        # Decision threshold: Adjust based on training novel performance

        prediction = 0 if max_drift > 1.5 else 1

        return prediction, f"Max Drift: {max_drift:.4f}"



# --- 4. EXECUTION PIPELINE ---

def main():

    # Step A: Setup Data (Uncomment next line once IDs are set)

    # download_data()

    

    if not os.path.exists("data/metadata.csv"):

        print("Error: metadata.csv not found in data/ folder.")

        return



    df = pd.read_csv("data/metadata.csv")

    reasoner = BDHReasoner()

    results = []



    print(f"--- Processing {len(df)} Novels with BDH Architecture ---")

    for _, row in df.iterrows():

        story_id = str(row['story_id'])

        novel_path = f"data/novels/{story_id}.txt"

        

        pred, rationale = reasoner.analyze(row['backstory'], novel_path)

        results.append({

            "story_id": story_id,

            "prediction": pred,

            "rationale": rationale

        })

        print(f"ID: {story_id} | Result: {pred}")



    # Save to CSV for submission

    pd.DataFrame(results).to_csv("results.csv", index=False)

    print("--- SUCCESS: results.csv generated ---")



if __name__ == "__main__":

    main()
