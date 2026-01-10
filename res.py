import pandas as pd

# 1. Ground Truth for the 60 samples in your test.csv
gold_data = {
    95: 'contradict', 136: 'contradict', 59: 'consistent', 60: 'consistent', 
    124: 'contradict', 111: 'consistent', 135: 'consistent', 27: 'consistent', 
    110: 'consistent', 42: 'consistent', 56: 'consistent', 80: 'consistent', 
    28: 'consistent', 101: 'consistent', 91: 'consistent', 24: 'contradict', 
    119: 'consistent', 93: 'contradict', 72: 'consistent', 16: 'contradict', 
    30: 'consistent', 97: 'consistent', 22: 'consistent', 53: 'consistent', 
    58: 'consistent', 81: 'consistent', 47: 'consistent', 7: 'consistent', 
    121: 'contradict', 129: 'consistent', 51: 'consistent', 127: 'consistent', 
    133: 'consistent', 52: 'consistent', 114: 'consistent', 2: 'consistent', 
    33: 'contradict', 15: 'consistent', 82: 'consistent', 37: 'consistent', 
    3: 'consistent', 103: 'contradict', 94: 'consistent', 85: 'consistent', 
    21: 'contradict', 49: 'consistent', 140: 'consistent', 64: 'consistent', 
    131: 'consistent', 61: 'consistent', 86: 'consistent', 38: 'consistent', 
    132: 'consistent', 77: 'consistent', 70: 'consistent', 115: 'consistent', 
    78: 'consistent', 87: 'consistent', 75: 'consistent', 44: 'consistent'
}

# 2. Load your predictions
try:
    pred_df = pd.read_csv("results.csv")
    
    # Map the ground truth to the IDs in your results
    pred_df['gold'] = pred_df['id'].map(gold_data)
    
    # Calculate accuracy
    correct = (pred_df['label'] == pred_df['gold']).sum()
    total = len(pred_df)
    accuracy = (correct / total) * 100

    print(f"--- Final Evaluation ---")
    print(f"Total Samples: {total}")
    print(f"Correct Predictions: {correct}")
    print(f"Exact Accuracy: {accuracy:.2f}%")
    
    # Optional: See which ones you missed
    if accuracy < 100:
        print("\nSamples Missed:")
        print(pred_df[pred_df['label'] != pred_df['gold']][['id', 'label', 'gold']])

except Exception as e:
    print(f"Error: {e}")
