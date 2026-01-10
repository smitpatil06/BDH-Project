import torch
from sentence_transformers import SentenceTransformer
from bdh.model import BDHMemory
from bdh.reasoner import ConsistencyReasoner

NOVEL_PATH = "data/monte_cristo.txt"
CHUNK_SIZE = 700
EMBED_DIM = 384


def chunk_text(text, chunk_size):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]


def load_text(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def main():
    print("Loading encoder...")
    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    print("Loading novel...")
    novel = load_text(NOVEL_PATH)
    chunks = chunk_text(novel, CHUNK_SIZE)

    print(f"Chunks: {len(chunks)}")

    memory = BDHMemory(EMBED_DIM)

    with torch.no_grad():
        for i, chunk in enumerate(chunks):
            emb = encoder.encode(chunk, convert_to_tensor=True)
            memory(emb)

            if i % 50 == 0:
                print(f"Processed chunk {i}")

    narrative_state = memory.state

    backstory_text = (
        "The character experienced a traumatic childhood "
        "which shaped their distrust of authority."
    )

    backstory_emb = encoder.encode(backstory_text, convert_to_tensor=True)

    reasoner = ConsistencyReasoner(EMBED_DIM)
    score = reasoner(narrative_state, backstory_emb)

    label = int(score.item() > 0.5)

    print("\n===== FINAL OUTPUT =====")
    print("Consistency score:", round(score.item(), 4))
    print("Prediction:", "CONSISTENT" if label == 1 else "CONTRADICT")


if __name__ == "__main__":
    main()
