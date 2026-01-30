from pathlib import Path
from prototype_files.query import search, pretty_print

if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent

    queries = [
        "Where is data preprocessing or cleaning done?",
        "Where is the model training loop?",
        "How are features engineered?"
    ]

    for q in queries:
        print("\n" + "#" * 100)
        print(f"Query: {q}")
        results = search(q, repo_root=repo_root, top_k=6)
        pretty_print(results)
