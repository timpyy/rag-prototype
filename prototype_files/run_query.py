from pathlib import Path
from prototype_files.query import search, pretty_print

if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent

    queries = [
        "What is the overall data science pipeline implemented in this project?",
        "Which steps happen before model training, and where are they implemented?",
        "Where does model evaluation happen, and what metrics are used?",
        "Where is raw data cleaned or transformed before modeling?",
        "How are features engineered from the raw data?",
        "What assumptions or heuristics are documented about the preprocessing steps?",
        "Which model(s) are trained in this project?",
        "Where is the training loop defined, and what inputs does it expect?",
        "What hyperparameters are configured for the model, and where are they set?",
        "Which notebook contains the main end-to-end analysis?",
        "What does the first half of the main notebook do, step by step?",
        "Where in the notebook is feature scaling or normalization performed?",
        "What files or scripts would someone run to reproduce the results?",
        "What external dependencies or libraries are required for training?",
        "If you wanted to modify the feature engineering step, where would you start?"
    ]

    for q in queries:
        print("\n" + "#" * 100)
        print(f"Query: {q}")
        results = search(q, repo_root=repo_root, top_k=6)
        pretty_print(results)
