import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data_loader import load_test_ratings
from src.metrics import precision_at_k, recall_at_k, f1_at_k, ndcg_at_k

def evaluate_model(pickle_path, k=10):
    with open(pickle_path, "rb") as f:
        recommend_data = pickle.load(f)

    test = load_test_ratings()

    # Ground truth: chỉ lấy rating >= 3
    test_gt = (
        test[test["Rating"] >= 3]
        .groupby("UserID")["MovieID"]
        .apply(list)
    )

    all_precision = []
    all_recall = []
    all_f1 = []
    all_ndcg = []

    for user, rec_items in recommend_data.items():
        if user not in test_gt:
            continue

        relevant_items = test_gt[user]

        p = precision_at_k(rec_items, relevant_items, k)
        r = recall_at_k(rec_items, relevant_items, k)
        f1 = f1_at_k(rec_items, relevant_items, k)
        ndcg = ndcg_at_k(rec_items, relevant_items, k)

        all_precision.append(p)
        all_recall.append(r)
        all_f1.append(f1)
        all_ndcg.append(ndcg)

    results = {
        "Precision@K": float(np.mean(all_precision)),
        "Recall@K": float(np.mean(all_recall)),
        "F1@K": float(np.mean(all_f1)),
        "NDCG@K": float(np.mean(all_ndcg)),
        "UsersEvaluated": len(all_precision)
    }

    return results

if __name__ == "__main__":
    k = 10
    model_files = {
        "Content-Based": "./output/result/cb_recommendations.pkl",
        "CF User-Based": "./output/result/cf_user_recommendations.pkl",
        "CF Item-Based": "./output/result/cf_item_recommendations.pkl",
        "Hybrid": "./output/result/hybrid_recommendations.pkl"
    }

    for model_name, path in model_files.items():
        if not Path(path).exists():
            print(f"⚠️ File {path} không tồn tại, bỏ qua {model_name}")
            continue

        results = evaluate_model(path, k)
        print(f"\n=== Evaluation Results: {model_name} ===")
        for key, val in results.items():
            print(f"{key}: {val}")
