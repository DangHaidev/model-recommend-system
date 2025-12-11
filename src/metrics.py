import numpy as np

def rmse(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def precision(recommended_items, relevant_items):
    if len(recommended_items) == 0:
        return 0.0

    hits = len(set(recommended_items) & set(relevant_items))
    return hits / len(recommended_items)


def recall(recommended_items, relevant_items):
    if len(relevant_items) == 0:
        return 0.0

    hits = len(set(recommended_items) & set(relevant_items))
    return hits / len(relevant_items)


def f1_score(recommended_items, relevant_items):
    p = precision(recommended_items, relevant_items)
    r = recall(recommended_items, relevant_items)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def precision_at_k(recommended_items, relevant_items, k):
    return precision(recommended_items[:k], relevant_items)


def recall_at_k(recommended_items, relevant_items, k):
    return recall(recommended_items[:k], relevant_items)


def f1_at_k(recommended_items, relevant_items, k):
    return f1_score(recommended_items[:k], relevant_items)


def ndcg_at_k(recommended_items, relevant_items, k):
    dcg = 0.0
    for i, item in enumerate(recommended_items[:k]):
        if item in relevant_items:
            dcg += 1 / np.log2(i + 2)

    ideal_hits = min(k, len(relevant_items))
    idcg = sum(1 / np.log2(i + 2) for i in range(ideal_hits))

    return dcg / idcg if idcg > 0 else 0.0

def precision_at_10(recommended_items, relevant_items):
    """Precision@10"""
    return precision_at_k(recommended_items, relevant_items, 10)


def recall_at_10(recommended_items, relevant_items):
    """Recall@10"""
    return recall_at_k(recommended_items, relevant_items, 10)


def f1_at_10(recommended_items, relevant_items):
    """F1@10"""
    return f1_at_k(recommended_items, relevant_items, 10)


def ndcg_at_10(recommended_items, relevant_items):
    """NDCG@10"""
    return ndcg_at_k(recommended_items, relevant_items, 10)
