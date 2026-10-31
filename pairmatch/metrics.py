"""Fixed binary threshold and explicit zero-division policy."""

from __future__ import annotations


def calculate(labels: list[int], probabilities: list[float], threshold: float = 0.5) -> dict:
    if not labels or len(labels) != len(probabilities):
        raise ValueError("labels/probabilities must be nonempty and equal length")
    if any(type(y) is not int or y not in (0, 1) for y in labels):
        raise ValueError("labels must be binary integers")
    import math
    if not 0 <= threshold <= 1 or any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
        raise ValueError("invalid probabilities or threshold")
    predictions = [int(p >= threshold) for p in probabilities]
    matrix = [[sum(y == i and p == j for y, p in zip(labels, predictions)) for j in (0, 1)] for i in (0, 1)]
    classes = {}
    for label in (0, 1):
        tp = matrix[label][label]
        support = sum(matrix[label])
        predicted = sum(matrix[i][label] for i in (0, 1))
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        classes[str(label)] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    return {"threshold": threshold, "confusion_matrix_true_rows_predicted_columns": matrix,
            "accuracy": sum(matrix[i][i] for i in (0, 1)) / len(labels),
            "macro_f1": sum(classes[str(i)]["f1"] for i in (0, 1)) / 2,
            "balanced_accuracy_fixed_two_classes": sum(classes[str(i)]["recall"] for i in (0, 1)) / 2,
            "classes": classes, "zero_division": 0}
