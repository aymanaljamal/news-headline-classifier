"""
Evaluation metrics utilities
"""

from collections import defaultdict


from collections import defaultdict

def calculate_metrics(y_true, y_pred):
    """
    Calculate Accuracy, Precision, Recall, and F1-Score (macro-averaged)
    """

    classes = sorted(set(y_true + y_pred))


    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for true, pred in zip(y_true, y_pred):
        if true == pred:
            tp[true] += 1
        else:
            fp[pred] += 1
            fn[true] += 1

    # Accuracy
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / len(y_true) if len(y_true) > 0 else 0.0

    precisions = []
    recalls = []
    f1_scores = []

    for cls in classes:
        # Precision = TP / (TP + FP)
        precision = tp[cls] / (tp[cls] + fp[cls]) if (tp[cls] + fp[cls]) > 0 else 0.0
        precisions.append(precision)

        # Recall = TP / (TP + FN)
        recall = tp[cls] / (tp[cls] + fn[cls]) if (tp[cls] + fn[cls]) > 0 else 0.0
        recalls.append(recall)

        # F1 = 2PR/(P+R)
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)

    macro_precision = sum(precisions) / len(precisions) if precisions else 0.0
    macro_recall = sum(recalls) / len(recalls) if recalls else 0.0
    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    return {
        "accuracy": accuracy,
        "precision": macro_precision,
        "recall": macro_recall,
        "f1_score": macro_f1,
        "per_class": {
            cls: {
                "precision": precisions[i],
                "recall": recalls[i],
                "f1_score": f1_scores[i],
            }
            for i, cls in enumerate(classes)
        }
    }



def print_confusion_matrix(y_true, y_pred, class_names=None):
    """Print a simple confusion matrix"""
    classes = sorted(set(y_true + y_pred))

    matrix = [[0] * len(classes) for _ in range(len(classes))]

    for true, pred in zip(y_true, y_pred):
        true_idx = classes.index(true)
        pred_idx = classes.index(pred)
        matrix[true_idx][pred_idx] += 1

    print("\nConfusion Matrix:")
    print("-" * 50)

    if class_names:
        print("     ", end="")
        for cls in classes:
            print(f"{class_names.get(cls, cls):>8}", end="")
        print()

    for i, row in enumerate(matrix):
        if class_names:
            print(f"{class_names.get(classes[i], classes[i]):>4}", end=" ")
        else:
            print(f"{classes[i]:>4}", end=" ")

        for val in row:
            print(f"{val:>8}", end="")
        print()
    print("-" * 50)