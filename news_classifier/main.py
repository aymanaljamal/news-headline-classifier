"""
Main entry point for News Headline Classifier
Requirements covered:
- Preprocess & clean text
- TF-IDF vectorization
- Train & evaluate Decision Tree + Logistic Regression
- Compare metrics (Accuracy, Precision, Recall, F1)
- Generate graphs
- Interpret results (strengths/weaknesses)
"""

from utils.config_loader import load_config
from news_classifier.classifier import TextClassifier


from utils.plot_utils import (
    plot_model_comparison,
    plot_confusion_matrix,
    plot_decision_tree_manual,
)


def interpret_results(results: dict) -> str:
    """
    Simple interpretation of strengths/weaknesses based on metrics.
    """
    dt = results["decision_tree"]["metrics"]
    lr = results["logistic_regression"]["metrics"]

    def winner(metric):
        return "Logistic Regression" if lr[metric] > dt[metric] else "Decision Tree"

    lines = []
    lines.append("INTERPRETATION")
    lines.append("-" * 60)
    lines.append(f"- Best Accuracy:  {winner('accuracy')}")
    lines.append(f"- Best Precision: {winner('precision')}")
    lines.append(f"- Best Recall:    {winner('recall')}")
    lines.append(f"- Best F1-Score:  {winner('f1_score')}")
    return "\n".join(lines)


def main():

    print("Loading configuration...")
    config, stopwords = load_config()

    print(f"\n{config['app']['name']} v{config['app']['version']}")
    print("=" * 60)


    classifier = TextClassifier(config)


    data_file = config["data"]["train_file"]
    print("\n[1/6] Preparing data (cleaning + TF-IDF)...")
    classifier.prepare_data(data_file)


    print("\n[2/6] Training models (Decision Tree + Logistic Regression)...")
    classifier.train_models()


    print("\n[3/6] Evaluating models...")
    results = classifier.evaluate_models()


    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    dt_metrics = results["decision_tree"]["metrics"]
    lr_metrics = results["logistic_regression"]["metrics"]

    print("\nDecision Tree:")
    print(f"  Accuracy:  {dt_metrics['accuracy']:.4f}")
    print(f"  Precision: {dt_metrics['precision']:.4f}")
    print(f"  Recall:    {dt_metrics['recall']:.4f}")
    print(f"  F1-Score:  {dt_metrics['f1_score']:.4f}")

    print("\nLogistic Regression:")
    print(f"  Accuracy:  {lr_metrics['accuracy']:.4f}")
    print(f"  Precision: {lr_metrics['precision']:.4f}")
    print(f"  Recall:    {lr_metrics['recall']:.4f}")
    print(f"  F1-Score:  {lr_metrics['f1_score']:.4f}")


    print("\n[4/6] Generating graphs...")


    plot_model_comparison(
        {
            "decision_tree": dt_metrics,
            "logistic_regression": lr_metrics,
        },
        save_path="data/model_comparison.png"
    )


    plot_confusion_matrix(
        results["decision_tree"]["y_true"],
        results["decision_tree"]["y_pred"],
        title="Decision Tree - Confusion Matrix",
        save_path="data/confusion_dt.png"
    )

    plot_confusion_matrix(
        results["logistic_regression"]["y_true"],
        results["logistic_regression"]["y_pred"],
        title="Logistic Regression - Confusion Matrix",
        save_path="data/confusion_lr.png"
    )

    print("\n[4.5/6] Plotting Decision Tree structure...")
    dt_model = classifier.decision_tree
    feature_names = getattr(classifier, "feature_names", None)
    class_names = config["data"].get("class_names", None)

    plot_decision_tree_manual(
        dt_model.root,
        feature_names=feature_names,
        class_names=class_names,
        save_path="data/decision_tree_structure.png"
    )


    print("\n[5/6] Interpreting results...")
    print("\n" + interpret_results(results))


    print("\n" + "=" * 60)
    print("SAMPLE PREDICTIONS")
    print("=" * 60)

    test_headlines = [
        "Stock market reaches new high",
        "World Cup final draws huge crowd",
        "New AI breakthrough announced",
        "International summit discusses climate"
    ]

    for headline in test_headlines:
        dt_pred = classifier.predict(headline, model_type="decision_tree")
        lr_pred = classifier.predict(headline, model_type="logistic_regression")
        print(f"\nHeadline: {headline}")
        print(f"  Decision Tree: {dt_pred}")
        print(f"  Logistic Regression: {lr_pred}")

    print("\n[6/6] Done.")
    print("=" * 60)
    print("Classification complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
