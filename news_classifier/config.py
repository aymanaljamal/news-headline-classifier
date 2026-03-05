"""
Configuration file for News Headline Classifier
All settings are defined here as Python dictionaries.
"""

CONFIG = {
    "app": {
        "name": "News Headline Classifier",
        "version": "1.0.0"
    },
    "data": {
        "train_file": "data/ag_news_full.csv",
        "test_split": 0.2,
        "random_seed": 42,
        "max_train_samples": 50000,
        "class_names": ["World", "Sports", "Business", "Sci/Tech"]
    },
    "preprocessing": {
        "min_word_length": 2,
        "max_features": 1000,
        "lowercase": True,
        "remove_punctuation": True,
        "remove_stopwords": True
    },
    "models": {
        "decision_tree": {
            "max_depth": 10,
            "min_samples_split": 5,
            "sample_size":50000
        },
        "logistic_regression": {
            "max_iterations": 100,
            "learning_rate": 0.01,
            "sample_size":50000
        }
    },
    "categories": {
        0: "World",
        1: "Sports",
        2: "Business",
        3: "Sci/Tech"
    }
}

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "can", "this", "that",
    "these", "those", "i", "you", "he", "she", "it", "we", "they"
}
