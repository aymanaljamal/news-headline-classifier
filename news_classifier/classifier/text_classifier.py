"""
Main Text Classifier class that uses Decision Tree and Logistic Regression
"""
import time
import os
import csv
import random
import math

from news_classifier.utils.text_processor import TextProcessor
from news_classifier.utils.metrics import calculate_metrics
from news_classifier.models import DecisionTree, LogisticRegressionOVR


class TextClassifier:
    """Main classifier class for news headlines"""

    def __init__(self, config):
        self.config = config
        self.text_processor = TextProcessor(config["preprocessing"])

        self.decision_tree = None
        self.logistic_regression = None

        self.vocabulary = None
        self.idf = None

        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.feature_names = None

    def load_data(self, filepath): # Data Loading (ROBUST CSV)

        if not filepath or not isinstance(filepath, str):
            raise ValueError("Invalid filepath.")

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"CSV file not found: {filepath}")

        data = []
        delimiters = [",", ";", "\t"]# Try common delimiters automatically

        for delim in delimiters:
            data.clear()
            with open(filepath, "r", encoding="utf-8", newline="") as f:
                reader = csv.reader(f, delimiter=delim)
                header = next(reader, None)

                for row in reader:
                    if not row or len(row) < 2:
                        continue

                    row = [c.strip() for c in row if c is not None]


                    first = row[0]
                    rest_text = delim.join(row[1:]).strip().strip('"')
                    if first.isdigit() and rest_text:
                        data.append((rest_text, int(first)))
                        continue

                    last = row[-1]
                    text_first = delim.join(row[:-1]).strip().strip('"')
                    if last.isdigit() and text_first:
                        data.append((text_first, int(last)))
                        continue

            if len(data) > 0:
                return data

        return []

    # Split Data (SAFE)
    def split_data(self, data, test_split=0.2, random_seed=42):
        """Split data into train and test sets (guarantee non-empty splits)."""
        if data is None or len(data) == 0:
            raise ValueError("Dataset is empty. Check the CSV path/format.")

        if not (0 < test_split < 1):
            raise ValueError(f"test_split must be between 0 and 1 (got {test_split}).")


        random.seed(random_seed)
        random.shuffle(data)

        n = len(data)
        test_size = int(n * test_split)


        test_size = max(1, min(test_size, n - 1))
        split_idx = n - test_size

        train_data = data[:split_idx]
        test_data = data[split_idx:]


        max_train_samples = self.config["data"].get("max_train_samples", None)
        if max_train_samples is not None and len(train_data) > max_train_samples:
            random.seed(random_seed)
            train_data = random.sample(train_data, max_train_samples)
            print(f"Train samples after limit: {len(train_data)}, Test samples: {len(test_data)}")
        else:
            print(f"Train samples: {len(train_data)}, Test samples: {len(test_data)}")

        return train_data, test_data



    # Vocabulary
    def build_vocabulary(self, texts, max_features=1000):
        """Build vocabulary from texts"""
        word_counts = {}

        for text in texts:
            words = self.text_processor.tokenize(text)
            for word in words:
                if not word:
                    continue
                word_counts[word] = word_counts.get(word, 0) + 1

        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        vocabulary = {word: idx for idx, (word, _) in enumerate(sorted_words[:max_features])}
        return vocabulary




    def _compute_tf(self, texts): # TF-IDF (Train fit / Transform)
        """Compute normalized term-frequency vectors for texts using existing vocabulary."""
        tf_vectors = []
        vocab_size = len(self.vocabulary)

        for text in texts:
            words = self.text_processor.tokenize(text)
            tf = [0] * vocab_size

            for word in words:
                idx = self.vocabulary.get(word)
                if idx is not None:
                    tf[idx] += 1

            total = sum(tf)
            if total > 0:
                tf = [count / total for count in tf]

            tf_vectors.append(tf)

        return tf_vectors

    def fit_tfidf(self, train_texts):
        """Fit IDF on training set and return TF-IDF for training."""
        if self.vocabulary is None or len(self.vocabulary) == 0:
            raise ValueError("Vocabulary is empty. Cannot compute TF-IDF.")

        tf_train = self._compute_tf(train_texts)

        n_docs = len(train_texts)
        vocab_size = len(self.vocabulary)
        idf = [0.0] * vocab_size

        # Smooth IDF to avoid log(0): idf = log((N + 1) / (df + 1)) + 1
        for word, idx in self.vocabulary.items():
            df = sum(1 for tf in tf_train if tf[idx] > 0)
            idf[idx] = math.log((n_docs + 1) / (df + 1)) + 1.0

        self.idf = idf

        X_train = []
        for tf in tf_train:
            X_train.append([tf[i] * idf[i] for i in range(vocab_size)])

        return X_train

    def transform_tfidf(self, texts):
        """Transform texts to TF-IDF using already-fitted IDF."""
        if self.vocabulary is None or self.idf is None:
            raise ValueError("TF-IDF not fitted. Call fit_tfidf() first.")

        tf_vectors = self._compute_tf(texts)
        vocab_size = len(self.vocabulary)

        X = []
        for tf in tf_vectors:
            X.append([tf[i] * self.idf[i] for i in range(vocab_size)])
        return X


    # Prepare Data
    def prepare_data(self, filepath):
        """Load and prepare data for training"""
        print("Loading data...")

        print("DATA FILE =", filepath)
        print("EXISTS?   =", os.path.exists(filepath))

        data = self.load_data(filepath)
        print(f"Loaded {len(data)} samples")

        if len(data) == 0:
            raise ValueError(
                "Loaded 0 samples. Your CSV parsing didn't match the file format.\n"
                "Tip: Print first lines of the CSV and verify delimiter and which column is the label."
            )

        train_data, test_data = self.split_data(
            data,
            test_split=self.config["data"]["test_split"],
            random_seed=self.config["data"]["random_seed"],
        )

        print(f"Train samples: {len(train_data)}, Test samples: {len(test_data)}")

        if len(train_data) == 0 or len(test_data) == 0:
            raise ValueError(
                f"Split resulted in empty set: train={len(train_data)}, test={len(test_data)}. "
                f"Adjust test_split or check dataset size."
            )

        # Build vocabulary from training data
        train_texts = [text for text, _ in train_data]
        print("Building vocabulary...")
        self.vocabulary = self.build_vocabulary(
            train_texts,
            max_features=self.config["preprocessing"]["max_features"],
        )
        print(f"Vocabulary size: {len(self.vocabulary)}")

        if len(self.vocabulary) == 0:
            raise ValueError("Vocabulary ended up empty. Check tokenizer/stopwords settings.")

        # TF-IDF (fit on train, transform test)
        print("Converting to TF-IDF vectors...")
        self.X_train = self.fit_tfidf(train_texts)
        self.y_train = [label for _, label in train_data]

        test_texts = [text for text, _ in test_data]
        self.X_test = self.transform_tfidf(test_texts)
        self.y_test = [label for _, label in test_data]

        print("Data preparation complete!")
        print("DEBUG -> X_train:", len(self.X_train), "y_train:", len(self.y_train))
        print("DEBUG -> X_test :", len(self.X_test), "y_test :", len(self.y_test))


    # Train
    def train_models(self):
        """Train both Decision Tree and Logistic Regression"""
        if self.X_train is None or self.y_train is None or len(self.y_train) == 0:
            raise ValueError("Training data is not prepared. Call prepare_data() first.")

        print("\n" + "=" * 50)
        print("Training Decision Tree (SAMPLED)...")
        print("=" * 50)

        dt_config = self.config["models"]["decision_tree"]

        # sample
        sample_size = dt_config.get("sample_size", 5000)
        n = len(self.X_train)
        k = min(sample_size, n)

        random.seed(self.config["data"]["random_seed"])
        idx = random.sample(range(n), k)

        X_dt = [self.X_train[i] for i in idx]
        y_dt = [self.y_train[i] for i in idx]

        self.decision_tree = DecisionTree(
            max_depth=dt_config["max_depth"],
            min_samples_split=dt_config["min_samples_split"],
        )

        t0 = time.time()
        print(
            f"DT sample: {k}/{n} (max_depth={dt_config['max_depth']}, min_samples_split={dt_config['min_samples_split']})")
        self.decision_tree.fit(X_dt, y_dt)
        print("Decision Tree training complete!")
        print("DT time:", round(time.time() - t0, 2), "seconds")

        # Logistic Regression
        print("\n" + "=" * 50)
        print("Training Logistic Regression (SAMPLED)...")
        print("=" * 50)

        lr_config = self.config["models"]["logistic_regression"]

        lr_sample_size = lr_config.get("sample_size", 5000)
        n_lr = len(self.X_train)
        k_lr = min(lr_sample_size, n_lr)

        random.seed(self.config["data"]["random_seed"])
        idx_lr = random.sample(range(n_lr), k_lr)

        X_lr = [self.X_train[i] for i in idx_lr]
        y_lr = [self.y_train[i] for i in idx_lr]

        self.logistic_regression = LogisticRegressionOVR(
            max_iterations=lr_config["max_iterations"],
            learning_rate=lr_config["learning_rate"],
        )

        t1 = time.time()
        print(f"LR sample: {k_lr}/{n_lr} (max_iter={lr_config['max_iterations']}, lr={lr_config['learning_rate']})")
        self.logistic_regression.fit(X_lr, y_lr)
        print("Logistic Regression training complete!")
        print("LR time:", round(time.time() - t1, 2), "seconds")


    # Evaluate
    def evaluate_models(self):
        """Evaluate both models and return metrics + predictions (format expected by main.py)"""
        if self.X_test is None or self.y_test is None or len(self.y_test) == 0:
            raise ValueError("Test data is not prepared. Call prepare_data() first.")

        print("\n" + "=" * 50)
        print("Evaluating Decision Tree...")
        print("=" * 50)

        dt_pred = self.decision_tree.predict(self.X_test)
        dt_metrics = calculate_metrics(self.y_test, dt_pred)

        print(f"Accuracy:  {dt_metrics['accuracy']:.4f}")
        print(f"Precision: {dt_metrics['precision']:.4f}")
        print(f"Recall:    {dt_metrics['recall']:.4f}")
        print(f"F1-Score:  {dt_metrics['f1_score']:.4f}")

        print("\n" + "=" * 50)
        print("Evaluating Logistic Regression...")
        print("=" * 50)

        lr_pred = self.logistic_regression.predict(self.X_test)
        lr_metrics = calculate_metrics(self.y_test, lr_pred)

        print(f"Accuracy:  {lr_metrics['accuracy']:.4f}")
        print(f"Precision: {lr_metrics['precision']:.4f}")
        print(f"Recall:    {lr_metrics['recall']:.4f}")
        print(f"F1-Score:  {lr_metrics['f1_score']:.4f}")


        return {
            "decision_tree": {
                "metrics": dt_metrics,
                "y_true": self.y_test,
                "y_pred": dt_pred
            },
            "logistic_regression": {
                "metrics": lr_metrics,
                "y_true": self.y_test,
                "y_pred": lr_pred
            }
        }

    # Predict One
    def predict(self, text, model_type="decision_tree"):
        """Predict category for a single headline"""
        if self.vocabulary is None or self.idf is None:
            raise ValueError("Model not ready. Train the model first.")

        # TF for single text
        words = self.text_processor.tokenize(text)
        tf = [0] * len(self.vocabulary)

        for word in words:
            idx = self.vocabulary.get(word)
            if idx is not None:
                tf[idx] += 1

        total = sum(tf)
        if total > 0:
            tf = [count / total for count in tf]

        tfidf = [tf[i] * self.idf[i] for i in range(len(tf))]# TF-IDF for single text

        # Predict
        if model_type == "decision_tree":
            prediction = self.decision_tree.predict([tfidf])[0]
        else:
            prediction = self.logistic_regression.predict([tfidf])[0]

        return self.config["categories"][prediction]
