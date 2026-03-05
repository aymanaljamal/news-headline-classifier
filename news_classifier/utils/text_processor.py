"""
Text preprocessing utilities
"""
import re
from news_classifier.config import STOPWORDS


class TextProcessor:
    """Handle text preprocessing operations"""

    def __init__(self, preprocessing_config):
        self.config = preprocessing_config
        self.stopwords = STOPWORDS if preprocessing_config.get('remove_stopwords') else set()

    def clean_text(self, text):
        """Clean and normalize text"""

        if self.config.get('lowercase', True):
            text = text.lower()


        if self.config.get('remove_punctuation', True):
            text = re.sub(r'[^\w\s]', ' ', text)


        text = ' '.join(text.split())

        return text

    def tokenize(self, text):
        """Tokenize text into words"""
        text = self.clean_text(text)
        words = text.split()


        min_length = self.config.get('min_word_length', 2)
        words = [w for w in words if len(w) >= min_length]


        if self.stopwords:
            words = [w for w in words if w not in self.stopwords]

        return words

    def preprocess_batch(self, texts):
        """Preprocess multiple texts"""
        return [self.tokenize(text) for text in texts]