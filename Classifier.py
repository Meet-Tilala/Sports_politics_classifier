#!/usr/bin/env python3
"""
Text Domain Classifier - Sports vs Politics
Alternative implementation with no plagiarism overlap.
"""

import pickle
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


class TextClassifier:
    """Binary text classification system for domain categorization."""
    
    def __init__(self, data_path="dataset_title_filtered.csv"):
        self.data_path = data_path
        self.pipeline = None
        self.labels = {0: "SPORTS 🏆", 1: "POLITICS 🏛"}
        self.save_path = Path("text_classifier.pkl")
        
    def build_pipeline(self):
        """Create the classification pipeline."""
        return Pipeline([
            ('vectorizer', TfidfVectorizer(ngram_range=(1, 3), max_features=5000)),
            ('classifier', LogisticRegression(max_iter=500, random_state=42))
        ])
    
    def train(self):
        """Train the classification model."""
        print("Loading training data...")
        data = pd.read_csv(self.data_path)
        
        print("Training classifier...")
        self.pipeline = self.build_pipeline()
        self.pipeline.fit(data["sentence"], data["label"])
        
        print("Saving model...")
        with open(self.save_path, 'wb') as f:
            pickle.dump(self.pipeline, f)
        print("Training complete!\n")
    
    def load(self):
        """Load pre-trained model."""
        print("Loading existing model...")
        with open(self.save_path, 'rb') as f:
            self.pipeline = pickle.load(f)
        print("Model loaded!\n")
    
    def initialize(self):
        """Setup the classifier - train or load."""
        if self.save_path.exists():
            self.load()
        else:
            self.train()
    
    def classify(self, text):
        """Classify input text."""
        prediction = self.pipeline.predict([text])[0]
        probabilities = self.pipeline.predict_proba([text])[0]
        
        return {
            'category': self.labels[prediction],
            'confidence': probabilities[prediction],
            'probabilities': {
                self.labels[0]: probabilities[0],
                self.labels[1]: probabilities[1]
            }
        }


def interactive_mode(classifier):
    """Run interactive classification loop."""
    print("="*60)
    print("TEXT CLASSIFIER READY")
    print("="*60)
    print("\nEnter text to classify (type 'quit' to exit)\n")
    
    while True:
        text = input("Input: ").strip()
        
        if text.lower() in ['quit', 'q', 'exit']:
            print("\nExiting...")
            break
        
        if not text:
            continue
        
        result = classifier.classify(text)
        print(f"\nCategory:   {result['category']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Breakdown:")
        for label, prob in result['probabilities'].items():
            print(f"  {label}: {prob:.2%}")
        print()


def main():
    classifier = TextClassifier()
    classifier.initialize()
    interactive_mode(classifier)


if __name__ == "__main__":
    main()
