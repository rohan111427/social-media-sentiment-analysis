import os
from pathlib import Path

import joblib
import nltk
import pandas as pd
from rake_nltk import Rake

from preprocessing import clean_text

# Download NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    print(f"Error downloading NLTK data: {e}")

# Define paths
current_dir = Path(__file__).parent
project_root = current_dir.parent

model_path = project_root / "models" / "sentiment_model.pkl"
vectorizer_path = project_root / "models" / "tfidf_vectorizer.pkl"

try:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
    if not vectorizer_path.exists():
        raise FileNotFoundError(f"Vectorizer file not found at {vectorizer_path}")

    vectorizer = joblib.load(vectorizer_path)
    model = joblib.load(model_path)
except Exception as e:
    print(f"Error loading model files: {e}")
    exit(1)

label_map = {0: "negative", 1: "neutral", 2: "positive"}

def predict_sentiment(tweet: str) -> str:
    try:
        cleaned = clean_text(tweet)
        vec = vectorizer.transform([cleaned])
        prediction = model.predict(vec)[0]
        return label_map[prediction]
    except Exception as e:
        print(f"Prediction error: {e}")
        return "error"
    
rake = Rake()

def extract_reason(text: str) -> str:
    try:
        if not isinstance(text, str) or pd.isna(text) or text.strip() == "":
            return "N/A"
        rake.extract_keywords_from_text(text)
        ranked_phrases = rake.get_ranked_phrases()
        return ', '.join(ranked_phrases[:2]) if ranked_phrases else "N/A"
    except Exception as e:
        print(f"Reason extraction error: {e}")
        return "N/A"

if __name__ == "__main__":
    try:
        mode = input("Run mode - single or batch? (s/b): ").strip().lower()
        if mode == 's':
            tweet = input("Enter tweet: ").strip()
            if not tweet:
                print("Error: Empty input")
            else:
                print("🔍 Sentiment:", predict_sentiment(tweet))

        elif mode == 'b':
            input_path = input("Enter path to input CSV file: ").strip()
            output_path = input("Enter path to save output CSV: ").strip()

            df = pd.read_csv(input_path)
            if 'text' not in df.columns:
                print("Error: CSV must contain a 'text' column.")
                exit(1)

            import joblib
            label_encoder_path = project_root / "models" / "label_encoder.pkl"
            label_encoder = joblib.load(label_encoder_path)

            df['predicted_sentiment'] = df['text'].apply(lambda x: model.predict(vectorizer.transform([clean_text(x)]))[0])
            df['sentiment_label'] = df['predicted_sentiment'].apply(lambda x: label_encoder.inverse_transform([x])[0])
            df['reason'] = df['text'].apply(extract_reason)

            df = df[['text', 'predicted_sentiment', 'sentiment_label', 'reason']]
            df.to_csv(output_path, index=False)
            print(f"✅ Clean labeled output saved to {output_path}")
        else:
            print("Invalid mode. Use 's' for single or 'b' for batch.")

    except KeyboardInterrupt:
        print("\nOperation cancelled")
    except Exception as e:
        print(f"Unexpected error: {e}")