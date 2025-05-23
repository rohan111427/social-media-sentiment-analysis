import os

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, "data", "processed")
model_dir = os.path.join(base_dir, "models")
os.makedirs(model_dir, exist_ok=True)

# Load datasets
X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))['cleaned_text']
X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))['cleaned_text']
y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).squeeze()
y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).squeeze()

# Combine X and y into a DataFrame, drop NaNs, and then separate again
train_df = pd.DataFrame({'text': X_train, 'label': y_train})
train_df = train_df.dropna()

X_train = train_df['text'].astype(str)
y_train = train_df['label']

test_df = pd.DataFrame({'text': X_test, 'label': y_test})
test_df = test_df.dropna()

X_test = test_df['text'].astype(str)
y_test = test_df['label']

vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(class_weight='balanced', max_iter=200)
model.fit(X_train_vec,y_train)

y_pred = model.predict(X_test_vec)

print("Classification Report:")
print(classification_report(y_test,y_pred))
print("Acuraccy:", accuracy_score(y_test,y_pred))

joblib.dump(model, os.path.join(model_dir,"sentiment_model.pkl"))
joblib.dump(vectorizer,os.path.join(model_dir,"tfidf_vectorizer.pkl"))
print("Model and Vectorizer saved.")