import os

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 6],
    'learning_rate': [0.1, 0.3],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

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

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

xgb = XGBClassifier( eval_metric='mlogloss', random_state=42)
grid_search = GridSearchCV(estimator= xgb, param_grid= param_grid, cv = 3, scoring= 'accuracy', n_jobs= -1 , verbose=1)
grid_search.fit(X_train_vec, y_train)

# Best model
best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Val Accuracy:", grid_search.best_score_)

# Evaluate on test set
y_pred = best_model.predict(X_test_vec)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save best model and vectorizer
joblib.dump(best_model, os.path.join(model_dir, "sentiment_model.pkl"))
joblib.dump(vectorizer, os.path.join(model_dir, "tfidf_vectorizer.pkl"))
print("Best model and vectorizer saved.")