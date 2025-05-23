import os

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
input_file = os.path.join(base_dir, "data", "processed", "cleaned_tweets.csv")
output_folder = os.path.join(base_dir, "data", "processed")

import os

print("Exists:", os.path.exists(input_file))
print("Path:", input_file)

df = pd.read_csv(input_file)

print("Label distribution before Encoding")
print(df['airline_sentiment'].value_counts())

label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['airline_sentiment'])

print("Label Mapping:" , dict(zip(label_encoder.classes_,label_encoder.transform(label_encoder.classes_))))

joblib.dump(label_encoder,os.path.join(base_dir,"models","label_encoder.pkl"))

X = df['cleaned_text']
y = df['label']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=42)

X_train.to_csv(os.path.join(output_folder, "X_train.csv"), index = False)
X_test.to_csv(os.path.join(output_folder, "X_test.csv"), index = False)
y_train.to_csv(os.path.join(output_folder, "y_train.csv"), index = False)
y_test.to_csv(os.path.join(output_folder, "y_test.csv"), index = False)

print("Data is split and saved")
