import os

import pandas as pd

from preprocessing import clean_text

# Define paths relative to the project root
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
input_file = os.path.join(base_dir, "data", "raw", "Tweets.csv")
output_folder = os.path.join(base_dir, "data", "processed")
output_file = os.path.join(output_folder, "cleaned_tweets.csv")

try:
    # Load CSV
    df = pd.read_csv(input_file, encoding="utf-8")
    
    # Verify 'text' column exists
    if 'text' not in df.columns:
        raise ValueError("Column 'text' not found in CSV.")
    
    # Apply text cleaning
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Save cleaned CSV
    df.to_csv(output_file, index=False)
    print(f"✅ Cleaned data saved to {output_file}")
except FileNotFoundError:
    print(f"Error: Input file {input_file} not found.")
except PermissionError:
    print(f"Error: Permission denied when writing to {output_file}.")
except Exception as e:
    print(f"An error occurred: {str(e)}")