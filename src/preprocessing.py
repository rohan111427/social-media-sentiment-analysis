import re
import string

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK resources (run once, already included in your code)
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab')

# Initialize stopwords globally
stop_words = set(stopwords.words('english'))

def clean_text(text):
    # Handle non-string input (e.g., NaN, None, numbers)
    if not isinstance(text, str) or pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+|\S+\.\S+', '', text)
    
    # Remove mentions and hashtags
    text = re.sub(r'@\w+|#\w+', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize text
    tokens = word_tokenize(text)
    
    # Remove punctuation and stopwords
    tokens = [word for word in tokens if word not in string.punctuation and word not in stop_words]
    
    # Join tokens with spaces
    cleaned = ' '.join(tokens)
    
    return cleaned