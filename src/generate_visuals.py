import os

import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud

# Setup paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(base_dir, "data", "processed", "output_clean.csv")
visuals_path = os.path.join(base_dir, "visuals")
os.makedirs(visuals_path, exist_ok=True)

# Load data
try:
    df = pd.read_csv(data_path)
except FileNotFoundError:
    print(f"❌ Could not find file at {data_path}")
    exit(1)

# ================================
# 📊 Bar Chart: Sentiment Counts
# ================================
try:
    sentiment_counts = df['Predicted Sentiment'].value_counts()

    plt.figure(figsize=(6, 4))
    sentiment_counts.plot(kind='bar', color=['green', 'gray', 'red'])
    plt.title("Sentiment Distribution")
    plt.ylabel("Number of Tweets")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(visuals_path, "sentiment_distribution.png"))
    plt.close()
    print("✅ Saved sentiment_distribution.png")
except Exception as e:
    print(f"❌ Error creating sentiment bar chart: {e}")

# ======================================
# ☁️ Word Cloud: Reasons by Sentiment
# ======================================
try:
    for sentiment in df['Predicted Sentiment'].dropna().unique():
        sentiment_str = str(sentiment).lower()

        subset = df[df['Predicted Sentiment'] == sentiment]
        text = ' '.join(subset['Reason'].dropna().astype(str))

        if not text.strip():
            print(f"⚠️ Skipped word cloud for {sentiment} (no valid text)")
            continue

        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            collocations=False
        ).generate(text)

        plt.figure(figsize=(8, 4))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f"Word Cloud for {sentiment} Sentiment")

        filename = f"wordcloud_{sentiment_str}.png"
        plt.savefig(os.path.join(visuals_path, filename))
        plt.close()
        print(f"✅ Saved {filename}")

except Exception as e:
    print(f"❌ Error creating word clouds: {e}")