# Social Media Sentiment Analysis for Brand Health

This project applies machine learning to analyze social media posts and classify sentiment (Positive, Neutral, Negative) toward a brand or product. It helps businesses monitor brand reputation, understand customer feedback, and make data-driven decisions.

---

## Project Objective

- Analyze public sentiment using real-world tweet data
- Identify why users feel positively or negatively (reason extraction)
- Generate data visualizations to communicate insights clearly
- Enable batch predictions and dashboard-ready outputs

---

## Key Features

- Load and preprocess social media data from CSV
- Clean text using NLP techniques (NLTK)
- Classify sentiment using XGBoost model
- Extract key phrases from text using RAKE
- Batch predict sentiment and extract reasoning
- Save cleaned, labeled output CSVs
- Generate:
  - Sentiment distribution bar charts
  - Word clouds by sentiment
- Support for integration with Tableau dashboards

---

## Tech Stack

| Area              | Tools/Libs                             |
|-------------------|----------------------------------------|
| Language          | Python                                 |
| ML Model          | XGBoost (via scikit-learn wrapper)     |
| Preprocessing     | NLTK, pandas, RAKE-NLTK                |
| Visualizations    | Matplotlib, WordCloud, Tableau         |
| File Management   | joblib, os                             |

---

## Project Structure

```
social-media-sentiment-analysis/
│
├── data/
│   ├── raw/                 # Input CSV files (unprocessed)
│   └── processed/           # Cleaned and labeled CSVs (model-ready)
│
├── models/                  # Trained ML models (e.g., .pkl files)
│
├── notebooks/               # Jupyter notebooks for experimentation
│
├── src/                     # Python scripts for model training and prediction
│   ├── preprocessing.py     # Text cleaning and preprocessing functions
│   ├── train_model.py       # Training and saving the sentiment classifier
│   ├── predict_sentiment.py # Predicts sentiment for single/batch inputs
│   └── generate_visuals.py  # Creates sentiment plots and word clouds
│
├── visuals/                 # Auto-generated plots (e.g., word clouds, charts)
│
├── dashboards/              # Tableau dashboards for interactive analysis
│
└── readme.md                # Project documentation
```
 
---

## How to Run

1. Clone the repository

   
   git clone https://github.com/your-username/social-media-sentiment-analysis.git
   cd social-media-sentiment-analysis


2. Install dependencies

   
   pip install -r requirements.txt


3. Train the sentiment classifier

   
   python src/train_model.py
   

4. Predict sentiment on new data

   
   python src/predict_sentiment.py
   

5. Generate visualizations

   
   python src/generate_visuals.py
   


## Future Enhancements

- Add support for time-based sentiment trends
- Extend model to work with Reddit and other platforms
- Deploy as a web app using Streamlit
- Improve phrase extraction with spaCy or BERT-based models

---

## Author

Rohan Srinivasa
GitHub: [rohan111427](https://github.com/rohan111427)

---

## License

MIT License. See `LICENSE` file for details.