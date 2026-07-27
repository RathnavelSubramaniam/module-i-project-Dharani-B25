# Stock News Sentiment Analysis using GenAI Embeddings

## Problem Statement

### Business Context
Stock prices are heavily influenced by company financial performance, innovations, collaborations, and market sentiment. News and media reports can rapidly shift investor perception and, consequently, stock prices. With the sheer volume of news and opinions available, investors and financial analysts often struggle to manually track and interpret sentiment at scale.

### Problem Definition
An investment startup wants to leverage AI to interpret stock-related news and its impact on stock prices. Historical daily news for a NASDAQ-listed company, along with its daily stock price and trade volume, has been collected. This project builds an AI-driven sentiment analysis system that classifies news sentiment and studies its relationship with stock price movement.

## Data Dictionary
- `Date` : The date the news was released
- `News` : Content of the news article
- `Open` : Stock price ($) at the beginning of the day
- `High` : Highest stock price ($) reached during the day
- `Low` : Lowest stock price ($) reached during the day
- `Close` : Adjusted stock price ($) at the end of the day
- `Volume` : Number of shares traded during the day
- `Label` : Sentiment polarity of the news (-1: negative, 0: neutral, 1: positive)

## Project Workflow

1. **Data Loading & Overview** — Load `stock_news.csv`, inspect shape, nulls, duplicates, and data types.
2. **Exploratory Data Analysis**
   - Univariate: sentiment label distribution, numeric variable distributions, news length distribution, month-wise price/volume trends.
   - Bivariate: correlation heatmap, sentiment vs. price, price trend over time, news length vs. sentiment.
3. **Data Preprocessing** — Target variable (`Label`) extraction; 80:20 train-test split (random_state=42).
4. **Word Embeddings**
   - **Word2Vec** (Gensim) — custom-trained embeddings, sentence vectors via averaging.
   - **Sentence Transformers** — `BAAI/bge-base-en-v1.5` and `all-MiniLM-L6-v2` pretrained embeddings.
5. **Model Building** — Random Forest and Neural Network (Keras Sequential) classifiers trained on each embedding type:
   - Word2Vec + Random Forest
   - Word2Vec + Neural Network
   - Sentence Transformer (BAAI) + Random Forest
   - Sentence Transformer (BAAI) + Neural Network
   - MiniLM + Random Forest
   - MiniLM + Neural Network
6. **Model Evaluation** — Confusion matrices and classification metrics (Accuracy, Recall, Precision, F1-score) for train and test sets, compared across all six models.
7. **Conclusions & Recommendations** — Best-performing model identified and next steps for improvement outlined.

## Key Findings

- **Word2Vec** embeddings underperformed across both model types, with Random Forest showing severe overfitting (100% train accuracy vs. ~46% test accuracy) and Neural Networks staying around 44–50% accuracy.
- **Sentence Transformer embeddings** (BAAI and MiniLM) captured semantic meaning better than Word2Vec.
- **MiniLM + Neural Network** was the best-performing combination, achieving **95.7% training accuracy** and **55.7% test accuracy / 54.5% F1-score**, with the smallest train-test gap of all models tested.
- **Random Forest** models consistently overfit (100% train accuracy, 45–50% test accuracy) regardless of embedding used.

## Recommendations

- Overall test accuracy (max ~56%) is limited by the small dataset size — collecting more labeled news data is likely to improve performance significantly.
- Apply hyperparameter tuning (e.g., `max_depth`, `min_samples_leaf`, `GridSearchCV`) to reduce Random Forest overfitting.
- For the Neural Network, explore early stopping, dropout tuning, and class-imbalance handling to further close the train-test gap.
- **Final recommended model: MiniLM + Neural Network** — it is lighter (384 dimensions vs. BAAI's 768), faster to run, and generalizes best to unseen data.

## Tech Stack / Libraries

- **Data handling & viz:** numpy, pandas, matplotlib, seaborn
- **Word embeddings:** gensim (Word2Vec), sentence-transformers (BAAI/bge-base-en-v1.5, all-MiniLM-L6-v2)
- **Modeling:** scikit-learn (RandomForestClassifier), tensorflow.keras (Sequential Neural Network)
- **Deep learning backend:** torch, transformers
