# 🧠 Forecasting & Text Classification Projects

This repository presents two applied machine learning tasks:

1. **Forecasting**: A comparative analysis of time series forecasting methods, including the Random Walk model and Facebook Prophet.
2. **Text Classification**: An NLP task to classify text data using preprocessing, feature engineering, and supervised learning models.

---

## 👥 Authors

- **Antonella Convertini**
- **Adriano Meligrana**
- **Inas El Kouch**
- **Pooya Sabbagh**

---

## 📁 Repository Structure

Forecasting_And_TextClassification/

├── Prophet_Task.ipynb # Time series forecasting using Prophet

├── TEXT_CLASSIFICATION_task.ipynb # Text classification with supervised models

├── Presentation.pdf # Summary deck on forecasting evaluation

├── README.md # Project documentation


---

## 📈 1. Time Series Forecasting with Prophet vs Random Walk

**Notebook**: `Prophet_Task.ipynb`  
**Slides**: `Presentation.pdf`

### 🧭 Objective

Evaluate the performance of **Facebook Prophet** versus a naive **Random Walk** model in predicting a univariate time series.

### 🔍 Key Concepts

- **Random Walk**: Predicts next value as the current one (i.e., no change). Surprisingly difficult to outperform in noisy time series.
- **Prophet**: A robust, decomposable model by Meta (Facebook) designed for time series with trend and seasonality.

### 📊 Metrics

- **Mean Absolute Error (MAE)**  
- **Sign Accuracy**: Measures how well the model predicts the direction of change

### 📌 Key Findings

- Random Walk surprisingly achieved **lower MAE** than Prophet in this setup
- Prophet added structure but didn’t consistently improve directional accuracy
- Simple baselines can outperform complex models on volatile data

---

## 📝 2. Text Classification Task

**Notebook**: `TEXT_CLASSIFICATION_task.ipynb`

### 🧭 Objective

Perform supervised classification on a text dataset by transforming raw text into meaningful features and training classification models.

### 🧪 Workflow

1. **Data Preprocessing**:
   - Tokenization
   - Stopword removal
   - Lemmatization/stemming

2. **Feature Engineering**:
   - TF-IDF Vectorization
   - Word embeddings (if applicable)

3. **Modeling**:
   - Logistic Regression
   - SVM / Naive Bayes (if included)
   - Evaluation using Accuracy, F1-score, Confusion Matrix

### 🔍 Key Learnings

- Importance of preprocessing in text classification
- Trade-off between model complexity and interpretability
- Model evaluation should go beyond accuracy (especially in imbalanced data)

---

## ⚙️ Requirements

### 🐍 Python Libraries (for both notebooks)

```bash
pip install pandas numpy matplotlib seaborn
pip install scikit-learn
pip install plotly
pip install prophet
pip install nltk spacy
python -m nltk.downloader stopwords
python -m spacy download en_core_web_sm
