# AI-vs-Human-Content-Detection
Analyze and classify AI vs Human-generated text using linguistic features. Includes EDA, statistical testing, feature importance (Random Forest &amp; Logistic Regression), and model evaluation (ROC/AUC). Focuses on interpretability, not just accuracy.

# AI vs Human Content Detection

This project investigates whether machine learning can distinguish between AI-generated and human-written text using linguistic features.

## 📂 Dataset
The dataset contains 1,367 samples labeled as `AI` or `Human`, with various features like:
- `flesch_reading_ease`
- `burstiness`
- `predictability_score`
- `lexical_diversity`, etc.

## 🔧 Methods
- Exploratory Data Analysis (EDA)
- Statistical hypothesis testing (T-tests, Mann-Whitney U)
- Correlation analysis
- Feature importance via Random Forest and Logistic Regression
- Model performance evaluation with ROC curves

## 📊 Results
- **Top Features (RF):** Burstiness, Sentiment, Passive Voice
- **Logistic Regression AUC:** ~0.53
- **Random Forest AUC:** ~0.52

## 📎 Files
- `AI.py`: Main data loading and summary
- `AI_Data_Tests.py`: Statistical testing
- `Logistic_Regression.py`: Feature importance and modeling
- `Model_Evaluation.py`: ROC + Accuracy comparison

## 💡 Future Work
- Leverage LLM embeddings or attention weights
- Apply deep learning for feature extraction
