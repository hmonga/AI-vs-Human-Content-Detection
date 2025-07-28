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
- **AI.py: Dataset loading and initial preprocessing
- **AI_Data_Cleaning.py: Handles missing values and imputation
- **AI_Data_Tests.py: Runs statistical tests (t-test, Mann-Whitney U)
- **AI_Data_Visualization.py: Creates visualizations like boxplots and distributions
- **Logistic_Regression.py: Fits logistic regression and analyzes coefficients
- **Random_Forest.py: Trains a Random Forest and ranks feature importance
- **Model_Eval.py: Compares model performance using ROC/AUC
- **ai_human_content_detection_dataset.csv: The dataset used throughout the analysis

## 💡 Future Work
- Leverage LLM embeddings or attention weights
- Apply deep learning for feature extraction
