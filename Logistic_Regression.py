import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("ai_human_content_detection_dataset.csv")

# Drop non-numeric + non-predictive columns
df_model = df.drop(columns=['text_content', 'content_type'])

# Fill missing values
df_model = df_model.fillna(df_model.mean(numeric_only=True))

print("Missing values after fill:\n", df_model.isnull().sum())

# Prepare X and y
X = df_model.drop(columns=['label'])
y = df_model['label']

# Scale the features (important for LR)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train logistic regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_scaled, y)

# Map coefficients to feature names
coeffs = pd.Series(lr.coef_[0], index=X.columns)

# Show top features by absolute strength
top_coeffs = coeffs.abs().sort_values(ascending=False)

print("\n🔎 Top 5 Most Influential Features (Logistic Regression):")
print(top_coeffs.head(5))
