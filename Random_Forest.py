import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd

df = pd.read_csv("ai_human_content_detection_dataset.csv")

# Drop non-numeric columns
df_model = df.drop(columns=['text_content', 'content_type'])

# Split into features and target
X = df_model.drop(columns=['label'])
y = df_model['label']

# Train-test split (optional, here just for generalization)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Get importance scores
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

print("\n🌟 Top 5 Most Important Features:")
print(importances.head(5))

importances.head(10).plot(kind='barh')
plt.title("Top 10 Feature Importances (Random Forest)")
plt.gca().invert_yaxis()
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()



from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

# Predict probabilities and labels
y_probs = rf.predict_proba(X_test)[:, 1]
y_pred = rf.predict(X_test)

# Metrics
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_probs)

print(f"\n🎯 Accuracy: {acc:.2f}")
print(f"📈 ROC AUC Score: {auc:.2f}")

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_probs)
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f'Random Forest (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Random Forest')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
