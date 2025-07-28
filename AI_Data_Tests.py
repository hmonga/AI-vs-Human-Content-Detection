import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as mpatches

df = pd.read_csv("ai_human_content_detection_dataset.csv")

from scipy.stats import ttest_ind, mannwhitneyu

print(df['flesch_reading_ease'].isnull().sum())
print(df['flesch_reading_ease'].describe())

# Clip extreme invalid values (Flesch score should be roughly between 0 and 100)
df['flesch_reading_ease'] = df['flesch_reading_ease'].clip(lower=0, upper=100)

# Fill remaining NaNs with the mean
df['flesch_reading_ease'] = df['flesch_reading_ease'].fillna(df['flesch_reading_ease'].mean())
print(df['flesch_reading_ease'].isnull().sum())

features = [
    'lexical_diversity',
    'flesch_reading_ease',
    'avg_word_length',
    'burstiness',
    'predictability_score'
]

print("\n🔬 Statistical Comparison (AI vs Human):\n")
for feature in features:
    ai = df[df['label'] == 1][feature]
    human = df[df['label'] == 0][feature]

    # Run both tests
    t_stat, t_p = ttest_ind(ai, human, equal_var=False)
    u_stat, u_p = mannwhitneyu(ai, human, alternative='two-sided')

    print(f"{feature}:")
    print(f"  T-test p-value:         {t_p:.5f}")
    print(f"  Mann-Whitney p-value:   {u_p:.5f}\n")



plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap (Numeric Features)")
plt.tight_layout()
plt.show()


print(df['flesch_reading_ease'].describe())
