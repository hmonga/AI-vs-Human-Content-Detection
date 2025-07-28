import pandas as pd
 
df = pd.read_csv("ai_human_content_detection_dataset.csv")
print(df.info())

# Print null value count for each column
print("\n Missing Values Summary:")
print(df.isnull().sum())

print("Rows before dropping:", len(df))
print("Rows after dropping:", len(df.dropna()))

# Fill all numeric columns with their column means
df.fillna(df.mean(numeric_only=True), inplace=True)


df['sentiment_score'] = df['sentiment_score'].fillna(df['sentiment_score'].mean())

print("\n🧼 Null values after fill:")
print(df.isnull().sum())
