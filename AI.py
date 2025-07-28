import pandas as pd
 
df = pd.read_csv("ai_human_content_detection_dataset.csv")
print(df.info())

# Print null value count for each column
print("\n Missing Values Summary:")
print(df.isnull().sum())



df['sentiment_score'] = df['sentiment_score'].fillna(df['sentiment_score'].mean())
