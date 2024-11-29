import json
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import matplotlib.pyplot as plt

# Ensure you have the VADER lexicon for sentiment analysis
nltk.download('vader_lexicon')

# Load the JSON file
file_path = '/Chapter17_cleaned_character_data.json'
with open(file_path, 'r') as file:
    data = json.load(file)

# Flatten data into a dataframe
character_data = []
for character, content in data.items():
    for text_type, texts in content.items():
        for text in texts:
            character_data.append({
                'character': character,
                'text_type': text_type,
                'text': text
            })

df = pd.DataFrame(character_data)

# Initialize VADER SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Analyze sentiment for each text
df['sentiment'] = df['text'].apply(lambda x: sia.polarity_scores(x))

# Extract sentiment components
df['positive'] = df['sentiment'].apply(lambda x: x['pos'])
df['negative'] = df['sentiment'].apply(lambda x: x['neg'])
df['neutral'] = df['sentiment'].apply(lambda x: x['neu'])
df['compound'] = df['sentiment'].apply(lambda x: x['compound'])

# Drop the sentiment dictionary column for clarity
df = df.drop(columns=['sentiment'])

# Save processed data
df.to_csv('/mnt/data/processed_sentiment_analysis.csv', index=False)

# Aggregated sentiment scores per character and text type
summary = df.groupby(['character', 'text_type']).mean()[['positive', 'negative', 'neutral', 'compound']]
summary.reset_index(inplace=True)

# Visualize the data
for character in df['character'].unique():
    char_data = df[df['character'] == character]
    plt.figure(figsize=(10, 5))
    plt.title(f"Sentiment Analysis for {character.capitalize()}")
    plt.hist(char_data['compound'], bins=20, alpha=0.7, label='Compound Sentiment')
    plt.axvline(0, color='red', linestyle='--', label='Neutral Line')
    plt.xlabel('Compound Sentiment Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

# Display processed dataframe to user
import ace_tools as tools; tools.display_dataframe_to_user(name="Sentiment Analysis Results", dataframe=df)
