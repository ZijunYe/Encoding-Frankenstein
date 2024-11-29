# Import necessary libraries
import json
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob

# Load the JSON file
with open('/content/timeChunkQuote.json', 'r') as file:
    data = json.load(file)

# Initialize a DataFrame to store the analysis
sentiment_data = []

# Process each incident in order of appearance
for incident in data.keys():
    for entry in data[incident]:
        character = entry['character']
        text = entry['text']
        # Perform sentiment analysis using TextBlob
        sentiment = TextBlob(text).sentiment.polarity
        sentiment_data.append({
            'incident': incident,
            'character': character,
            'text': text,
            'sentiment': sentiment
        })

# Convert the sentiment data to a DataFrame
df = pd.DataFrame(sentiment_data)

# Ensure incidents appear in the correct order
incident_order = list(data.keys())
df['incident'] = pd.Categorical(df['incident'], categories=incident_order, ordered=True)

# Separate data for Victor and the Creature
victor_df = df[df['character'].str.lower() == 'victor']
creature_df = df[df['character'].str.lower() == 'creature']

# Aggregate sentiment by incidents
victor_sentiment = victor_df.groupby('incident')['sentiment'].mean()
creature_sentiment = creature_df.groupby('incident')['sentiment'].mean()

# Create a DataFrame for the plot
plot_data = pd.DataFrame({'Victor': victor_sentiment, 'Creature': creature_sentiment})

# Fill missing sentiment values with the previous value (to maintain continuity in the line plot)
plot_data.fillna(method='ffill', inplace=True)
plot_data.fillna(method='bfill', inplace=True)  # Handle any leading NaNs

# Define a function to map sentiment values to emojis
def sentiment_to_emoji(sentiment):
    if sentiment > 0.1:
        return "😊"  # Positive
    elif sentiment > 0.12:
        return "😢"  # really positive 
    elif sentiment < -0.05:
        return "😢"  # Negative
    elif sentiment < -0.1:
        return "😢"  # extremely Negative
    else:
        return "😐"  # Neutral

# Get x-axis positions and labels
x_positions = range(len(plot_data.index))
x_labels = plot_data.index

# Plot the sentiment analysis with emojis
plt.figure(figsize=(14, 7))
plt.plot(x_positions, plot_data['Victor'], label='Victor', linestyle='-', linewidth=2, alpha=1)  # Line for Victor
plt.plot(x_positions, plot_data['Creature'], label='Creature', linestyle='-', linewidth=2, alpha=1)  # Line for Creature

# Add emojis as markers
for i, incident in enumerate(plot_data.index):
    victor_emoji = sentiment_to_emoji(plot_data['Victor'].iloc[i])
    creature_emoji = sentiment_to_emoji(plot_data['Creature'].iloc[i])

    # Place emojis for Victor
    plt.text(
        x=i, 
        y=plot_data['Victor'].iloc[i], 
        s=victor_emoji, 
        fontsize=12, 
        ha='center', 
        va='center'
    )
    
    # Place emojis for Creature
    plt.text(
        x=i, 
        y=plot_data['Creature'].iloc[i], 
        s=creature_emoji, 
        fontsize=12, 
        ha='center', 
        va='center'
    )

# Add labels and legend
plt.title('Emotional Fluctuations with Emojis in Sentiment Analysis', fontsize=14)
plt.xlabel('Incidents (Timeline)', fontsize=12)
plt.ylabel('Sentiment Level', fontsize=12)
plt.xticks(ticks=x_positions, labels=x_labels, rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.grid(alpha=0.3)
plt.legend(fontsize=12)
plt.tight_layout()

# Display the plot
plt.show()
