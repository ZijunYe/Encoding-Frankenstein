# Import necessary libraries
import json
import requests
import pandas as pd
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
from textblob import TextBlob


url = 'https://raw.githubusercontent.com/ZijunYe/Encoding-Frankenstein/main/DataPreparation/Chapter17-Quotes/timeChunkQuote.json'

response = requests.get(url)
response.raise_for_status()

data = response.json()

# Load the JSON file
# with open('url', 'r') as file:
#     data = json.load(file)

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

# Define a function to map sentiment values to PNG file paths
# def sentiment_to_png(sentiment):
#     if sentiment > 0.12:
#         return '/content/PartyingFace.png'  # Extreme Positive
#     elif sentiment > 0.1:
#         return '/content/GrinningFace.png'    # Positive 
#     elif sentiment < -0.1:
#         return '/content/CryingFace.png'    # Extreme negative 
#     elif sentiment < -0.05:
#         return '/content/FrowningFace.png'    # negative 
#     else:
#         return '/content/NeutralFace.png'  # neutral 
def sentiment_to_png(sentiment):
    base_url = 'https://raw.githubusercontent.com/ZijunYe/Encoding-Frankenstein/main/DataPreparation/sentimentalImage/'
    
    if sentiment > 0.12:
        return base_url + 'PartyingFace.png'  # Extreme Positive
    elif sentiment > 0.1:
        return base_url + 'GrinningFace.png'  # Positive
    elif sentiment < -0.1:
        return base_url + 'CryingFace.png'    # Extreme Negative
    elif sentiment < -0.05:
        return base_url + 'FrowningFace.png'  # Negative
    else:
        return base_url + 'NeutralFace.png'   # Neutral


# if sentiment > 0.1:
#         return "😊"  # Positive
#     elif sentiment > 0.12:
#         return "😢"  # really positive 
#     elif sentiment < -0.05:
#         return "😢"  # Negative
#     elif sentiment < -0.1:
#         return "😢"  # extremely Negative
#     else:
#         return "😐"  # Neutral
# Function to load a PNG image as an OffsetImage

# def get_png_image(path, zoom=0.1):
#     img = Image.open(path)  # Open the PNG image file
#     return OffsetImage(img, zoom=zoom)  # Convert to an OffsetImage


def get_png_image_from_url(url, zoom=0.1):
    response = requests.get(url)
    response.raise_for_status()  # Check for errors
    img = Image.open(BytesIO(response.content))  # Load image from the response content
    return OffsetImage(img, zoom=zoom)  # Convert to an OffsetImage


# Plot the sentiment analysis
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(plot_data.index, plot_data['Victor'], label='Victor', linestyle='-', linewidth=1, alpha=1)
ax.plot(plot_data.index, plot_data['Creature'], label='Creature', linestyle='-', linewidth=1, alpha=1)

# Add PNG images as markers for Victor
for i, (x, y) in enumerate(zip(plot_data.index, plot_data['Victor'])):
    png_path = sentiment_to_png(y)
    ab = AnnotationBbox(get_png_image_from_url(png_path, zoom=0.35), (i, y), frameon=False)
    ax.add_artist(ab)

# Add PNG images as markers for Creature
for i, (x, y) in enumerate(zip(plot_data.index, plot_data['Creature'])):
    png_path = sentiment_to_png(y)
    ab = AnnotationBbox(get_png_image_from_url(png_path, zoom=0.35), (i, y), frameon=False)
    ax.add_artist(ab)

# Add labels and legend
ax.set_title('Emotional Fluctuations with PNG Markers', fontsize=14)
ax.set_xlabel('Incidents (Timeline)', fontsize=12)
ax.set_ylabel('Sentiment Level', fontsize=12)
ax.set_xticks(range(len(plot_data.index)))
ax.set_xticklabels(plot_data.index, rotation=45, ha='right', fontsize=10)
ax.tick_params(axis='y', labelsize=10)
ax.grid(alpha=0.3)
ax.legend(fontsize=12)
plt.tight_layout()

# Display the plot
plt.show()