import spacy
import csv
import re

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def extract_adjectives_from_text(text, chapter_narrator_map):
    """
    Extract adjectives from the text with chapter and narrator information.
    
    Parameters:
    - text: Full text of the novel.
    - chapter_narrator_map: Dictionary mapping chapter numbers to narrators.
    
    Returns:
    - List of dictionaries containing chapter, adjective, and narrator.
    """
    # Split text into chapters
    chapters = re.split(r'Chapter\s+\d+', text)
    chapters = [ch.strip() for ch in chapters if ch.strip()]

    adjectives_data = []

    for chapter_number, chapter_text in enumerate(chapters, 1):
        # Determine the narrator
        narrator = chapter_narrator_map.get(chapter_number, "Unknown")

        # Process the chapter text using spaCy
        doc = nlp(chapter_text)

        # Extract adjectives
        for token in doc:
            if token.pos_ == "ADJ":
                adjectives_data.append({
                    "Chapter": chapter_number,
                    "Adjective": token.text.lower(),
                    "Narrator": narrator
                })

    return adjectives_data

# Map chapters to narrators
chapter_narrator_map = {
    **{i: "Victor Frankenstein" for i in range(1, 11)},  # Chapters 1-10
    **{i: "The Creature" for i in range(11, 17)},        # Chapters 11-16
    **{i: "Victor Frankenstein" for i in range(17, 25)} # Chapters 17-24
}

# Load the text from the novel (update this path to your file's location)
text_file_path = "Frankenstein.txt"
with open(text_file_path, 'r', encoding='utf-8') as file:
    full_text = file.read()

# Extract adjectives
adjectives = extract_adjectives_from_text(full_text, chapter_narrator_map)

# Save adjectives to a CSV file
output_csv_path = "Frankenstein_Adjectives.csv"
with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ["Chapter", "Adjective", "Narrator"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    writer.writeheader()
    writer.writerows(adjectives)

print(f"Adjectives have been extracted and saved to {output_csv_path}")
