import requests
import re
import pandas as pd

def get_bible_corpus():
    """
    Downloads the King James Bible from Project Gutenberg and parses it into Old Testament and New Testament categories.
    """
    print("Downloading the King James Bible from Project Gutenberg...") 
    url = "https://www.gutenberg.org/cache/epub/10/pg10.txt"

    try:
        response = requests.get(url)
        response.raise_for_status()
        text = response.text
    except requests.RequestException as e:
        print(f"An error occurred while downloading the Bible: {e}")
        return pd.DataFrame()
        
    # remove Project Gutenberg header and footer
    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK THE KING JAMES VERSION OF THE BIBLE ***"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK THE KING JAMES VERSION OF THE BIBLE ***"

    start_index = text.find(start_marker) + len(start_marker)
    end_index = text.find(end_marker)

    text = text[start_index:end_index]

    # split between OT and NT
    split_marker = "The New Testament of the King James Bible"
    parts = text.split(split_marker)

    if len(parts) != 3: # we expect 3 parts: ToC, OT, NT
        raise ValueError("Could not split the text into Old Testament and New Testament.")
    
    ot_text = parts[1]
    nt_text = parts[2]

    dataset = []

    # helper function to parse chapters
    def parse_testament(raw_text, label):
        pattern = r"\b\d+:1\b"

        # split the text wherever a new chapter starts (verse 1)
        chapters = re.split(pattern, raw_text)

        for chapter in chapters[1:]:
            cleaned_chapter = chapter.strip()
            dataset.append({"text": cleaned_chapter, "category": label})
    
    # parse both testaments
    print("Parsing the Old Testament...")
    parse_testament(ot_text, "Old Testament")

    print("Parsing the New Testament...")
    parse_testament(nt_text, "New Testament")

    df = pd.DataFrame(dataset)
    print(f"Data loaded successfully: {len(df)} documents (chapters) found.")

    return df
