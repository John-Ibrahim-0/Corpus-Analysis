import requests
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

    if len(parts) != 2:
        raise ValueError("Could not split the text into Old Testament and New Testament.")
    
    ot_text = parts[0]
    nt_text = parts[1]
