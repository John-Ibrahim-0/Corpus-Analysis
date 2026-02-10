import re
import nltk
from nltk.corpus import stopwords

# download stopword if not present
nltk.download('stopwords', quiet=True)

def get_custom_stopwords():
    """
    Returns a set of standard English stopwords plus archaic Biblical terms.
    """
    base_stops = set(stopwords.words('english'))
    archaic_stops = {"thou", "thee", "thy", "thine", "hath", "shalt", "wilt", "art", "dost", "doth", "ye", "lo", "would", "said", "unto", "came"}

    all_stops = base_stops.union(archaic_stops)

    clean_stops = {clean_text(word) for word in all_stops}

    return list(clean_stops)

def clean_text(text):
    """
    Removes verse numbers, newlines, and non-alphabetic characters.
    """
    # remove verse numbers (ex: "1:2")
    text = re.sub(r"\d+:\d+", "", text)

    # remove stray numbers
    text = re.sub(r"\d+", "", text)

    # remove newlines and extra whitespace
    text = text.replace("\r\n", " ").replace("\n", " ")

    # remove punctuation and special characters
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    return text.lower().strip()
