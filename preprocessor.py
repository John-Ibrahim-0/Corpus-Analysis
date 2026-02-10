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

    return list(base_stops.union(archaic_stops))
