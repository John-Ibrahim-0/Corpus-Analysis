import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import preprocessor

def run_naive_bayes(df, vectorizer_type="count"):
    """
    Computes and prints the Log-Likelihood Ratio for distinctive words.
    """
    print(f"\n--- Running Naive Bayes Analysis ({vectorizer_type.upper()}) ---")

    stops = preprocessor.get_custom_stopwords()

    if vectorizer_type == "tfidf":
        vec = TfidfVectorizer(stop_words=stops, max_features=2000, preprocessor=preprocessor.clean_text)
    else:
        vec = CountVectorizer(stop_words=stops, max_features=2000, preprocessor=preprocessor.clean_text)

    X = vec.fit_transform(df['text'])
    y = df['category']
    feature_names = vec.get_feature_names_out()

    # train NB model
    clf = MultinomialNB(alpha = 1.0) # add-one smoothing
    clf.fit(X, y)

    classes = clf.classes_
    results = {}

    for i, category in enumerate(classes):
        other_index = 1 - i

        # LLR calculation: log(P(word|category)) - log(P(word|not category))
        log_prob_c = clf.feature_log_prob_[i]
        log_prob_not_c = clf.feature_log_prob_[other_index]
        llr = log_prob_c - log_prob_not_c

        # get top 10 words
        top_indices = np.argsort(llr)[::-1][:10]
        top_words = [(feature_names[idx], llr[idx]) for idx in top_indices]
        results[category] = top_words
    
    # display results
    print(f"{"Category":<20} | {'Top Distinctive Words (Word, LLR)':<60}")
    print("-" * 85)
    
    for cat, words in results.items():
        word_str = ", ".join([f"({w}, {s:.2f})" for w, s in words])
        print(f"{cat:<20} | {word_str}")
