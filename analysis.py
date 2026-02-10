import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import gensim
from gensim import corpora
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

def run_topic_modelling(df, num_topics=10):
    """
    Runs LDA Topic Modelling using Gensim.
    """
    print(f"\n--- Running LDA Topic Modelling ({num_topics} Topics) ---")

    stops = preprocessor.get_custom_stopwords()

    # tokenize for Gensim
    texts = [
        [word for word in preprocessor.clean_text(doc).split() if word not in stops and len(word) > 2]
        for doc in df['text']
    ]

    # create dictionary and corpus
    dictionary = corpora.Dictionary(texts)
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    corpus = [dictionary.doc2bow(text) for text in texts]

    # run LDA
    lda_model = gensim.models.LdaModel(
        corpus = corpus,
        id2word = dictionary,
        num_topics = num_topics,
        passes = 15,
        random_state = 42
    )

    # print topics
    print("Topics Found:")

    topn = 25

    for index in range(num_topics):
        topic_terms = lda_model.show_topic(index, topn=topn)

        formatted_terms = [f"{word} ({prob:.3f})" for word, prob in topic_terms]
        terms_string = ", ".join(formatted_terms)

        print(f"\nTopic {index}: {terms_string}")
    
    # calculate topic distribution per category
    print("\nTop Topics by Category:")
    df["topic_distribution"] = [lda_model.get_document_topics(dictionary.doc2bow(text), minimum_probability=0) for text in texts]

    for cat in df["category"].unique():
        cat_docs = df[df["category"] == cat]["topic_distribution"]
        avg_distribution = np.zeros(num_topics)
        count = 0

        for doc_list in cat_docs:
            for topic_id, prob in doc_list:
                avg_distribution[topic_id] += prob
            count += 1
        
        if count > 0:
            avg_distribution /= count
        
        # get top 3 topics
        top_topics = np.argsort(avg_distribution)[::-1][:3]
        print(f"\n{cat}:")
        for t in top_topics:
            print(f"  Topic {t} (Prob: {avg_distribution[t]:.4f})")
