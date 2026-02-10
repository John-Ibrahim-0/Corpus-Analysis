import data_loader
import analysis

def main():
    print("=== Corpus Analysis of the King James Bible ===\n")

    # load the dataset
    df = data_loader.get_bible_corpus()

    if df.empty:
        print("Failed to load data. Exiting.")
        exit(1)
    
    # print basic stats
    print("\n" + "-" * 30)
    print("Dataset Summary:")
    print(df.groupby("category").size())
    print("-" * 30)

    # Naive Bayes analysis with Count Vectorizer
    analysis.run_naive_bayes(df, vectorizer_type="count")

if __name__ == "__main__":
    main()
