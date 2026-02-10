# Corpus Analysis

## Overview
This project implements a text analysis pipeline to compare the Old Testament (OT) and New Testament (NT) of the King James Bible. Using the Project Gutenberg corpus, the program:

* **Collects and Parses Data**: Automatically downloads the King James Bible and splits it into two categories (OT vs. NT) using chapter-level segmentation.
* **Preprocessing**: Cleans the text by removing verse numbers, normalizing case, and removing standard English and archaic Biblical stopwords (e.g. "thou", "thee").
* **Naive Bayes Analysis**: Computes the Log-Likelihood Ratios (LLR) to identify the most distinctive words for each testament.
* **Topic Modelling**: Uses Latent Dirichlet Allocation (LDA) via `gensim` to discover latent themes across the corpus.
* **Experimentation**: Compares results using standard Count Vectorization vs. TF-IDF normalization.

## Files and Directories
```
.
|--- main.py            # Entry point script
|--- data_loader.py     # Downloads and parses the Bible from Project Gutenberg
|--- preprocessor.py    # Text cleaning logic and custom stopword lists
|--- analysis.py        # Contains Naive Bayes (LLR) and LDA Topic Modelling logic
|--- README.md          # Project documentation
|--- pyproject.toml     # Project configuration and dependencies
|--- uv.lock            # Lockfile ensuring exact package version reproducibility
|--- .python-version    # Specifies the Python version (3.12) used by uv
```

## Requirements
* **Python**: 3.12
* **Package Manager**: `uv` (recommended) or `pip`
* **Required Python libraries**:
  * `pandas`
  * `numpy`
  * `scikit-learn`
  * `gensim`
  * `nltk`
  * `requests`

## Installation

This project is managed usin `uv`.

**1. Sync Dependencies (Recommended)**

If you have `uv` installed, simply run:
```
uv sync
```

**2. Manual Installation (Alternative)**

If you are using standard `pip`:
```
pip install requests pandas numpy scikit-learn gensim nltk
```

**3. NLTK Data**

The script will attempt to download required NLTK data (`stopwords`) automatically. If this fails, run:
```python
import nltk
nltk.download("stopwords")
```

## Usage
Run the main script from the command line.

**Using** `uv`:
```
uv run main.py
```

**Using standard Python**:
```
python main.py
```

## Output
The program outputs three main sections to the console:

**1. Dataset Statistics**

Displays the number of documents (chapters) fround per category.
```
Dataset Summary:
category
New Testament    260
Old Testament    929
```

**2. Naive Bayes Analysis**

A table showing the top 10 most distinctive words for each category based on LLRs.
```
Category             | Top Distinctive Words (Word, LLR)                           
-------------------------------------------------------------------------------------
New Testament        | (jesus, 8.16), (christ, 7.59), ...
Old Testament        | (judah, 4.73), (hosts, 4.43), ...
```

**3. Topic Modelling (LDA)**
A list of the top 15 latent topics found in the corpus, listing the top 25 terms per topic with their probabilities.
```
Topics Found:

Topic 0: earth (0.0162), like (0.0116), great (0.0099), ...

Topic 1: jacob (0.0281), son (0.0272), begat (0.0256), ...

.
.
.
```
