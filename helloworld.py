print("Hello") + " World"

import sys
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.classify import NaiveBayesClassifier
from nltk.util import split

# Book input handling
def get_book_description():
    return "This is a sample book description. Please provide the actual content."

# Title, author, and book description input
title = sys.stdin.readline()
author = sys.stdin.readline()
book_description = get_book_description()

# Basic summarization logic using TF-IDF for key words related to genres
vectorizer = TfidfVectorizer(max_features=200)
tfidf = vectorizer.fit([title.lower(), author.lower(), book_description.lower()])
features = split([title.lower(), author.lower(), book_description.lower()], maxsplit=3)

# Classify into genres
classes = NaiveBayesClassifier()
classes.fit(features, ['Fiction', 'Non-Fiction'])

def summarize():
    for c in classes:
        if c['name'] == "Fiction":
            print("The book is about Fiction. Summary:\n")
            summary_words = [word for word in features if word in c['words']]
            print(f"{len(summary_words)} key words related to Fiction.")
        else:
            print("The book is about Non-Fiction. Summary:\n")
            summary_words = [word for word in features if word in classes['non_fiction_words']]
            print(f"{len(summary_words)} key words related to Non-Fiction.")

def display_summary():
    print("Summary:")
    print("1. Title: ", title)
    print("2. Author: ", author)
    print("3. Book Description: ", book_description)

def main():
    summary = summarize()
    display_summary()

if __name__ == "__main__":
    main()
