from itertools import product

import gensim.downloader
from gensim.parsing import preprocess_string

from gensim.parsing import remove_stopwords
from gensim.parsing import strip_multiple_whitespaces
from gensim.parsing import strip_punctuation


# name of a pre-trained model which word embeddings should be used
MODEL = "glove-wiki-gigaword-300"


def get_embeddings():
    """Get word embeddings."""
    return gensim.downloader.load(MODEL)


def most_similar(embeddings, text: str, topn=10) -> list:
    words = preprocess_string(
        text,
        filters=[
            lambda x: x.lower(),
            strip_punctuation,
            strip_multiple_whitespaces,
            remove_stopwords,
        ],
    )

    if not words:
        return []

    if len(words) == 1:
        return embeddings.most_similar(words[0], topn=topn)

    similar_words = {}
    for word in words:
        # evaluate similar words and
        # set current word as the most similar (for later cartesian product)
        similar_words[word] = [(word, 1)] + embeddings.most_similar(word, topn=topn)

    candidates = []
    for candidate in product(*similar_words.values()):
        words_, similarities = list(zip(*candidate))
        candidates.append((" ".join(words_), sum(similarities)))

    candidates.sort(key=lambda x: -x[1])  # descending similarity sums

    # remove a phrase that is built from original words only
    candidates = [c for c in candidates if c[0] != " ".join(words)]
    return candidates[:topn]


embeddings = get_embeddings()


if __name__ == "__main__":
    print(f"Most similar to 'Pepsi': {most_similar(embeddings, 'Pepsi')}\n")
    print(f"Most similar to 'PiS': {most_similar(embeddings, 'PiS')}\n")
    print(f"Most similar to 'Nike shoes': {most_similar(embeddings, 'Nike shoes')}\n")
    print(f"Most similar to 'Cold Pepsi': {most_similar(embeddings, 'Cold Pepsi')}\n")
    print(
        f"Most similar to 'Broken screen': {most_similar(embeddings, 'Broken screen')}\n"
    )
    print(f"Most similar to 'iPhone fix': {most_similar(embeddings, 'iPhone fix')}\n")
