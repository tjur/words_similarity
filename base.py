from itertools import product
from typing import Any
from typing import Tuple

from gensim.parsing import preprocess_string
from gensim.parsing import remove_stopwords
from gensim.parsing import strip_multiple_whitespaces
from gensim.parsing import strip_punctuation


class Embeddings:
    """Base class for embeddings."""

    _embeddings: Any

    def __init__(self, model_name: str):
        """Init."""
        # model_name is a name of a pre-trained model whose
        # word embeddings should be loaded and used
        self._embeddings = self._load_model(model_name)

    def _load_model(self, model_name: str) -> Any:
        """Load model."""
        raise NotImplementedError

    def _most_similar_words(self, word: str, topn: int) -> list[Tuple[str, float]]:
        """Return words most similar to the given word with similarity values."""
        raise NotImplementedError

    def most_similar(self, text: str, topn=10) -> list[Tuple[str, float]]:
        """Return phrases most similar to the given text with similarity values."""

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
            return self._most_similar_words(words[0], topn=topn)

        similar_words = {}
        for word in words:
            # evaluate similar words and
            # set current word as the most similar (for later cartesian product)
            similar_words[word] = [(word, 1.0)] + self._most_similar_words(
                word, topn=topn
            )

        candidates = []
        for candidate in product(*similar_words.values()):
            words_, similarities = list(zip(*candidate))
            candidates.append((" ".join(words_), sum(similarities)))

        candidates.sort(key=lambda x: -x[1])  # descending similarity sums

        # remove a phrase that is built from original words only
        candidates = [c for c in candidates if c[0] != " ".join(words)]
        return candidates[:topn]
