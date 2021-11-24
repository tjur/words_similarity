from typing import Any
from typing import Tuple

import gensim.downloader
import fasttext.util

from base import Embeddings


class GloveWikiEmbeddings(Embeddings):
    """Glove embeddings."""

    def _load_model(self, model_name: str) -> Any:
        """Load model."""
        return gensim.downloader.load(model_name)

    def _most_similar_words(self, word: str, topn: int) -> list[Tuple[str, float]]:
        """Return words most similar to the given word with similarity values."""
        return self._embeddings.most_similar(word, topn=topn)


class FasttextEmbeddings(Embeddings):
    """Fasttext embeddings."""

    def _load_model(self, model_name: str) -> Any:
        """Load model."""
        file_name = fasttext.util.download_model(lang_id=model_name, if_exists="ignore")
        return fasttext.load_model(file_name)

    def _most_similar_words(self, word: str, topn: int) -> list[Tuple[str, float]]:
        """Return words most similar to the given word with similarity values."""
        result = self._embeddings.get_nearest_neighbors(word, k=topn)
        return [(w, sim) for sim, w in result]


if __name__ == "__main__":
    english_texts = [
        "Pepsi",
        "PiS",
        "Nike shoes",
        "Cold Pepsi",
        "Broken screen",
        "Iphone fix",
    ]

    # glove_wiki_en = GloveWikiEmbeddings("glove-wiki-gigaword-300")
    # print("Model: GloVe Wikipedia (english)")
    # for text in english_texts:
    #     print(f"Most similar to {text!r}: {glove_wiki_en.most_similar(text)}\n")
    # print()

    # fasttext_en = FasttextEmbeddings("en")
    # print("Model: fastText (english)")
    # for text in english_texts:
    #     print(f"Most similar to {text!r}: {fasttext_en.most_similar(text)}\n")
    # print()
