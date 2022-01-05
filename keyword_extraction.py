from typing import Any
from typing import Tuple

import requests
from bs4 import BeautifulSoup
from gensim.parsing import strip_multiple_whitespaces
from readability import Document
from yake import yake

from base import KeywordExtractor


def get_page_content(url: str) -> Tuple[str, str]:
    """Get HTML page and return its cleaned content with a title."""
    response = requests.get(url)
    doc = Document(response.text)
    title = doc.title()
    # doc.summary() does not work well on bbc.com articles -
    # it finds only one paragraph
    soup = BeautifulSoup(doc.summary(), "html.parser")
    text = soup.get_text()
    text = strip_multiple_whitespaces(text)
    return text, title


class YAKEKeywordExtractor(KeywordExtractor):
    """YAKE keyword extractor."""

    _kw_extractor: Any

    def __init__(self, num_of_keywords=12, *args, **kwargs):
        """Init."""
        super().__init__()
        language = kwargs.get("language", "en")
        max_ngram_size = kwargs.get(
            "max_ngram_size", 3
        )  # max number of words in a keyword
        # the limit of the duplication of words in different keywords
        # for 0.9 the repetition of words is allowed in keywords
        deduplication_threshold = kwargs.get("deduplication_threshold", 0.9)
        self._kw_extractor = yake.KeywordExtractor(
            lan=language,
            n=max_ngram_size,
            dedupLim=deduplication_threshold,
            top=num_of_keywords,
            features=None,
            stopwords=None,
        )

    def extract_keywords(self, text: str, *args, **kwargs) -> list[tuple[str, float]]:
        """
        Extract keywords from the given text.

        For each keyword return the score as well.
        The lower the score, the more relevant the keyword is.
        """
        return self._kw_extractor.extract_keywords(text)


if __name__ == "__main__":
    web_page_urls = [
        "https://www.cnbc.com/2021/11/02/nike-is-quietly-preparing-for-the-metaverse-.html",
        "https://www.theguardian.com/technology/2021/nov/24/techscape-apple-self-repair",
        "https://www.fool.com/investing/2021/11/01/is-coca-cola-stock-a-buy-after-q3-earnings/",
        "https://news.sky.com/story/two-injured-as-pigs-storm-golf-course-in-west-yorkshire-12477545",
        "https://news.sky.com/story/new-zealand-bird-of-the-year-controversy-after-contest-is-won-by-a-bat-12456979",
        "https://www.cnet.com/features/future-proof-how-scotch-distilleries-are-crafting-whisky-for-tomorrows-world/",
        "https://www.theguardian.com/artanddesign/2020/aug/12/burning-man-pink-floyd-wish-you-were-here-aubrey-powell-best-photograph",
        "https://www.theguardian.com/music/2021/sep/17/genesis-prog-phil-collins-health-final-tour",
        "https://www.euronews.com/travel/2021/11/14/trentino-the-undiscovered-italian-region-that-s-a-must-for-foodies-and-skiers",
        "https://rossiwrites.com/italy/lake-garda/riva-del-garda-italy/",
        "https://www.theguardian.com/lifeandstyle/2021/nov/21/turn-over-a-new-leaf-five-inspiring-books-about-gardening-chosen-by-nell-card",
        "https://news.sky.com/story/covid-19-around-8-000-flights-cancelled-globally-between-christmas-eve-and-boxing-day-due-to-coronavirus-12504415",
        "https://www.theguardian.com/games/2021/nov/17/grand-theft-auto-the-trilogy-the-definitive-edition-review-an-infuriating-disappointment",
        "https://www.theguardian.com/travel/2021/dec/16/estonia-in-winter-into-the-wilds-by-canoe-and-bog-shoe",
    ]

    yake_en_kw_extractor = YAKEKeywordExtractor(
        language="en", max_ngram_size=3, num_of_keywords=10, deduplication_threshold=0.9
    )

    print("YAKE extractor (english)")
    for url in web_page_urls:
        content, title = get_page_content(url)
        print(
            f"Keywords for {url!r} ({title!r}):\n"
            f"{yake_en_kw_extractor.extract_keywords(content)}"
        )
        print()
