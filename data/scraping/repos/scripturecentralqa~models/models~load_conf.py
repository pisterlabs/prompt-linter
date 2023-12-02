"""Load conference talks."""

import json
import os
from typing import Any
from typing import Iterator
from typing import cast
from urllib.parse import urljoin
from urllib.parse import urlparse

from bs4 import BeautifulSoup
from langchain.document_loaders.base import BaseLoader
from langchain.schema.document import Document
from markdownify import MarkdownConverter  # type: ignore
from tqdm import tqdm

from models.load_utils import clean


class ConferenceMarkdownConverter(MarkdownConverter):  # type: ignore
    """Create a custom MarkdownConverter."""

    def __init__(self, **kwargs: Any):
        """Initialize custom MarkdownConverter."""
        super().__init__(**kwargs)
        self.base_url = kwargs.get("base_url", "")

    def convert_a(self, el, text, convert_as_inline):  # type: ignore
        """Join hrefs with a base url."""
        if "href" in el.attrs:
            el["href"] = urljoin(self.base_url, el["href"])
        return super().convert_a(el, text, convert_as_inline)

    def convert_p(self, el, text, convert_as_inline):  # type: ignore
        """Add anchor tags to paragraphs with ids."""
        if el.has_attr("id") and len(el["id"]) > 0:
            _id = el["id"]
            text = f'<a name="{_id}"></a>{text}'  # noqa: B907
        return super().convert_p(el, text, convert_as_inline)


# Create shorthand method for custom conversion
def _to_markdown(html: str, **options: Any) -> str:
    """Convert html to markdown."""
    return cast(str, ConferenceMarkdownConverter(**options).convert(html))


def load_conference_talk(url: str, html: str, bs_parser: str = "html.parser") -> Document:
    """Load a conference talk from a url and html."""
    path_components = urlparse(url).path.split("/")
    year, month = path_components[3:5]
    soup = BeautifulSoup(html, bs_parser)
    title = soup.select_one("article header h1")
    author = soup.select_one("article p.author-name")
    author_role = soup.select_one("article p.author-role")
    body = soup.select_one("article div.body-block")
    if body:
        content = clean(_to_markdown(str(body), base_url=url, heading_style="ATX", strip=["script", "style"]))
    else:
        content = ""
    metadata = {
        "year": year,
        "month": month,
        "url": url,
        "title": clean(title) if title else "",
        "author": clean(author) if author else "",
        "author_role": clean(author_role) if author_role else "",
    }
    return Document(page_content=content, metadata=metadata)


class ConferenceTalkLoader(BaseLoader):
    """Loader for General Conference Talks."""

    def lazy_load(self) -> Iterator[Document]:
        """A lazy loader for Documents."""
        raise NotImplementedError(f"{self.__class__.__name__} does not implement lazy_load()")

    def __init__(self, path: str = "", bs_parser: str = "html.parser"):
        """Initialize loader."""
        super().__init__()
        self.path = path
        self.bs_parser = bs_parser

    def load(self, verbose: bool = False) -> list[Document]:
        """Load documents from path."""
        docs = []
        for filename in tqdm(os.listdir(self.path), disable=not verbose):
            path = os.path.join(self.path, filename)
            with open(path, encoding="utf8") as f:
                data = json.load(f)
            doc = load_conference_talk(data["url"], data["html"], bs_parser=self.bs_parser)
            if not doc.metadata["title"] or not doc.page_content:
                if verbose:
                    print("Missing title or content - skipping", filename)
                continue
            if not doc.metadata["author"]:
                if verbose:
                    print("Missing author", filename)
            docs.append(doc)
        return docs
