"""Load bgbmm."""

from bs4 import BeautifulSoup
from bs4 import Tag
from langchain.schema.document import Document

from models.load_utils import clean
from models.load_utils import to_markdown


def load_bgbmm(url: str, html: str, bs_parser: str = "html.parser") -> Document:
    """Load dc people from a url and html."""
    soup = BeautifulSoup(html, bs_parser)
    title = soup.find("h1", class_="page-title")

    body = soup.find("div", class_="accordion-content")
    if isinstance(body, Tag):
        unwanted = body.find("h1")
        if isinstance(unwanted, Tag):
            unwanted.extract()
            # unwanted_text = unwanted.get_text(strip=True)
            # print(unwanted_text)

    content = clean(to_markdown(str(body), base_url=url)) if body else ""

    metadata = {
        "url": url,
        "title": clean(title) if title else "",
    }
    return Document(page_content=content, metadata=metadata)
