from datetime import datetime
from pathlib import Path
from typing import Iterator

from langchain.docstore.document import Document
from langchain.document_loaders import ReadTheDocsLoader


class RTDHtmlPageLoader(ReadTheDocsLoader):
    """directory path for readthedocs documents

    $ wget -r -np -A.html https://docs.djangoproject.com/en/4.2/
    $ python store.py -l rtdhtmlpage django ./docs.djangoproject.com/
    """
    def __init__(self, inputfile: Path, *args, **kwargs):
        kwargs["custom_html_tag"] = ("div", {"id": "docs-content"})
        super().__init__(inputfile, *args, **kwargs)

    def _my_clean_data(self, data: str) -> str:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(data, **self.bs_kwargs)

        # default tags
        html_tags = [
            ("div", {"role": "main"}),
            ("main", {"id": "main-content"}),
        ]

        if self.custom_html_tag is not None:
            html_tags.append(self.custom_html_tag)

        text = None

        # reversed order. check the custom one first
        for tag, attrs in html_tags[::-1]:
            text = soup.find(tag, attrs)
            # if found, break
            if text is not None:
                break

        if text is not None:
            title = "".join(t.text for t in text.find("h1") if t.name!="a")
            text = text.get_text()
        else:
            text = ""
            title = ""

        # trim empty lines
        text = "\n".join([t for t in text.split("\n") if t])

        return text, title

    def lazy_load(self) -> Iterator[Document]:
        """Load documents."""
        for p in self.file_path.rglob("*"):
            if p.is_dir():
                continue
            # FIXME: utf-8を指定したい
            # with open(p, encoding='utf-8', errors='ignore') as f:
            with open(p, encoding=self.encoding, errors=self.errors) as f:
                text, title = self._my_clean_data(f.read())

            if "docs.djangoproject.com" in p.parts and p.name == "index.html":
                # Djangoドキュメントではindex.htmlにアクセスすると404になる
                p = p.parent
                url = f"https://{str(p)}/"
            else:
                url = f"https://{str(p)}"

            metadata = {
                "title": title,
                "ctime": int(datetime.now().timestamp()),
                "user": "rtd",
                "type": "rtd",
                "url": url,
                "id": str(p),
            }
            # print(metadata)
            yield Document(page_content=text, metadata=metadata)


    def load(self) -> list[Document]:
        return list(self.lazy_load())
