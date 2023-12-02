from langchain.document_loaders import WebBaseLoader
from langchain.docstore.document import Document
from typing import Any, Dict, List, Optional, Union

def _build_metadata(soup: Any, url: str) -> dict:
    """Build metadata from BeautifulSoup output."""
    metadata = {"source": url}
    if title := soup.find("title"):
        metadata["title"] = title.get_text()
    if description := soup.find("meta", attrs={"name": "description"}):
        metadata["description"] = description.get("content", None)
    if html := soup.find("html"):
        metadata["language"] = html.get("lang", None)
    return metadata

class SherherdPageLoader(WebBaseLoader):
    def load(self) -> List[Document]:
        """Load text from the url(s) in web_path."""
        docs = []
        for path in self.web_paths:
            soup = self._scrape(path)
            # kill all script and style elements
            for script in soup(["script", "style"]):
                script.extract()    # rip it out
            text = soup.get_text()
            metadata = _build_metadata(soup, path)
            docs.append(Document(page_content=text, metadata=metadata))

        return docs

    def aload(self) -> List[Document]:
        """Load text from the urls in web_path async into Documents."""

        results = self.scrape_all(self.web_paths)
        docs = []
        for i in range(len(results)):
            soup = results[i]
              # kill all script and style elements
            for script in soup(["script", "style"]):
                script.extract()    # rip it out

            text = soup.get_text()
            metadata = _build_metadata(soup, self.web_paths[i])
            docs.append(Document(page_content=text, metadata=metadata))

        return docs
