"""Loader that loads Obsidian vault and splits content with the provided splitter."""
import re
from pathlib import Path
from typing import List

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


class ObsidianSplittingLoader(BaseLoader):
    """Loader that loads Obsidian files from disk."""

    FRONT_MATTER_REGEX = re.compile(r"^---\n(.*?)\n---\n", re.MULTILINE | re.DOTALL)

    def __init__(
        self, path: str, splitter=RecursiveCharacterTextSplitter(), encoding: str = "UTF-8", collect_metadata: bool = True
    ):
        """Initialize with path."""
        self.file_path = path
        self.splitter = splitter
        self.encoding = encoding
        self.collect_metadata = collect_metadata

    def _parse_front_matter(self, content: str) -> dict:
        """Parse front matter metadata from the content and return it as a dict."""
        if not self.collect_metadata:
            return {}
        match = self.FRONT_MATTER_REGEX.search(content)
        front_matter = {}
        if match:
            lines = match.group(1).split("\n")
            for line in lines:
                if ":" in line:
                    key, value = line.split(":", 1)
                    front_matter[key.strip()] = value.strip()
                else:
                    # Skip lines without a colon
                    continue
        return front_matter

    def _remove_front_matter(self, content: str) -> str:
        """Remove front matter metadata from the given content."""
        if not self.collect_metadata:
            return content
        return self.FRONT_MATTER_REGEX.sub("", content)

    def load(self) -> List[Document]:
        """Load documents."""
        ps = list(Path(self.file_path).glob("**/*.md"))
        docs = []
        for p in ps:
            with open(p, encoding=self.encoding) as f:
                text = f.read()

            front_matter = self._parse_front_matter(text)
            text = self._remove_front_matter(text)
            chunks = self.splitter.split_text(text)
            for i, chunk in enumerate(chunks):
                metadata = {
                    "source": str(p.name) + f"#{i}",
                    "path": str(p),
                    "created": p.stat().st_ctime,
                    "last_modified": p.stat().st_mtime,
                    "last_accessed": p.stat().st_atime,
                    **front_matter,
                }
                docs.append(Document(page_content=chunk, metadata=metadata))

        return docs
