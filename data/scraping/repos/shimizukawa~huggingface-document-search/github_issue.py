import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterator

from dateutil.parser import parse
from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader

from models import GithubIssue


def date_to_int(dt_str: str) -> int:
    dt = parse(dt_str)
    return int(dt.timestamp())


def get_contents(inputfile: Path) -> Iterator[tuple[GithubIssue, str]]:
    with inputfile.open("r") as f:
        obj = [json.loads(line) for line in f]
    for data in obj:
        title = data["title"]
        body = data["body"]
        issue = GithubIssue(
            id=data["number"],
            title=title,
            ctime=date_to_int(data["created_at"]),
            user=data["user.login"],
            url=data["html_url"],
            labels=data["labels_"],
        )
        text = title
        if body:
            text += "\n\n" + body
        yield issue, text
        comments = data["comments_"]
        for comment in comments:
            issue = GithubIssue(
                id=comment["id"],
                title=data["title"],
                ctime=date_to_int(comment["created_at"]),
                user=comment["user.login"],
                url=comment["html_url"],
                labels=data["labels_"],
                type="issue_comment",
            )
            yield issue, comment["body"]


class GithubIssueLoader(BaseLoader):
    def __init__(self, inputfile: Path):
        self.inputfile = inputfile

    def lazy_load(self) -> Iterator[Document]:
        for issue, text in get_contents(self.inputfile):
            metadata = asdict(issue)
            yield Document(page_content=text, metadata=metadata)

    def load(self) -> list[Document]:
        return list(self.lazy_load())
