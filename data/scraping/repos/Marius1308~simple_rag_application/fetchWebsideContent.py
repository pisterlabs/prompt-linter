import requests
from langchain.document_loaders import BSHTMLLoader
from pageLinks import root, pages

from html.parser import HTMLParser


class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        # Only parse the 'anchor' tag.
        if tag == "a":
            # Check the list of defined attributes.
            for name, value in attrs:
                # If href is defined, print it.
                if name == "href":
                    if value.startswith("/docs"):
                        print(value)


paths = []

parser = MyHTMLParser()
for page in pages:
    response = requests.get(f"{root}/{page}")
    pageName = page.replace("/", "-")
    pagePath = f"./langchainPages/html/{pageName}.html"

    text_file = open(pagePath, "w")
    parser.feed(response.content.decode("utf-8"))
    text_file.write(response.content.decode("utf-8"))

    text_file.close()
    paths.append(pagePath)


for path in paths:
    loader = BSHTMLLoader(path)
    data = loader.load()
    newPath = path.replace("html", "txt")
    text_file = open(newPath, "w")

    file_text = ""
    for index, page in enumerate(data):
        pageText = page.page_content
        while "\n\n" in pageText:
            pageText = pageText.replace("\n\n", "\n")
        while pageText[-1] == "\n":
            pageText = pageText[:-1]
        while pageText[0] == "\n":
            pageText = pageText[1:]
        prefix = "\n\n" if index > 0 else ""
        file_text = prefix + pageText
    text_file.write(file_text)
    text_file.close()
