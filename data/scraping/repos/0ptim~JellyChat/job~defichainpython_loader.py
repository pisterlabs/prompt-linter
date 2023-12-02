"""Loader that loads from DefichainPython."""
from typing import List, Dict

from langchain.docstore.document import Document
from langchain.document_loaders.web_base import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


class DefichainPythonLoader(WebBaseLoader):
    """Loader that loads from DefichainPython."""

    @staticmethod
    def replace_enter(text: str) -> str:
        while text.find("\n\n") != -1:
            text = text.replace("\n\n", "\n")
        return text

    @staticmethod
    def split_documents(docs: List[Document]):
        chunk_size = 800
        chunk_overlap = 50

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )
        return text_splitter.split_documents(docs)

    @staticmethod
    def append_metadata(docs: List[Document]):
        for doc in docs:
            header = "---\n"

            for key in doc.metadata:
                header += f"{key.capitalize()}: {doc.metadata.get(key)}\n"

            header += "---\n"
            doc.page_content = header + doc.page_content
        return docs

    def to_json(self) -> Dict:
        """
        The Content of a webpage will be compressed into a JSON format:

        {
            title: str
            source: str
            area: str
            tech: str
            page_description: str
            classes: [
                {
                    class_name: str
                    class_signature: str
                    class_description: str
                    class_methods: [
                        method_name: str
                        method_signature: str
                        method_description: str
                    ]
                }
            ]
            functions: [
                function_name: str
                function_signature: str
                function_description: str
            ]
        }
        """

        page_json = {}

        soup = self.scrape()

        """Title and Source"""
        title_tags = soup.find_all("h1")
        if title_tags:
            title = ", ".join([tag.get_text().replace("#", "") for tag in title_tags])
        else:
            print(self.web_path)
            raise ValueError("Title tag not found.")
        for tag in title_tags:
            tag.decompose()

        page_json.update({"title": title, "source": self.web_path})

        """Area"""
        area = self.web_path.split("/")[5:][:-1]
        area = area[0] if area else ""
        page_json.update({"area": area})

        """Tech"""
        tech = "/".join(self.web_path.split("/")[6:][:-1])
        page_json.update({"tech": tech})

        """Classes and Methods"""
        classes_list = []

        class_tags = soup.find_all("dl", class_="class")  # Find all classes
        for class_tag in class_tags:
            class_content = class_tag.find("dd")

            class_methods_list = []
            method_tags = class_content.find_all("dl", class_="method")  # Find all methods inside the class
            for method_tag in method_tags:
                method_signature = method_tag.find("dt").get_text().replace("#", "").replace("\n", "")
                method_name = method_signature.split("(")[0]
                method_description = DefichainPythonLoader.replace_enter(method_tag.find("dd").get_text())

                class_methods_list.append({"method_name": method_name,
                                           "method_signature": method_signature,
                                           "method_description": method_description})
                method_tag.decompose()

            class_signature = class_tag.find("dt").get_text().replace("#", "").replace("\n", "")
            class_name = class_signature.split("(")[0].split(".")[-1]
            class_description = DefichainPythonLoader.replace_enter(class_content.get_text())

            classes_list.append({"class_name": class_name,
                                 "class_signature": class_signature,
                                 "class_description": class_description,
                                 "class_methods": class_methods_list})

            class_tag.decompose()

        """Functions"""
        functions_list = []
        functions_tags = soup.find_all("dl", class_="function")  # Find all functions
        for function_tag in functions_tags:
            function_signature = function_tag.find("dt").get_text().replace("#", "").replace("\n", "")
            function_name = function_signature.split("(")[0].split(".")[-1]
            function_description = DefichainPythonLoader.replace_enter(function_tag.find("dd").get_text())

            functions_list.append({"function_name": function_name,
                                   "function_signature": function_signature,
                                   "function_description": function_description})
            function_tag.decompose()

        """Page Description"""
        article = soup.find("article")
        page_description = DefichainPythonLoader.replace_enter(article.get_text()).replace("#", "")

        page_json.update({"page_description": page_description})
        page_json.update({"classes": classes_list})
        page_json.update({"functions": functions_list})

        return page_json

    def load(self) -> List[Document]:
        """
        Load DefichainPython WebPage
        """

        page_json = self.to_json()
        documents = []

        base_metadata = {"title": page_json.get("title"),
                         "source": page_json.get("source"),
                         "area": page_json.get("area"),
                         "tech": page_json.get("tech")}

        """Page Content"""
        page_content_docs = DefichainPythonLoader.split_documents(
            [Document(page_content=page_json.get("page_description"), metadata=base_metadata)])
        #page_content_docs = DefichainPythonLoader.append_metadata(page_content_docs)
        documents.extend(page_content_docs)

        """Classes"""
        classes = page_json.get("classes")
        for class_ in classes:
            class_content = f'{class_.get("class_signature")}\n{class_.get("class_description")}'
            class_metadata = base_metadata.copy()
            class_metadata.update({"class_name": class_.get("class_name")})

            class_content_docs = DefichainPythonLoader.split_documents(
                [Document(page_content=class_content, metadata=class_metadata)])
            #class_content_docs = DefichainPythonLoader.append_metadata(class_content_docs)

            documents.extend(class_content_docs)

            """Methods"""
            methods = class_.get("class_methods")
            for method in methods:
                method_content = f'{method.get("method_signature")}\n{method.get("method_description")}'
                method_metadata = base_metadata.copy()
                method_metadata.update({"class_name": class_.get("class_name"),
                                        "method_name": method.get("method_name")})

                method_content_docs = DefichainPythonLoader.split_documents(
                    [Document(page_content=method_content, metadata=method_metadata)])
                #method_content_docs = DefichainPythonLoader.append_metadata(method_content_docs)
                documents.extend(method_content_docs)

        """Functions"""
        functions = page_json.get("functions")
        for function in functions:
            function_content = f'{function.get("function_signature")}\n{function.get("function_description")}'
            function_metadata = base_metadata.copy()
            function_metadata.update({"function_name": function.get("function_name")})

            function_content_docs = DefichainPythonLoader.split_documents(
                [Document(page_content=function_content, metadata=function_metadata)])
            #function_content_docs = DefichainPythonLoader.append_metadata(function_content_docs)
            documents.extend(function_content_docs)

        return documents


if __name__ == "__main__":
    url = "https://docs.defichain-python.de/build/html/api/node/index.html"
    loader = DefichainPythonLoader(url)
    docs = loader.load()

    for doc in docs:
        for key in doc.metadata:
            print(f"{key.capitalize()}: {doc.metadata.get(key)}")
        print("Content:", doc.page_content.split("---")[2].replace("\n", "\\n")[:100])
        print("----------------------------------------")
