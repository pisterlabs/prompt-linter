"""
Copyright 2023 by Sergei Belousov

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
# PDF parser
from pypdf import PdfReader
# openai
import openai
# langchain
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document


class SmartPDF:
    """Smart PDF document.

    Args:
        path (str): Path to the PDF document.
        openai_api_key (str): OpenAI API key.
    """
    def __init__(
            self,
            path: str,
            openai_api_key: str
    ):
        # params
        self.path = path
        self.openai_api_key = openai_api_key
        # utils
        self.llm = OpenAI(
            temperature=0,
            openai_api_key=openai_api_key,
        )
        self.knowledge_base, self.docs = self._preare_doc(path)

    def __repr__(self):
        """Representation of the SmartPDF object."""
        return f"SmartPDF(path={self.path})"

    def smart_search(self, request: str):
        """Smart search in the PDF document.

        Args:
            request (str): Request for search.

        Returns:
            str: Answer on the request.
        """
        docs = self.knowledge_base.similarity_search(request)
        chain = load_qa_chain(self.llm, chain_type="stuff")
        response = chain.run(input_documents=docs, question=request)
        return response

    def summary(self):
        """Summary of the PDF document.

        Returns:
            str: Summary of the PDF document.
        """
        chain = load_summarize_chain(self.llm, chain_type="map_reduce")
        summarized_docs = chain.run(self.docs)
        return summarized_docs

    def _preare_doc(self, path: str):
        """Prepare document for smart search.

        Args:
            path (str): Path to the PDF document.

        Returns:
            knowledge_base (FAISS): Knowledge base for smart search.
            docs (list): List of documents.
        """
        pdf_reader = PdfReader(path)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separator="\n",
            length_function=len
        )
        chunks = splitter.split_text(text)
        docs = [Document(page_content=t) for t in chunks[:3]]
        embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
        knowledge_base = FAISS.from_texts(chunks, embeddings)
        return knowledge_base, docs


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="Path to the PDF document.")
    parser.add_argument("--openai-api-key", type=str, help="OpenAI API key.")
    args = parser.parse_args()

    smart_doc = SmartPDF(
        path=args.path,
        openai_api_key=args.openai_api_key
    )
    print(smart_doc)
    print(f"Summary: {smart_doc.summary()}")
    print("Press q! to exit")
    while True:
        request = input("> ")
        if request == "q!":
            break
        print(smart_doc.smart_search(request))
