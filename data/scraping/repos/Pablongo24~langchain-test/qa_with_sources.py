from dotenv import load_dotenv  # noqa
import requests
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.docstore.document import Document
from langchain.llms import OpenAI

load_dotenv()


class QAWithSourcesChain:
    def __init__(self):
        self.documents = []
        self.sources = []
        self.search_index = None
        self.chain = load_qa_with_sources_chain(OpenAI(temperature=0), chain_type="map_reduce")

    def add_wiki_document(self, title: str, first_paragraph_only: bool = False):
        self.documents.extend(self.get_wiki_data(title=title, first_paragraph_only=first_paragraph_only))

    def add_wiki_sources(self, sources: list):
        self.sources.extend(sources)

    @staticmethod
    def get_wiki_data(title: str, first_paragraph_only: bool = False):
        url = f"https://en.wikipedia.org/w/api.php?format=json&action=query&prop=extracts&explaintext=1&titles={title}"
        if first_paragraph_only:
            url += "&exintro=1"
        data = requests.get(url).json()
        return Document(
            page_content=list(data["query"]["pages"].values())[0]["extract"],
            metadata={"source": f"https://en.wikipedia.org/wiki/{title}"},
        )

    def build_search_index(self):
        source_chunks = []
        splitter = CharacterTextSplitter(separator=" ", chunk_size=1024, chunk_overlap=0)
        for document in self.documents:
            for chunk in splitter.split_text(document.page_content):
                source_chunks.append(Document(page_content=chunk, metadata=document.metadata))

        self.search_index = FAISS.from_documents(source_chunks, OpenAIEmbeddings())

    def print_answer(self, sources: list, question: str):
        self.add_wiki_sources(sources)
        for source in sources:
            self.documents.append(self.get_wiki_data(source, first_paragraph_only=False))

        self.build_search_index()

        print(f"QUESTION:\n{question}\nANSWER:\n")
        print(
            self.chain(
                {
                    "input_documents": self.search_index.similarity_search(question, k=4),
                    "question": question,
                },
                return_only_outputs=True,
            )["output_text"].strip()
        )


if __name__ == "__main__":
    import pickle
    wiki_sources = ["Brad Pitt"]
    qa_chain = QAWithSourcesChain()
    test_questions = [
        "Where does Brad Pitt live?",
        "What are the main differences between Linux and Windows?"
    ]
    test_q_idx = 0

    qa_chain.print_answer(sources=wiki_sources, question=test_questions[test_q_idx])
    breakpoint()
