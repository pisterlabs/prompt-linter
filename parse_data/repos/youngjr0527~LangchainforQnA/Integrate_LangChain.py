#pip install unstructured  (공식문서에서는 !pip install unstructured > /dev/null 를 하라고 한다.)
#pip install markdown
from dotenv import load_dotenv
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from pathlib import Path
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO)

class CampusGuideBot:
    def __init__(self, db_path="./Chroma_DB", llm_model="gpt-3.5-turbo", temperature=0, template=None):
        self.db_path = db_path
        self.embedding_function = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model_name=llm_model, temperature=temperature)
        if template:
            self.template = template
        else:
            try:
                with open("default_template.txt", "r", encoding='utf-8') as f:
                    default_template = f.read()
                self.template = PromptTemplate.from_template(default_template)
            except FileNotFoundError:
                logging.warning("default_template.txt not found. Using hardcoded template.")
                self.template = None

    def ingest_documents(self, md_path=False): # Markdown 문서 저장
        logging.info(f"Ingesting documents from {md_path if md_path else 'all markdown files'}")
        try:
            if md_path:
                ps = Path(md_path)
                if not ps.exists():
                    logging.warning(f"The path {md_path} does not exist.")
                    return None
            else:
                ps = list(Path('.').glob("**/*.md"))
                if not ps:
                    logging.warning("No markdown files found.")
                    return None
        except Exception as e:
            logging.error(f"An error occurred while accessing the path: {e}")
            return
        for p in ps:
            logging.info(f"Processing file: {p}")
            loader = UnstructuredMarkdownLoader(str(p))
            pages = loader.load_and_split()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=20,
                length_function=len,
                is_separator_regex=False,
            )
            texts = text_splitter.split_documents(pages)

            Chroma.from_documents(texts, self.embedding_function, persist_directory=self.db_path)

    def generate_answer(self, question):
        logging.info(f"Generating answer for [question: {question}]")
        db_from_disk = Chroma(persist_directory=self.db_path, embedding_function=self.embedding_function)
        qa_chain = RetrievalQA.from_chain_type(self.llm, retriever=db_from_disk.as_retriever(), chain_type_kwargs={"prompt": self.template})
        result = qa_chain({"query": question})
        return result['result']


if __name__ == "__main__":
    bot = CampusGuideBot(db_path="./Chroma_DB")
    bot.ingest_documents()

    question = "시립대에는 편의시설 뭐 있어?"
    result = bot.generate_answer(question=question)
    print(result)
