from dotenv import load_dotenv
from langchain.chains import create_citation_fuzzy_match_chain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

load_dotenv()


def query_test(sources, query):
    documents = []
    for source in sources:
        documents.extend(TextLoader(source, encoding="utf-8").load())

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    question = "What did the author do during college?"
    context = """
    My name is Jason Liu, and I grew up in Toronto Canada but I was born in China.
    I went to an arts highschool but in university I studied Computational Mathematics and physics. 
    As part of coop I worked at many companies including Stitchfix, Facebook.
    I also started the Data Science club at the University of Waterloo and I was the president of the club for 2 years.
    """

    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")

    chain = create_citation_fuzzy_match_chain(llm)

    result = chain.run(question=question, context=context)

    return result


if __name__ == "__main__":
    query = "Where does Olga live?"
    sources = ["data/file1.txt", "data/file2.txt", "data/state_of_the_union_full.txt"]

    query_test(sources, query)
