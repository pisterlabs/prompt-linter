from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import pinecone
from lib import constants
import click
import textwrap
import logging

log = logging.getLogger(__name__)


def bonus():  # pragma: no cover
    pinecone.init(
        api_key=constants.PINECONE_API_KEY,  # find at app.pinecone.io
        environment=constants.PINECONE_API_ENV,  # next to api key in console
    )
    embeddings = OpenAIEmbeddings(openai_api_key=constants.OPENAI_API_KEY)
    llm = OpenAI(temperature=0, openai_api_key=constants.OPENAI_API_KEY)

    index_name = constants.INDEX_NAME

    click.secho(
        'Implementing Semantic search usin pinecone+langchain\n'
        'ask me Anything about our scientients scrapped',
        fg='yellow')

    while True:
        doc_search = Pinecone.from_existing_index(index_name, embeddings)
        query = click.prompt('query > ')

        docs = doc_search.similarity_search(query)

        titles = set()
        for i in docs:
            titles.add(i.metadata["title"])

        if titles:
            print(f'found information in {",".join(titles)} doc')

        chain = load_qa_chain(llm, chain_type="stuff")
        answer = chain.run(input_documents=docs, question=query)
        click.secho('\n'.join(textwrap.wrap(f'Answer: {answer}')), fg='green')


if __name__ == "__main__":
    bonus()
