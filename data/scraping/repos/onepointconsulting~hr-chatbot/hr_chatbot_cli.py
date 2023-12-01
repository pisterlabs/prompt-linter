from langchain.chains import RetrievalQAWithSourcesChain
from prompt_toolkit import HTML, prompt, PromptSession
from prompt_toolkit.history import FileHistory

from chain_factory import create_retrieval_chain, load_embeddinges

from log_init import logger

import sys


def init_chain():
    humour = False
    if len(sys.argv) > 1:
        if sys.argv[1] == "humor":
            humour = True
            logger.warning("Humor flag activated")
    session = PromptSession(history=FileHistory(".agent-history-file"))
    docsearch, documents = load_embeddinges()
    chain: RetrievalQAWithSourcesChain = create_retrieval_chain(
        docsearch, humour=humour
    )
    return session, chain


if __name__ == "__main__":
    session, chain = init_chain()

    while True:
        question = session.prompt(
            HTML("<b>Type <u>Your question</u></b>  ('q' to exit): ")
        )
        if question.lower() in ["q", "exit", "quit"]:
            break
        response = chain({"question": question})
        logger.info(f"Answer: {response['answer']}")
        logger.info(f"Sources: {response['sources']}")
