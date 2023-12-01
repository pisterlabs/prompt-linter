from langchain.chains import RetrievalQA
import os
import argparse
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import VertexAI

def parse_arguments():
    parser = argparse.ArgumentParser(description='GPT: Ask questions to your documents without an internet connection, '
                                                 'using the power of LLMs.')
    parser.add_argument("--hide-source", "-S", action='store_true',
                        help='Use this flag to disable printing of source documents used for answers.')

    parser.add_argument("--mute-stream", "-M",
                        action='store_true',
                        help='Use this flag to disable the streaming StdOut callback for LLMs.')

    return parser.parse_args()


def retrieval_qa(db) :

    template = """
    Use the following context (delimited by <ctx></ctx>) and the chat history (delimited by <hs></hs>) to answer the question:
    ------
    <ctx>
    {context}
    </ctx>
    ------
    <hs>
    {history}
    </hs>
    ------
    {question}
    Answer:
    """
    prompt = PromptTemplate(
        input_variables=["history", "context", "question"],
        template=template,
    )

    target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))
    args = parse_arguments()
    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})

    llm = VertexAI(callbacks=callbacks, verbose=False, temperature=0, model_name='text-bison' )

    return RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        chain_type_kwargs={
                                            "verbose":True,
                                            "prompt":prompt,
                                            "memory": ConversationBufferMemory(
                                                memory_key = "history",
                                                input_key="question"),
                                            }
                                        )