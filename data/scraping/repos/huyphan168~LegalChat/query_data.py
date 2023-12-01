from langchain.callbacks.base import AsyncCallbackManager
from langchain.callbacks.tracers import LangChainTracer
from langchain.chains import ChatVectorDBChain
from langchain.chains.chat_vector_db.prompts import (CONDENSE_QUESTION_PROMPT,
                                                     QA_PROMPT)   
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.prompts.prompt import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.llms import OpenAI
from langchain.vectorstores.base import VectorStore
from chains import ChatEBAVectorDBChain, load_qaf_chain


eba_template = """As an AI law assistant specialized in regulation and policy of the European Banking Authority (EBA), you are here to assist user in answering your intensive knowledge question. 
you have been provided with relevant extracted parts of different related documents, user question, and its related meta-data. 
Let's work this question out in a step by step to be sure we have the right answer but do not write down the process just provide a answer with sources in seperated paragraph. 
If the question requests test code, you will provide a code block directly from the documents. If you don't know the answer, you will ask for more information. 
If the question is not related to EBA, you will politely inform you that you can only answer questions about EBA.
Please note that the relevant documents I have been given access to include: \n{context}, and I will be using these to provide you with accurate and relevant information.
{few_shot_samples} {question}
"""

def get_chain(
    vectorstore: VectorStore, sample_vectorstore: VectorStore, question_handler, stream_handler, tracing: bool = False
) -> ChatVectorDBChain:
    """Create a ChatVectorDBChain for question/answering."""
    manager = AsyncCallbackManager([])
    question_manager = AsyncCallbackManager([question_handler])
    stream_manager = AsyncCallbackManager([stream_handler])
    if tracing:
        tracer = LangChainTracer()
        tracer.load_default_session()
        manager.add_handler(tracer)
        question_manager.add_handler(tracer)
        stream_manager.add_handler(tracer)

    question_gen_llm = OpenAI(
        temperature=0,
        verbose=True,
        callback_manager=question_manager
    )
    streaming_llm = OpenAI(
        streaming=True,
        callback_manager=stream_manager,
        verbose=True,
        temperature=0
    )
    question_generator = LLMChain(
        llm=question_gen_llm, prompt=CONDENSE_QUESTION_PROMPT, callback_manager=manager
    )
    EBA_PROMPT = PromptTemplate(template=eba_template, input_variables=["question", "context", "few_shot_samples"])
    example_selector = SemanticSimilarityExampleSelector(
        vectorstore=sample_vectorstore,
        k=2
    )
    example_prompt = PromptTemplate(input_variables=["question", "answer"], template="Question: {question}\n{answer}")
    fewshots_prompt = FewShotPromptTemplate(
        example_selector=example_selector, 
        example_prompt=example_prompt, 
        suffix = "Question: {question}",
        input_variables=["question"]
    )

    doc_chain = load_qaf_chain(
        streaming_llm, fewshots_prompt=fewshots_prompt, prompt=EBA_PROMPT, callback_manager=manager
    )

    qa = ChatEBAVectorDBChain(
        vectorstore=vectorstore,
        combine_docs_chain=doc_chain,
        question_generator=question_generator,
        callback_manager=manager,
    )
    return qa
