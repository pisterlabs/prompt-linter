from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.llms import LlamaCpp
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
from langchain import PromptTemplate, LLMChain
import json


model_path = '../models/Llama-2-13B-chat-GGUF/llama-2-13b-chat.Q4_K_M.gguf'

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
llm = LlamaCpp(
    model_path=model_path,
    n_ctx=5000,
    n_gpu_layers=40,
    n_threads=15,
    n_batch=512,
    f16_kv=True,
    callback_manager=callback_manager,
    verbose=True,
    streaming=True
)

stream = llm(
    "Question: What are the names of the planets in the solar system? Answer: ",
    max_tokens=48,
    stop=["Q:", "\n"],
    # stream=True,
)
print(f"{stream=}")
for output in stream:
    print(json.dumps(output, indent=2))

# memory = ConversationSummaryMemory(llm=llm, memory_key="chat_history", return_messages=True)
# qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)


# B_INST, E_INST = "[INST]", "[/INST]"
# B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
# system_prompt = """You are a helpful assistant, you will use the provided context to answer user questions.
# Read the given context before answering questions and think step by step. If you can not answer a user question based on
# the provided context, inform the user. Do not use any other information for answering user"""
# instruction = """
# Context: {context}
# User: {question}"""
#
#
# def prompt_format(instruction=instruction, system_prompt=system_prompt):
#     SYSTEM_PROMPT = B_SYS + system_prompt + E_SYS
#     prompt_template = B_INST + SYSTEM_PROMPT + instruction + E_INST
#     return prompt_template
#
#
# template = prompt_format()
# QA_CHAIN_PROMPT = PromptTemplate(
#     input_variables=["context", "question"],
#     template=template
# )


# python_splitter = RecursiveCharacterTextSplitter.from_language(
#     language=Language.PYTHON,
#     chunk_size=2000,
#     chunk_overlap=200
# )
# texts = python_splitter.split_documents(documents)
# db = Chroma.from_documents(texts, embedding)
# retriever = db.as_retriever(
#     search_type="mmr",
#     search_kwargs={"k": 8},
# )
# docs = retriever.get_relevant_documents(question)

# chain = load_qa_chain(
#     llm,
#     chain_type="stuff",
#     prompt=QA_CHAIN_PROMPT
# )
#

# question = "use C# write hello"
# result = chain({
#     "input_documents": docs,
#     "question": question
#     },
#     return_only_outputs=True
# )

# print(f"{result=}")
