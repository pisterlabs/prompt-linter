import asyncio

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain import PromptTemplate
from langchain.memory import ConversationBufferMemory

from lanchainlit import load_faiss_vector_db, load_documents, split_documents_to_vector_db, ChatGLMService

print("start")
llm_model_name = 'Qwen/Qwen-7B-Chat'

# 请注意：分词器默认行为已更改为默认关闭特殊token攻击防护。
tokenizer = AutoTokenizer.from_pretrained(llm_model_name, trust_remote_code=True)
print("tokenizer")
# bf16精度，A100、H100、RTX3060、RTX3070
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="auto", trust_remote_code=True, bf16=True).eval()
# 打开fp16精度，V100、P100、T4等显卡建议启用以节省显存
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="auto", trust_remote_code=True, fp16=True).eval()
# 使用CPU进行推理，需要约32GB内存
model = AutoModelForCausalLM.from_pretrained(llm_model_name, device_map="cpu", trust_remote_code=True).eval()
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat",
#                                              device_map="auto",
#                                              bf16=True,
#                                              trust_remote_code=True).eval()

print("model")
# 可指定不同的生成长度、top_p等相关超参
# model.generation_config = GenerationConfig.from_pretrained(llm_model_name, trust_remote_code=True)

# -------------------------
default_custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, please just sat that you don't know the answer,
don't try to make up an answer.

Context: {context}
Question: {question}

Only returns the helpful answer below and nothing else.
Helpful answer: 
"""


def set_custom_prompt(prompt_template: str = None):
    prompt = PromptTemplate(template=prompt_template, input_variables=[
        'context', 'question'])
    return prompt

#     if prompt_template is None:
#         prompt_template = """Use the following context (delimited by <ctx></ctx>) and the chat history (delimited by <hs></hs>) to answer the question:
# ------
# <ctx>
# {context}
# </ctx>
# ------
# <hs>
# {history}
# </hs>
# ------
# {question}
# Answer:
# """
#     prompt = PromptTemplate(
#         template=prompt_template,
#         input_variables=["history", "context", "question"],
#     )
#     return prompt


def create_qa_pipe():
    db_faiss_path = 'models/db_faiss'
    documents = load_documents('data')
    split_documents_to_vector_db(documents, chunk_size=500, db_faiss_path=db_faiss_path)

    print("create qa pipe")
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512,
        temperature=0.1,
        pad_token_id=tokenizer.eos_token_id,
        top_p=0.95,
        repetition_penalty=1.2
    )
    print("local_llm")
    local_llm = HuggingFacePipeline(pipeline=pipe)
    vector_store = load_faiss_vector_db(db_faiss_path)
    print("FAISS db")
    prompt_template = """Answer concisely and professionally based on the following provided information. 
    If unable to derive an answer from it, please say 'Unable to answer this question based on the provided information' 
    or 'Insufficient relevant information provided.' 
    Do not add fabricated content to the answer. Please respond in English.
    Provided Context: {context}
    Question: {question}"""
    chat_prompt = set_custom_prompt(prompt_template)
    top_k = 3
    llm_service = ChatGLMService()
    llm_service.model = model
    # llm_service.load_model(model_name_or_path=llm_model_name)
    # knowledge_chain = RetrievalQA.from_llm(
    #     llm=llm_service,
    #     retriever=vector_store.as_retriever(search_kwargs={"k": top_k}),
    #     prompt=chat_prompt)
    # knowledge_chain.combine_documents_chain.document_prompt = PromptTemplate(
    #     input_variables=["page_content"], template="{page_content}")
    # knowledge_chain.return_source_documents = True
    # result = knowledge_chain({"query": query})

    # qa_prompt = set_custom_prompt(default_custom_prompt_template)
    knowledge_chain = RetrievalQA.from_chain_type(llm=local_llm,
                                                  chain_type="stuff",
                                                  retriever=vector_store.as_retriever(search_kwargs={'k': top_k}),
                                                  return_source_documents=True,
                                                  chain_type_kwargs={
                                                      'prompt': chat_prompt,
                                                      "verbose": False,
                                                      #"memory": ConversationBufferMemory(memory_key="history", input_key="question")
                                                  })
    # qa_chain_fn = knowledge_chain(local_llm, qa_prompt, vector_store)
    print("ready")
    return knowledge_chain


async def bot_doc(qa_chain_fn, query):
    res = await qa_chain_fn.acall(query)
    answer = res["result"]
    sources = res["source_documents"]
    if sources:
        answer += f"\nSources:" + str(sources[0].metadata['source'])
    else:
        answer += f"\nNo Sources Found"
    print(f"{answer}")


def main_chat():
    history = None
    while True:
        user_input = input("query: ")
        response, history = model.chat(tokenizer, user_input, history=history)
        print(response)


async def main_doc():
    while True:
        user_input = input("query: ")
        qa_chain_fn = create_qa_pipe()
        response = await bot_doc(qa_chain_fn, user_input)
        print(response)


if __name__ == '__main__':
    asyncio.run(main_doc())
    # main_chat()
