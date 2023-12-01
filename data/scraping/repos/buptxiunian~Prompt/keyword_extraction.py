from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM
from langchain.chat_models import ChatOpenAI
import os
import openai
from langchain.llms import OpenAI

os.environ["OPENAI_API_KEY"] = 'sk-H2nBsaKMbIGSh2uSl7QvKtGttNPGeOacPUeqy2fXJOr58AhP'
os.environ["OPENAI_API_BASE"] = 'https://api.chatanywhere.com.cn/v1'
llm = ChatOpenAI(temperature=0,
                 model_name='gpt-3.5-turbo',
                 openai_api_key='sk-H2nBsaKMbIGSh2uSl7QvKtGttNPGeOacPUeqy2fXJOr58AhP',
                 openai_api_base='https://api.chatanywhere.com.cn/v1')

with open("report.txt", encoding='utf-8') as f:
    report_2023 = f.read()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=100)

texts = text_splitter.split_text(report_2023)

docs = [Document(page_content=t) for t in texts]
prompt_template = """Please use five key words to summarize the passage
{text}
"""
prompt = PromptTemplate.from_template(prompt_template)

refine_template = (
    '''Your job is to use five key words to summarize the passage\n"
    We have provided an existing summary up to a certain point: {existing_answer}\n
    please answer the question in five words,not a sentence'''

)
refine_prompt = PromptTemplate(
    input_variables=["existing_answer"],
    template=refine_template,
)
chain = load_summarize_chain(llm, chain_type="refine",
                             return_intermediate_steps=True, question_prompt=prompt,
                             refine_prompt=refine_prompt,)
summ = chain({"input_documents": docs}, return_only_outputs=True)
print(summ["output_text"])
