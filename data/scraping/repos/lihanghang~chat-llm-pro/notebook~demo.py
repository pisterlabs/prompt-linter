'''
https://python.langchain.com/en/latest/modules/indexes/document_loaders/examples/pdf.html#using-pymupdf
Using PyPDF
Using Unstructured
Using PDFMiner
Using PyMuPDF
'''
import asyncio
import itertools
import json
import logging
import os

import dotenv
import openai
from kor import create_extraction_chain, extract_from_documents
from langchain import FAISS
from langchain.chat_models import AzureChatOpenAI
from langchain.document_loaders import PyMuPDFLoader
from langchain.document_loaders import UnstructuredWordDocumentLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter

from src.gpt import set_openai_key
from model.schema import medical_event_schema

# pdf
# loader = PyMuPDFLoader("docs/测试chatbot.pdf")
# pages = loader.load_and_split()
# print(pages)

# word
# loader = UnstructuredWordDocumentLoader(
#     "docs/test_pdf/杭州华测检测技术有限公司-杨总-20230328-10：30(1).docx",
#     mode='elements'
# )

# path = 'docs/test_pdf'
# files = os.listdir(path)  # 得到文件夹下的所有文件名称
# loaders = []
# for file in files:
#     loaders.append(UnstructuredWordDocumentLoader(
#         f"{path}/{file}",
#         mode='elements'
#     ))
#
# docs = []
# for loader in loaders:
#     docs.extend(loader.load())

# embedding_model_dict = {
#     "ernie-tiny": "nghuyong/ernie-3.0-nano-zh",
#     "ernie-base": "nghuyong/ernie-3.0-base-zh",
#     "text2vec": "GanymedeNil/text2vec-large-chinese",
# }
# embedding_obj = HuggingFaceEmbeddings(model_name=embedding_model_dict['text2vec'], )
# vector_db = FAISS.from_documents(docs, embedding=embedding_obj)
#
# query = "总共有几家公司？"
# docs = vector_db.similarity_search(query)
# print(docs)


# txt
from langchain.document_loaders import TextLoader

from web import azure_openai_key, api_version, api_base, api_type

# loader = TextLoader('docs/medical.txt')
# documents = loader.load()
# text_splitter = CharacterTextSplitter(separator='\n', chunk_size=200)
# docs = text_splitter.split_documents(documents)
#
# print(len(docs))
# print(docs)


config = dotenv.dotenv_values(".env")
openai.api_type = config["API_TYPE"]
openai.api_base = config["OPENAI_API_BASE"]
openai.api_version = config["OPENAI_API_VERSION"]
# fix 尽量用这种方式设置azure的可以，测试了下openai_api_key不起作用。
os.environ['OPENAI_API_KEY'] = config["AZURE_OPENAI_API_KEY"]

document = []
test_txt_path = 'docs/medical.txt'
with open(test_txt_path, 'r') as fb:
    lines = fb.readlines()

docs = [Document(page_content=line, metadata={'uid': idx}) for idx, line in enumerate(lines, 1)]
txt = '''该陪申特因股骨下端骨折于2027年6月5日用药注射用乳糖酸JQKA，每次0.5克，每日1次，静脉滴注，输液进行约20分钟，陪申特出现恶心，呕吐症状。<br>处理情况：立即停药，休息30分钟后，不适症状好转。<br>
'''
azure_llm = AzureChatOpenAI(
    deployment_name="gpt-35-turbo",
    temperature=0.9,
    max_tokens=2048
)

# define llm extract chain
extraction_chain = create_extraction_chain(azure_llm, medical_event_schema,
                                           encoder_or_encoder_class='json')
# extract from text
# result = extraction_chain.predict_and_parse(text=txt)['data']
# result_format = json.dumps(result, ensure_ascii=False, indent=4)
# print(result_format)


# extract from document
logging.info("Start extract")
extraction_results = asyncio.run(
    extract_from_documents(
        chain=extraction_chain,
        documents=docs,
        use_uid=True,
        max_concurrency=4,
        return_exceptions=True
    )
)
# print(extraction_results)
# ret = json.dumps(extraction_results, ensure_ascii=False, indent=4)
# validated_data = list(
#     itertools.chain.from_iterable(
#         extraction["validated_data"] for extraction in extraction_results
#     )
# )
# print(validated_data)


with open('docs/result.json', 'w') as fb:
    # fb.write(ret)
    ret = []
    for item in extraction_results:
        print(item['data'])
        try:
            # json_format = json.dumps(item['data'], ensure_ascii=False, indent=4)
            ret.append(item['data'])
            # fb.write(json_format)
        except Exception as e:
            logging.warning(e)
            # fb.write(item)
            ret.append(item['data'])
    fb.write(json.dumps(ret, ensure_ascii=False, indent=4))
