import os
import qianfan
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

load_dotenv()
# 替换下列示例中参数，应用API Key替换your_ak，Secret Key替换your_sk
comp = qianfan.Completion(ak=os.getenv('QIANFAN_AK'), sk=os.getenv('QIANFAN_SK'))

class POContract(BaseModel):
    phone: str = Field(..., title="电话")
    fax: str = Field(..., title="传真")
    contactName: str = Field(..., title="联系人")
    contactPhone: str = Field(..., title="联系电话")
    contactEmail: str = Field(..., title="联系邮箱")

parser = PydanticOutputParser(pydantic_object=POContract)
# template = '''
#     You are a bot to extract assigned content from given content or file.
#     Answer the user query based on instruction and query content.
#     {format_instructions}\n{query}\n.
#     response only json format of extracted content according to schema, do not add anything else.
#     the response should starts with \'\'\'json and ends with \'\'\'.
#     '''
template = '''
You are a PDF parser expert to extract assigned content from given content or file.
Answer the user query based on instruction and query content.
Based on the following contract schema, extract the content from the given contract.

# input:

{query}

# now extract all tables from the given contract and reply the extracted content in json format.:

'''
prompt = PromptTemplate(
    template=template,
    input_variables=["query"],
    # partial_variables={"format_instructions": parser.get_format_instructions()},
)
loader = PyPDFLoader('docs/PurchaseOrder47648175.PDF')
text_splitter = CharacterTextSplitter(
    chunk_size = 5000,
    chunk_overlap  = 200,
    length_function = len,
    is_separator_regex = False,
)
pages = loader.load_and_split(text_splitter)
query = '\n'.join([page.page_content for page in pages])
_input = prompt.format_prompt(query=query)

resp = comp.do(model="ERNIE-Bot-turbo", prompt=_input.to_string())

print(resp.body.get('result'))