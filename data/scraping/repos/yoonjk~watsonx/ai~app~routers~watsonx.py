# general
import requests, os, json
# fastapi
from fastapi import APIRouter, File, UploadFile, Form, Query
# ibm-generative-ai package
from genai.credentials import Credentials 
from genai.extensions.langchain import LangChainInterface
from genai.schemas import GenerateParams
from genai.model import Model
from genai.prompt_pattern import PromptPattern
# watson
from ibm_watson import LanguageTranslatorV3
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
""" 
	User Define
"""

# models
from schemas import Message, PromptMessage
from config import getConfig
from typing import Optional, List
from decimal import Decimal
api_key = os.getenv("GENAI_KEY", None) 
api_url = os.getenv("GENAI_URL", None)

print('api_key:', api_key)
creds = Credentials(api_key, api_endpoint=api_url)
router = APIRouter(prefix='/api/v1', tags = ["watsonx"])

@router.post('/qna', 
          description="prompt message",     
          responses={
            404: {"model": Message, "description": "The item was not found"},
            200: {
                "description": "Item requested by ID",
                "content": {
                    "application/json": {
                        "example": {"id": "bar", "value": "The bar tenders"}
                    }
            },
        },
    })
async def qna(message: str = Form(), uploadfile: Optional[UploadFile] = File(None)):
    if uploadfile:
        try:
            contents = await uploadfile.read()
            print(contents)
        finally:
            uploadfile.file.close()

    print(message)
    print("\n------------- Example (LangChain)-------------\n")

    #translate(body)
    params = GenerateParams(decoding_method="greedy", max_new_tokens=700)
    
    langchain_model = LangChainInterface(model="google/flan-ul2", params=params, credentials=creds)
    result = langchain_model('{}'.format(message))
    print("------------result:", result)
    transMessage = langTranslate(result, 'en', 'ko')
    voiceText = transMessage.get('translations')
    msg = voiceText[0] 
    print(msg.get('translation'))
    return msg
        # return result


@router.post('/rag')
async def gen_rag(message: str = Form(),
                  path: Optional[str] = 'rag',
                  decoding_method: Optional[str] = 'sample',
                  min_new_tokens: Optional[int] = 50,
                  max_new_tokens: Optional[int] = 200,
                  repetition_penalty: Optional[Decimal] = 1.0,
                  temperature: Optional[Decimal] = 0.9,
                  top_k: Optional[int] = 50                  
                  ):
    # PDF 문서들을 로드 
    loader = PyPDFDirectoryLoader(path)
    documents = loader.load()
    
    # text 문서 분할
    text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=100)
    split_docs = text_splitter.split_documents(documents)
    
    # Embeddedings
    embeddings = HuggingFaceEmbeddings()
    
    # Create the vectorized db
    db = FAISS.from_documents(split_docs, embeddings)
    docs = db.similarity_search(message)
    
    params = GenerateParams(
        decoding_method = decoding_method,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        repetition_penalty=repetition_penalty,
        top_k=top_k,
        temperature=temperature
    ).dict() 
    
    watsonx_model = LangChainInterface(model="google/flan-t5-xl", credentials=creds, params=params)
    chain = load_qa_chain(watsonx_model, chain_type="stuff")

    response = watsonx_model(message)
    print(response) 
    
    response = chain({'input_documents': docs, 'question': message}, return_only_outputs = True)
    
    return response


@router.post('/summary')
async def summarize(message: str = Form(), 
                    upload_file: Optional[UploadFile] = File(None),
                    lang : Optional[str] = 'kr',
                    llm_model: Optional[str] = 'bigscience/mt0-xxl',
                    decoding_method: Optional[str] = 'sample',
                    min_new_tokens: Optional[int] = 50,
                    max_new_tokens: Optional[int] = 200,
                    repetition_penalty: Optional[Decimal] = 1.0,
                    temperature: Optional[Decimal] = 0.9,
                    top_k: Optional[int] = 50
                    ):
    json_data = json.load(upload_file.file)
    content = 'new article'
    instruct = 'summary'
    
    if not message:
        message ='다음 본문은 뉴스 기사입니다. 본분을 새 문장으로 요약해주세요.'
        # ​The following document is a news article from Korea. Read the document and then write 3 sentences summary.
    if not lang:
        lang = 'kr'
    
    if lang == 'kr' or lang == 'ko':
        content = '본문'
        instruct = '요약'
        
    params = GenerateParams(
        decoding_method = decoding_method,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        repetition_penalty=repetition_penalty,
        top_k=top_k,
        temperature=temperature
    )
    # Prompt pattern
    prompt_str = """
        {0}
        
        {1}:
        {2}
        {3}:
    """.format(message, content, json_data['text'], instruct)  
    
    pattern = PromptPattern.from_str(prompt_str)
    model = Model(model=llm_model, params=params, credentials=creds)
    responses = model.generate_as_completed([str(pattern)])
    
    result = []
    
    for response in responses:
        print("Generated text:")
        result.append(response.generated_text)
        
    return {'result': '\n'.join(result)}
        

def langTranslate(message: str, source: str, target: str):
    apikey=os.environ['LANG_TRANSLATOR_APIKEY']
    url = os.environ['LANG_TRANSLATOR_URL']
    print(f'url:{url}')
    
    authenticator = IAMAuthenticator(apikey)
    language_translator = LanguageTranslatorV3(
        version='2018-05-01',
        authenticator=authenticator
    )

    language_translator.set_service_url(url)
    
    print(message)
    
    translation = language_translator.translate(
        text=message,
        source=source,
        target=target
        ).get_result()

    return translation 