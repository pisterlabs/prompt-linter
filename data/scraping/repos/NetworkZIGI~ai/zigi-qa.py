import requests
import openai  
import envinfo
from azure.core.credentials import AzureKeyCredential  
from azure.search.documents import SearchClient  
from tenacity import retry, wait_random_exponential, stop_after_attempt 

class zigi_OpenAPI:

    def __init__(self):
        ### Cognitive Service 환경 변수 ###
        global service_endpoint
        global key
        global credential

        service_endpoint = envinfo.cg_endpoint
        key = envinfo.cg_key
        credential = AzureKeyCredential(key)

        ### Azure OpenAI 환경 변수 ###
        openai.api_type = envinfo.openai_api_type  
        openai.api_key = envinfo.openai_api_key
        openai.api_base = envinfo.openai_api_base 
        openai.api_version = envinfo.openai_api_version 




    def init_Var(self):
        global index_name, deployment_name, rname
        global prompt
        
        index_name = "zigi-index" 


        prompt ="""You are an assistant who answers by referring to the given "Context". Please be as professional and detailed as possible. If you don't know the answer, you should answer that you don't know, without generating an incorrect answer. All answers are in Korean only.  Leave the string 'end answer' at the end of the answer.
                Context : {result_context}
                Question: {question}
                Answer in Only Korean: """
 
        global search_client
        search_client = SearchClient(service_endpoint, index_name, AzureKeyCredential(key))  


    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def generate_embeddings(self, text):
        response = openai.Embedding.create(
            input=text, engine="text-embedding-ada-002")
        embeddings = response['data'][0]['embedding']
        return embeddings

    def search_cognitive(self,query):        
        vector = self.generate_embeddings(text=query)

        vector_only = True
        if vector_only:
            search_text = None
        else:
            search_text = query
        
        results = search_client.search(search_text=search_text, 
                                       vector=vector, 
                                       vector_fields="content_vector", 
                                       top_k=3,
                                       select=["source","Content"])

        context = [] 
        result_context = []      
        global ref_documents
        ref_documents = []      
        
        for result in results:  
            ref_document = []
            ref_document.append(f"Source Document: {result['source']}")            
            ref_document.append(f"Score: {result['@search.score']}")  
            ref_document.append(f"Content : {result['Content']}")
            ref_documents.append(ref_document)
            context.append(str(result['Content']))
            

        result_context = '\n\n##\n\n'.join(context)
        return result_context
    
    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def QA_ChatGPT(self, query_result, question):
        deployment_name = "gpt-35-turbo"
        qa_str = prompt.format(result_context = query_result, question = question)
        url = f"""https://zigi-openai.openai.azure.com/openai/deployments/{deployment_name}/completions?api-version={openai.api_version}"""
        headers = {
            "api-key": "API 키정보",
            "Content-Type": "application/json",
        }

        data = {"prompt":qa_str, 
                "max_tokens" : 1000,     # (int) deafult 16
                'stop' : "end answer",
                'temperature' : 0}

        response = requests.post(url=url,headers=headers,json=data)
        print("질문 : " + question)

        if response.status_code == 200:
            print("답변 : " + response.json()['choices'][0]['text'])            
            return response.json()['choices'][0]['text']          
        else:
            errorcode = "Error Code(" + str(response.status_code) + ") : ChatGPT의 응답 값을 정상적으로 수신하지 못하였습니다."
            print(errorcode)
            return errorcode

if __name__ == "__main__":
    zigi = zigi_OpenAPI()
    zigi.init_Var()
    question = "질문_내용"
    query_result = zigi.search_cognitive(query=question)
    zigi.QA_ChatGPT(query_result=query_result, question=question)
