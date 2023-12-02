import os
from langchain.vectorstores import Qdrant
import qdrant_client
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv("/home/kap/Desktop/pythonGPT/keys.env")

#codigo com função de query para ser utilizada pelo código principal


memoria_chat_DOC = [] #lista que guarda a memória do chat, dictionary de 2 keys: contexto do do e input do user 

openai_api_key = os.getenv('OPENAI_API_KEY') or 'OPENAI_API_KEY'
model_name = 'text-embedding-ada-002'

embed = OpenAIEmbeddings(  #setando o modelo de embeddings
    model=model_name,
    openai_api_key=openai_api_key
)

client = qdrant_client.QdrantClient(
     url=os.getenv("QDRANT_HOST"),
     api_key=os.getenv('QDRANT_API'), # setando o client qdrant com minha API de host URL
 )

doc_store = Qdrant(  #inicializando objeto de documento_store
    client=client, 
    collection_name=os.getenv("QD_COLLECTION_NAME"), 
    embeddings=embed,
)

def Pesquisa_Questoes(topic):  #função a ser chamada para recuperar a questão mais similar
  
     docs = doc_store.similarity_search(query=topic, k=1)  #gera 1 pagina de resultado na busca por documentos
     inputs = [{"context": doc.page_content, "topic": topic} for doc in docs] #extrai um dictionary dos resultados da busca nos docs
     Quest_Docs = inputs[0]['context'] 
        #extrai o texto em si do resultado
        
     if not("(Enem/") in Quest_Docs:
      Quest_Docs  =  " \n (Enem/" + Quest_Docs  #adicionar (Enem/ caso o input não tenha isso  

     
     memoria_chat_DOC.append({"context":Quest_Docs, "UserInput": topic})     #adiciona o resultado na lista de memória
     Quest_forma_memo=""        
     for doc in memoria_chat_DOC:        #formatar os conteúdos da lista de memória para dar como contexto pro LLM
        Quest_forma_memo += (doc["context"] + "\n")
        Quest_forma_memo += (doc["UserInput"] + "\n")
     i=1
     while ("(RE" in Quest_Docs):
       Quest_Docs = Quest_Docs[:-i]   #remove a resposta sempre correta do output para o usuário, configurar um pouco melhor isso
       i+=1
    
     retornoFinal = "\nQuestão do ENEM: "+ Quest_Docs 

     return memoria_chat_DOC, retornoFinal  #memoria_chat_DOC é o output para o contexto da IA, retorno fianl é para o usuário



# memo_chat, user_return =  Pesquisa_Questoes("me de uma questão do enem sobre a escravidão no brasil")
# print(type(memo_chat[0]))
# print(memo_chat[0]['context'])