from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders.csv_loader import CSVLoader
import openai
from dotenv import load_dotenv
from chat import GPT
import workGS
import os
#from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import TextLoader
from langchain.schema import Document
from loguru import logger

# При загрузке можно поменять указание источника на любую колонку в таблице (для примера на клонку topic), если не указывать то будет имя файла
#loader = CSVLoader(file_path="/content/Dataset_Lenta.ru_mini.csv", source_column="topic")

load_dotenv()
openai_key = os.environ.get("OPENAI_API_KEY")
openai.api_key =openai_key 

#sheet = workGS.Sheet('kgtaprojects-8706cc47a185.json','цены на дома 4.0 актуально ')
#gsText = sheet.get_gs_text()
def create_info_vector():
    # Инициализирум модель эмбеддингов
    tes = """<Информация о проетке
описание проекта, цена проекта, хочу построить дом 
/info
<Справочная информация
номера телефонов, ссылки
/spravka
<Подборка домов или проектов
несколько, проекты, список, топ, лучшие
/podborka
"""
    embeddings = OpenAIEmbeddings()
    gpt = GPT()
    GPT.set_key(os.getenv('KEY_AI'))

    source_chunks = []
    #splitter = CharacterTextSplitter(separator="\n", chunk_size=1524, chunk_overlap=0)
    splitter = CharacterTextSplitter(separator="<", chunk_size=10, chunk_overlap=0)

    for i,chunk in enumerate(splitter.split_text(tes)):
        #print(f'{chunk=}')
        # up = True if chunk.lower().find('одноэтажный') >= 0 else False 
        # up2 = True if chunk.lower().find('двухэтажный') >= 0 else False 
        # heig = ''
        # if up: heig='одноэтажный'
        # if up2: heig = 'двухэтажный'
        # meta = metadata = {"source": heig, "row": i}
        text = chunk.split('/')[0]
        meta= chunk.split('/')[1]
        source_chunks.append(Document(page_content=text, metadata={'type': meta}))
        
    #model_project = gpt.create_embedding(gsText)

    #db = FAISS.from_documents(data, embeddings)
    db = FAISS.from_documents(source_chunks, embeddings)
    return db
#filter=dict(source='Мир')
#text = """
#Спасибо за информацию. У нас есть возможность соблюсти сроки и завершить строительство вашего дома в течение полугода. Это важно учесть при планировании проекта. Вопрос: Вы рассматриваете возможность взять ипотеку для финансирования строительства дома?
#"""
#results_with_scores = db.similarity_search_with_score(text, filter=dict(source=text), k=3)
#results_with_scores = db.similarity_search_with_score("Турция отозвала своего посла из США", filter=dict(source='Мир'), k=4)
#results_with_scores = db._similarity_search_with_relevance_scores(text, k=3, filter=dict(source=text))
def answer_info(text, db):
    results_with_scores = db.similarity_search_with_score(text, k=3,)
    #content_info = ''
    #for doc, score in results_with_scores:
        #print(doc)
        #logger.warning(f"Content: {doc.page_content}, \nMetadata: {doc.metadata}, \nScore: {score}")
        #content_info = f"Content: {doc.page_content}, \nMetadata: {doc.metadata}, \nScore: {score}"
        #logger.warning('===========================')

    #return results_with_scores[0][0].page_content 
    return results_with_scores[0][0].metadata 


