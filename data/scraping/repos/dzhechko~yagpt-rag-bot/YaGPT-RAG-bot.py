# создаем простое streamlit приложение для работы с вашими pdf-файлами при помощи YaGPT

import streamlit as st
import tempfile
import os
from opensearchpy import OpenSearch
from yandex_chain import YandexEmbeddings
from yandex_chain import YandexLLM


from langchain.prompts import PromptTemplate
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import OpenSearchVectorSearch

from langchain.chains import RetrievalQA

from streamlit_chat import message

# from dotenv import load_dotenv

ROOT_DIRECTORY = "."
MDB_OS_CA = f"{ROOT_DIRECTORY}/.opensearch/root.crt"

# использовать системные переменные из облака streamlit (secrets)
# yagpt_api_key = st.secrets["yagpt_api_key"]
# yagpt_folder_id = st.secrets["yagpt_folder_id"]
# yagpt_api_id = st.secrets["yagpt_api_id"]
# mdb_os_pwd = st.secrets["mdb_os_pwd"]
# mdb_os_hosts = st.secrets["mdb_os_hosts"].split(",")
# mdb_os_index_name = st.secrets["mdb_os_index_name"]

# MDB_OS_CA = st.secrets["mdb_os_ca"] # 

def ingest_docs(temp_dir: str = tempfile.gettempdir()):
    """
    Инъекция ваших pdf файлов в MBD Opensearch
    """
    try:
        # выдать ошибку, если каких-то переменных не хватает
        if not yagpt_api_key or not yagpt_folder_id or not yagpt_api_id or not mdb_os_pwd or not mdb_os_hosts or not mdb_os_index_name:
            raise ValueError(
                "Пожалуйста укажите необходимый набор переменных окружения")

        # загрузить PDF файлы из временной директории
        loader = DirectoryLoader(
            temp_dir, glob="**/*.pdf", loader_cls=PyPDFLoader, recursive=True
        )
        documents = loader.load()

        # разбиваем документы на блоки
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        documents = text_splitter.split_documents(documents)
        print(len(documents))

        # подключаемся к базе данных MDB Opensearch, используя наши ключи (проверка подключения)
        conn = OpenSearch(
            mdb_os_hosts,
            http_auth=('admin', mdb_os_pwd),
            use_ssl=True,
            verify_certs=False,
            ca_certs=MDB_OS_CA)
        # для включения проверки MDB сертификата используйте verify_certs=True, также надо будет загрузить сертификат используя инструкцию по ссылке 
        # https://cloud.yandex.ru/docs/managed-opensearch/operations/connect 
        # и положить его в папку .opensearch/root.crt
        
        # инициируем процедуру превращения блоков текста в Embeddings через YaGPT Embeddings API, используя API ключ доступа
        embeddings = YandexEmbeddings(folder_id=yagpt_folder_id, api_key=yagpt_api_key)

        # добавляем "документы" (embeddings) в векторную базу данных Opensearch
        docsearch = OpenSearchVectorSearch.from_documents(
            documents,
            embeddings,
            opensearch_url=mdb_os_hosts,
            http_auth=("admin", mdb_os_pwd),
            use_ssl = True,
            verify_certs = False,
            ca_certs = MDB_OS_CA,
            engine = 'lucene',
            index_name = mdb_os_index_name,
            bulk_size=1000000
        )
    # bulk_size - это максимальное количество embeddings, которое можно будет поместить в индекс

    except Exception as e:
        st.error(f"Возникла ошибка при добавлении ваших файлов: {str(e)}")


# это основная функция, которая запускает приложение streamlit
def main():
    # Загрузка логотипа компании
    logo_image = './images/logo.png'  # Путь к изображению логотипа

    # # Отображение логотипа в основной части приложения
    from PIL import Image
    # Загрузка логотипа
    logo = Image.open(logo_image)
    # Изменение размера логотипа
    resized_logo = logo.resize((100, 100))
    # Отображаем лого измененного небольшого размера
    st.image(resized_logo)
    # Указываем название и заголовок Streamlit приложения
    st.title('YaGPT-чат с вашими PDF файлами')
    st.warning('Загружайте свои PDF-файлы и задавайте вопросы по ним. Если вы уже загрузили свои файлы, то ***обязательно*** удалите их из списка загруженных и переходите к чату ниже.')

    # вводить все credentials в графическом интерфейсе слева
    # Sidebar contents
    with st.sidebar:
        st.title('\U0001F917\U0001F4ACИИ-помощник')
        st.markdown('''
        ## О программе
        Данный YaGPT-помощник реализует [Retrieval-Augmented Generation (RAG)](https://github.com/yandex-cloud-examples/yc-yandexgpt-qa-bot-for-docs/blob/main/README.md) подход
        и использует следующие компоненты:
        - [Yandex GPT](https://cloud.yandex.ru/services/yandexgpt)
        - [Yandex GPT for Langchain](https://pypi.org/project/yandex-chain/)
        - [YC MDB Opensearch](https://cloud.yandex.ru/docs/managed-opensearch/)
        - [Streamlit](https://streamlit.io/)
        - [LangChain](https://python.langchain.com/)
        ''')

    global  yagpt_folder_id, yagpt_api_id, yagpt_api_key, mdb_os_ca, mdb_os_pwd, mdb_os_hosts, mdb_os_index_name    
    yagpt_folder_id = st.sidebar.text_input("YAGPT_FOLDER_ID", type='password')
    yagpt_api_id = st.sidebar.text_input("YAGPT_API_ID", type='password')
    yagpt_api_key = st.sidebar.text_input("YAGPT_API_KEY", type='password')
    mdb_os_ca = MDB_OS_CA
    mdb_os_pwd = st.sidebar.text_input("MDB_OpenSearch_PASSWORD", type='password')
    mdb_os_hosts = st.sidebar.text_input("MDB_OpenSearch_HOSTS через 'запятую' ", type='password').split(",")
    mdb_os_index_name = st.sidebar.text_input("MDB_OpenSearch_INDEX_NAME")

    # Параметры chunk_size и chunk_overlap
    global chunk_size, chunk_overlap
    chunk_size = st.sidebar.slider("Выберите размер текстового 'окна' разметки документов в символах", 0, 2000, 1000)
    chunk_overlap = st.sidebar.slider("Выберите размер блока перекрытия в символах", 0, 400, 100)

    # Выводим предупреждение, если пользователь не указал свои учетные данные
    if not yagpt_api_key or not yagpt_folder_id or not yagpt_api_id or not mdb_os_pwd or not mdb_os_hosts or not mdb_os_index_name:
        st.warning(
            "Пожалуйста, задайте свои учетные данные (в secrets/.env или в раскрывающейся панели слева) для запуска этого приложения.")

    # Загрузка pdf файлов
    uploaded_files = st.file_uploader(
        "После загрузки файлов в формате pdf начнется их добавление в векторную базу данных MDB Opensearch.", accept_multiple_files=True, type=['pdf'])

    # если файлы загружены, сохраняем их во временную папку и потом заносим в vectorstore
    if uploaded_files:
        # создаем временную папку и сохраняем в ней загруженные файлы
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                for uploaded_file in uploaded_files:
                    file_name = uploaded_file.name
                    # сохраняем файл во временную папку
                    with open(os.path.join(temp_dir, file_name), "wb") as f:
                        f.write(uploaded_file.read())
                # отображение спиннера во время инъекции файлов
                with st.spinner("Добавление ваших файлов в базу ..."):
                    ingest_docs(temp_dir)
                    st.success("Ваш(и) файл(ы) успешно принят(ы)")
                    st.session_state['ready'] = True
        except Exception as e:
            st.error(
                f"При загрузке ваших файлов произошла ошибка: {str(e)}")

    # Логика обработки сообщений от пользователей
    # инициализировать историю чата, если ее пока нет 
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    # инициализировать состояние готовности, если его пока нет
    if 'ready' not in st.session_state:
        st.session_state['ready'] = True

    if st.session_state['ready']:

        # подключиться к векторной БД Opensearch, используя учетные данные (проверка подключения)
        conn = OpenSearch(
            mdb_os_hosts,
            http_auth=('admin', mdb_os_pwd),
            use_ssl=True,
            verify_certs=False,
            ca_certs=MDB_OS_CA
            )

        # инициализировать модели YandexEmbeddings и YandexGPT
        embeddings = YandexEmbeddings(folder_id=yagpt_folder_id, api_key=yagpt_api_key)

        # обращение к модели YaGPT
        llm = YandexLLM(api_key=yagpt_api_key, folder_id=yagpt_folder_id, temperature = 0.01, max_tokens=7000)

        # инициализация retrival chain - цепочки поиска
        vectorstore = OpenSearchVectorSearch (
            embedding_function=embeddings,
            index_name = mdb_os_index_name,
            opensearch_url=mdb_os_hosts,
            http_auth=("admin", mdb_os_pwd),
            use_ssl = True,
            verify_certs = False,
            ca_certs = MDB_OS_CA,
            engine = 'lucene'
        )  

        template = """Представь, что ты полезный ИИ-помощник. Твоя задача отвечать на вопросы на русском языке в рамках предоставленного ниже текста.
        Отвечай точно в рамках предоставленного текста, даже если тебя просят придумать.
        Отвечай вежливо в официальном стиле. Eсли знаешь больше, чем указано в тексте, а внутри текста ответа нет, отвечай 
        "Я могу давать ответы только по тематике загруженных документов. Мне не удалось найти в документах ответ на ваш вопрос."
        {context}
        Вопрос: {question}
        Твой ответ:
        """
        QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
        qa = RetrievalQA.from_chain_type(
            llm,
            retriever=vectorstore.as_retriever(search_kwargs={'k': 5}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
        )

        if 'generated' not in st.session_state:
            st.session_state['generated'] = [
                "Что бы вы хотели узнать о документе?"]

        if 'past' not in st.session_state:
            st.session_state['past'] = ["Привет!"]

        # контейнер для истории чата
        response_container = st.container()

        # контейнер для текстового поля
        container = st.container()

        with container:
            with st.form(key='my_form', clear_on_submit=True):
                user_input = st.text_input(
                    "Вопрос:", placeholder="О чем этот документ?", key='input')
                submit_button = st.form_submit_button(label='Отправить')

            if submit_button and user_input:
                # отобразить загрузочный "волчок"
                with st.spinner("Думаю..."):
                    print("История чата: ", st.session_state['chat_history'])
                    output = qa(
                        {"query": user_input})
                    print(output)
                    st.session_state['past'].append(user_input)
                    st.session_state['generated'].append(output['result'])

                    # # обновляем историю чата с помощью вопроса пользователя и ответа от бота
                    st.session_state['chat_history'].append(
                        {"вопрос": user_input, "ответ": output['result']})

        if st.session_state['generated']:
            with response_container:
                for i in range(len(st.session_state['generated'])):
                    message(st.session_state["past"][i], is_user=True, key=str(
                        i) + '_user')
                    message(st.session_state["generated"][i], key=str(
                        i))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # st.write(f"Что-то пошло не так. Возможно, не хватает входных данных для работы. {str(e)}")
        st.write(f"Не хватает входных данных для продолжения работы. См. панель слева.")