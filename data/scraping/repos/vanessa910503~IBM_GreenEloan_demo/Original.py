import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import TextLoader, PyPDFLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.prompts import (
    FewShotChatMessagePromptTemplate,
    ChatPromptTemplate,
    SemanticSimilarityExampleSelector
)

### Data Processing (PDF + CSV)
def data_processing(pdf_file_path, csv_file_path):
    # Load PDF File 
    pdf_file = f"{pdf_file_path}"
    loader = pdf_file.endswith(".pdf") and PyPDFLoader(pdf_file) or TextLoader(pdf_file)

    # PDF File Processing
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100, separators=['\n', '\n\n'])
    pdf_texts = loader.load_and_split(splitter)

    # Load CSV File
    csv_file = f"{csv_file_path}"
    csv_loader = CSVLoader(csv_file, encoding='utf-8')
    csv_texts = csv_loader.load()

    # Conmibe PDF and CSV Texts
    combined_texts = pdf_texts + csv_texts

    # Set up local db
    embeddings = OpenAIEmbeddings()  
    vectorstore = Chroma.from_documents(combined_texts, embeddings) 

    return vectorstore

### Selected Questions List (not sure if we need this function, but in case we will change the data type to csv. I keep this function here)
def selected_questions_list(questions):
    questions_list = questions
    return questions_list

### Establish chat chain
def chat_chain(vectorstore, questions):
    questions_list = selected_questions_list(questions)
    # memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0), vectorstore.as_retriever())

    for question in questions_list:
        chat_history = []  # Reset chat history for each new question
        print('Q:', question)
        query = question
        if not query:
            break

        # result = qa({"question": '您是一個經過訓練的語言模型，專門用於判斷是非題並生成相應的解釋。這些問題將涉及信貸核准，幫助公司評估個人或企業的信貸風險並做出信貸決策。您的任務是根據所提供的資料（一份永續報告書 pdf 檔案、和一份關於該公司於金管會所揭發之裁罰案件的 csv 檔案）產出<是>或<否>或<不確定>的答案，以及相應的解釋。請嚴格按以下格式提供是非題和解釋（每次回答都續按照此格式）：選擇：{是/否/不確定}, 解釋：{解釋}。以下為你將要判斷與回答的問題：' + query + ' (用繁體中文回答)',
        #             "chat_history": chat_history})

        result = qa({"question": '您是一個經過訓練的語言模型，專門用於判斷是非題並生成相應的解釋。這些問題將涉及信貸核准，幫助公司評估個人或企業的信貸風險並做出信貸決策。您的任務是根據所提供的資料產出<是>或<否>或<不確定>的答案，以及相應的解釋。請嚴格按以下格式提供是非題和解釋（每次回答都續按照此格式）：選擇：{是/否/不確定}, 解釋：{解釋}。以下為你將要判斷與回答的問題：' + query + ' (用繁體中文回答)',
                    "chat_history": chat_history})
        
        print('A:', result['answer'])
        print('-' * 100)
        chat_history.append((query, result['answer']))

    return chat_history

### Main Function
def main():
    os.environ["OPENAI_API_KEY"] = "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"
    pdf_file_path = "IBM_GreenEloan_demo/data/pdf/CTBC_2022_Sustainability_Report_zh3.pdf"
    csv_file_path = "IBM_GreenEloan_demo/data/csv/中國信託_ViolationItems.csv"
    selected_questions = [
                # 第一部分：E/S/G 違規項目
                '請根據資料中 <裁處書發文日期> 判斷近二年該公司是否發生洗錢或資助資恐活動情節重大或導致停工 / 停業者',
                # 第二部分：E/S/G 關鍵作為
                '近一年該公司是否曾獲得外部永續相關獎項',
                # 第三部分：Environmental
                '該公司是否投資於節能或綠色能源相關環保永續之機器設備，或投資於我國綠能產業（如:再生能源電廠）等，或有發行或投資其資金運用於綠色或社會效益投資計畫並具實質效益之永續發展金融商品，並揭露其投資情形及具體效益？',
                # 第三部分：Social
                '該公司是否揭露員工福利政策？（如：保險、育嬰假、退休制度、員工持股、工作者健康促進、在職訓練…等）',
                # 第三部分：Governancce
                '該公司是否已將「股利政策」、「董監事及經理人績效評估與酬金制度」、「員工權益」，揭露於公司網路、年報或永續報告書？']
    
    vectorstore = data_processing(pdf_file_path, csv_file_path)
    questions = selected_questions_list(selected_questions)
    chat_history = chat_chain(vectorstore, questions)

    return chat_history

### Execute
if __name__ == "__main__":
    main()

