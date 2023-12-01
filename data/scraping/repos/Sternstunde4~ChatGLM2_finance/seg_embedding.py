# 将切分好的数据存入向量数据库
import pickle

from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
import os
from tqdm import *

embeddings = HuggingFaceEmbeddings(model_name="D:/LLM_dev/LangChain/model/bge-large-zh", model_kwargs={'device': "cuda"})
# raw_documents = TextLoader('D:\LLM_dev\FinGPT-intern\pdf_to_txt\\test_txt\\2020-01-21__江苏安靠智能输电工程科技股份有限公司__300617__安靠智电__2019年__年度报告_txt.txt', encoding='utf-8').load()
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# documents = text_splitter.split_documents(raw_documents)
#
# print(documents)
# db = FAISS.from_documents(documents, embeddings)
#
# store_path = r"vector_store/faiss_index"
# os.mkdir(store_path)
# db.save_local(store_path)
# query = "江苏安靠智能输电工程科技股份有限公司法定代表人是多少？"
# new_db = FAISS.load_local("faiss_index", embeddings)
# docs = new_db.similarity_search(query)
# print(docs[0])


if __name__ == '__main__':
    # dict = {}
    # for id,folder in enumerate(os.listdir('Segmentation')):  # folder是公司名
    #     if folder.endswith('_txt.txt'):
    #         dict[id] = folder
    #     # print(documents)
    #     # print(folder)
    # with open('vector_store\dict.pkl', 'wb') as f:
    #     dict = pickle.dump(dict, f)


    # 获取公司_txt.txt对应的序号作为向量数据库的名称
    with open('vector_store0812\dict.pkl', 'rb') as f:
        dict = pickle.load(f)
    for key, value in tqdm(dict.items()):
        documents = []
        folder_id = key
        folder = value
        for file_name in os.listdir(f'Segmentation\\{folder}'):  # file_name是文件每一段的序号
            file_path = os.path.join(f'Segmentation\\{folder}', file_name)
            raw_documents = TextLoader(file_path, encoding='utf-8').load()
            documents.append(raw_documents[0])
        # print(documents)
        print(folder)
        try:
            db = FAISS.from_documents(documents, embeddings)
            store_path = f"vector_store0812/{folder_id}"
            os.mkdir(store_path)
            db.save_local(store_path)
            print("已存储")
        except:
            print("未存储")
            with open('vector_store0812\save_error', 'a') as f:
                f.write(folder +'\n')

    # 测试单个年报的向量数据库存储
    # with open('vector_store0810\dict.pkl', 'rb') as f:
    #     dict = pickle.load(f)
    # for key, value in dict.items():
    #     documents = []
    #     folder_id = key
    #     folder = value
    #     for file_name in os.listdir(f'Seg_Mask\\2020-01-21__江苏安靠智能输电工程科技股份有限公司__300617__安靠智电__2019年__年度报告_txt.txt'):  # file_name是文件每一段的序号
    #         file_path = os.path.join(f'Seg_Mask\\2020-01-21__江苏安靠智能输电工程科技股份有限公司__300617__安靠智电__2019年__年度报告_txt.txt', file_name)
    #         raw_documents = TextLoader(file_path, encoding='utf-8').load()
    #         documents.append(raw_documents[0])
    #     # print(documents)
    #     print(folder)
    #     try:
    #         db = FAISS.from_documents(documents, embeddings)
    #         store_path = f"vector_store0810/{folder_id}"
    #         os.mkdir(store_path)
    #         db.save_local(store_path)
    #         print("已存储")
    #     except:
    #         print("未存储")
    #         with open('vector_store0810\save_error', 'a') as f:
    #             f.write(folder +'\n')

    # query = "公司法人是谁？"
    # new_db = FAISS.load_local("vector_store0810\\0", embeddings)
    # docs = new_db.similarity_search(query)
    # print(docs[0])