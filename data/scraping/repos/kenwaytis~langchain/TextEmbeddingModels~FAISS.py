import os
import getpass
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from .FAISSSearch import FAISSSearch
from langchain.document_loaders import TextLoader
from .HuggingFaceHub import TextEmbeddings
from .chinese_text_splitter import ChineseTextSplitter
from loguru import logger

# 没有 AVX2 优化的情况下初始化 FAISS，取消注释以下行
# os.environ['FAISS_NO_AVX2'] = '1'
class FaissSearch():
    """
    对本地文档执行FAISS搜索
    """
    def __init__(self) -> None:
        """
        初始化:
            - 嵌入层。当前固定调用私有类TextEmbeddings
        """
        init_embeddings = TextEmbeddings()
        self.embeddings = init_embeddings.initmodel()
        
    def build_index(self, file_path, save_path = "/home/db/default_faiss_index"):
        """
        输入：
            - file_path: 本地文档文件路径。
        """
        loader = TextLoader(file_path,autodetect_encoding=True)
        text_splitter = ChineseTextSplitter(pdf=False,sentence_size=100)
        docs = loader.load_and_split(text_splitter)
        db = FAISSSearch.from_documents(docs, self.embeddings)
        db.save_local(save_path)
        logger.info(f'success save {file_path} at {save_path}')

    def search(self, query, db_local = '/home/db/default_faiss_index'):
        db = FAISSSearch.load_local(db_local,self.embeddings)
        return db.similarity_search_with_score(query)


if __name__ == "__main__":
    Test = FaissSearch()
    # Test.build_index('/home/test_combine.txt',save_path='/home/db/test_combine_db')
    ans = Test.search("蓝衣少年", db_local='/home/db/test_combine_db')
    logger.info(ans)

