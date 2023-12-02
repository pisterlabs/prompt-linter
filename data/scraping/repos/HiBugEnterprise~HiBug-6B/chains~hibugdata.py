"""
Product:HiBug1.0
"""
from chains.filterbase import word_dict
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from vectorstores import MyFAISS
from langchain.document_loaders import UnstructuredFileLoader, TextLoader, CSVLoader
from configs.config import *
import datetime
from hibugsplitter import ChineseTextSplitter
from typing import List
from utils import torch_gc
from tqdm import tqdm
from pypinyin import lazy_pinyin
from loader import UnstructuredPaddleImageLoader, UnstructuredPaddlePDFLoader
from models.base import (BaseAnswer,
                         AnswerResult)
#import datetime
from models.loader.args import parser
from models.loader import LoaderCheckPoint
import models.shared as shared
from langchain.docstore.document import Document
from functools import lru_cache
from hibugsplitter.enhance import zh_title_enhance
import time
import json


# patch HuggingFaceEmbeddings to make it hashable
def _embeddings_hash(self):
    return hash(self.model_name)


HuggingFaceEmbeddings.__hash__ = _embeddings_hash


# will keep CACHED_VS_NUM of vector store caches
@lru_cache(CACHED_VS_NUM)
def load_vector_store(vs_path, embeddings):
    return MyFAISS.load_local(vs_path, embeddings)


def tree(filepath, ignore_dir_names=None, ignore_file_names=None):
    """返回两个列表，第一个列表为 filepath 下全部文件的完整路径, 第二个为对应的文件名"""
    if ignore_dir_names is None:
        ignore_dir_names = []
    if ignore_file_names is None:
        ignore_file_names = []
    ret_list = []
    if isinstance(filepath, str):
        if not os.path.exists(filepath):
            print("路径不存在")
            return None, None
        elif os.path.isfile(filepath) and os.path.basename(filepath) not in ignore_file_names:
            return [filepath], [os.path.basename(filepath)]
        elif os.path.isdir(filepath) and os.path.basename(filepath) not in ignore_dir_names:
            for file in os.listdir(filepath):
                fullfilepath = os.path.join(filepath, file)
                if os.path.isfile(fullfilepath) and os.path.basename(fullfilepath) not in ignore_file_names:
                    ret_list.append(fullfilepath)
                if os.path.isdir(fullfilepath) and os.path.basename(fullfilepath) not in ignore_dir_names:
                    ret_list.extend(tree(fullfilepath, ignore_dir_names, ignore_file_names)[0])
    return ret_list, [os.path.basename(p) for p in ret_list]


def load_file(filepath, sentence_size=SENTENCE_SIZE, using_zh_title_enhance=ZH_TITLE_ENHANCE):
    if filepath.lower().endswith(".md"):
        loader = UnstructuredFileLoader(filepath, mode="elements")
        docs = loader.load()
    elif filepath.lower().endswith(".txt"):
        loader = TextLoader(filepath, autodetect_encoding=True)
        #file_size = os.path.getsize(filepath)
        textsplitter = ChineseTextSplitter(pdf=False, sentence_size=sentence_size)
        #docs = loader.load_and_split(textsplitter)
        docs = loader.load_and_split(textsplitter)
    elif filepath.lower().endswith(".pdf"):
        loader = UnstructuredPaddlePDFLoader(filepath)
        textsplitter = ChineseTextSplitter(pdf=True, sentence_size=sentence_size)
        docs = loader.load_and_split(textsplitter)
    elif filepath.lower().endswith(".jpg") or filepath.lower().endswith(".png"):
        loader = UnstructuredPaddleImageLoader(filepath, mode="elements")
        textsplitter = ChineseTextSplitter(pdf=False, sentence_size=sentence_size)
        docs = loader.load_and_split(text_splitter=textsplitter)
    elif filepath.lower().endswith(".csv"):
        loader = CSVLoader(filepath)
        docs = loader.load()
    else:
        loader = UnstructuredFileLoader(filepath, mode="elements")
        textsplitter = ChineseTextSplitter(pdf=False, sentence_size=sentence_size)
        docs = loader.load_and_split(text_splitter=textsplitter)
    if using_zh_title_enhance:
        docs = zh_title_enhance(docs)
    write_check_file(filepath, docs)
    return docs


def write_check_file(filepath, docs):
    folder_path = os.path.join(os.path.dirname(filepath), "tmp_files")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    fp = os.path.join(folder_path, 'load_file.txt')
    with open(fp, 'a+', encoding='utf-8') as fout:
        fout.write("filepath=%s,len=%s" % (filepath, len(docs)))
        fout.write('\n')
        for i in docs:
            fout.write(str(i))
            fout.write('\n')
        fout.close()


def generate_prompt(related_docs: List[str],
                    query: str,
                    prompt_template: str = PROMPT_TEMPLATE, ) -> str:
    context = "\n".join([doc.page_content for doc in related_docs])
    prompt = prompt_template.replace("{question}", query).replace("{context}", context)
    return prompt


def search_result2docs(search_results):
    docs = []
    for result in search_results:
        doc = Document(page_content=result["snippet"] if "snippet" in result.keys() else "",
                       metadata={"source": result["link"] if "link" in result.keys() else "",
                                 "filename": result["title"] if "title" in result.keys() else ""})
        docs.append(doc)
    return docs


class Hibugdoc:
    llm: BaseAnswer = None
    embeddings: object = None
    top_k: int = VECTOR_SEARCH_TOP_K
    chunk_size: int = CHUNK_SIZE
    chunk_conent: bool = True
    score_threshold: int = VECTOR_SEARCH_SCORE_THRESHOLD
    history = []
    isprocess: bool = False
    cancel_sse: bool =False

    def init_cfg(self,
                 embedding_model: str = EMBEDDING_MODEL,
                 embedding_device=EMBEDDING_DEVICE,
                 llm_model: BaseAnswer = None,
                 top_k=VECTOR_SEARCH_TOP_K,
                 ):
        self.llm = llm_model
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_dict[embedding_model],
                                                model_kwargs={'device': embedding_device})
        self.top_k = top_k
        self.history = []
        self.cancel_sse = False
        self.isprocess = False

    def init_knowledge_vector_store(self,
                                    filepath: str or List[str],
                                    vs_path: str or os.PathLike = None,
                                    sentence_size=SENTENCE_SIZE):
        loaded_files = []
        failed_files = []
        if isinstance(filepath, str):
            if not os.path.exists(filepath):
                print("The file path does not exist.")
                return None
            elif os.path.isdir(filepath):
                docs = []
                start_time = time.time()
                for fullfilepath, file in tqdm(zip(*tree(filepath, ignore_dir_names=['tmp_files'])), desc="Loading files from HiBug Database..."):
                    try:
                        docs += load_file(fullfilepath, sentence_size)
                        loaded_files.append(fullfilepath)
                    except Exception as e:
                        logger.error(e)
                        failed_files.append(file)
                elapsed_time = time.time() - start_time
                print("Loading Time：", elapsed_time, "seconds")

                if len(failed_files) > 0:
                    logger.info("The following files failed to load:")
                    for file in failed_files:
                        logger.info(f"{file}\n")

        else:
            docs = []
            for file in filepath:
                try:
                    docs += load_file(file)
                    logger.info(f"{file} 已成功加载")
                    loaded_files.append(file)
                except Exception as e:
                    logger.error(e)
                    logger.info(f"{file} 未能成功加载")
        if len(docs) > 0:
            logger.info("文件加载完毕，正在生成向量库")
            if vs_path and os.path.isdir(vs_path) and "index.faiss" in os.listdir(vs_path):
                vector_store = load_vector_store(vs_path, self.embeddings)
                vector_store.add_documents(docs)
                torch_gc()
            else:
                if not vs_path:
                    vs_path = os.path.join(KB_ROOT_PATH,
                                           f"""{"".join(lazy_pinyin(os.path.splitext(file)[0]))}_FAISS_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}""",
                                           "vector_store")
                vector_store = MyFAISS.from_documents(docs, self.embeddings)  # docs 为Document列表
                torch_gc()

            vector_store.save_local(vs_path)
            return vs_path, loaded_files
        else:
            logger.info("文件均未成功加载，请检查依赖包或替换为其他文件再次上传。")
            return None, loaded_files

    def one_knowledge_add(self, vs_path, one_title, one_conent, one_content_segmentation, sentence_size):
        try:
            if not vs_path or not one_title or not one_conent:
                logger.info("知识库添加错误，请确认知识库名字、标题、内容是否正确！")
                return None, [one_title]
            docs = [Document(page_content=one_conent + "\n", metadata={"source": one_title})]
            if not one_content_segmentation:
                text_splitter = ChineseTextSplitter(pdf=False, sentence_size=sentence_size)
                docs = text_splitter.split_documents(docs)
            if os.path.isdir(vs_path) and os.path.isfile(vs_path + "/index.faiss"):
                vector_store = load_vector_store(vs_path, self.embeddings)
                vector_store.add_documents(docs)
            else:
                vector_store = MyFAISS.from_documents(docs, self.embeddings)  ##docs 为Document列表
            torch_gc()
            vector_store.save_local(vs_path)
            return vs_path, [one_title]
        except Exception as e:
            logger.error(e)
            return None, [one_title]

    def get_knowledge_based_answer(self, query, vs_path, chat_history=[], streaming: bool = STREAMING):
        vector_store = load_vector_store(vs_path, self.embeddings)
        vector_store.chunk_size = self.chunk_size
        vector_store.chunk_conent = self.chunk_conent
        vector_store.score_threshold = self.score_threshold
        related_docs_with_score = vector_store.similarity_search_with_score(query, k=self.top_k)
        resp = "void"
        history = []
        #print(related_docs_with_score)
        torch_gc()
        if len(related_docs_with_score) > 0:
            prompt = generate_prompt(related_docs_with_score, query)
        else:
            prompt = query

        for answer_result in self.llm.generatorAnswer(prompt=prompt, history=chat_history,
                                                      streaming=streaming):
            if self.cancel_sse:
                self.cancel_sse=False
                break
            resp = answer_result.llm_output["answer"]
            history = answer_result.history
            #history[-1][0] = query
            for word, replacement in word_dict.items():
                resp = resp.replace(word, replacement)
            history[-1][-1]=resp
            now = datetime.datetime.now()
            time = now.strftime("%Y-%m-%d %H:%M:%S")
            response = json.dumps({
                        'response': resp,
                        'history': history,
                        'code': 200,
                        'time': time
            })
            self.isprocess = True
            yield response
        self.isprocess = False
        now = datetime.datetime.now()
        time = now.strftime("%Y-%m-%d %H:%M:%S")
        log = "[" + time + "] " + ', response:"' + repr(resp) + '"'+ ', history:'+ repr(history)
        print(log)
        source_documents = [
            f"""知识库来源 [{inum + 1}] {os.path.split(doc.metadata['source'])[-1]}：\n\n{doc.page_content}\n\n"""
            f"""文本相关性：{doc.metadata['score']}\n\n"""
            for inum, doc in enumerate(related_docs_with_score)
        ]
        print(source_documents) 


    def get_knowledge_based_answer_demo(self, query, vs_path, chat_history=[], streaming: bool = STREAMING):
        vector_store = load_vector_store(vs_path, self.embeddings)
        vector_store.chunk_size = self.chunk_size
        vector_store.chunk_conent = self.chunk_conent
        vector_store.score_threshold = self.score_threshold
        related_docs_with_score = vector_store.similarity_search_with_score(query, k=self.top_k)
        torch_gc()
        if len(related_docs_with_score) > 0:
            prompt = generate_prompt(related_docs_with_score, query)
        else:
            prompt = query

        for answer_result in self.llm.generatorAnswer(prompt=prompt, history=chat_history,
                                                      streaming=streaming):
            resp = answer_result.llm_output["answer"]
            history = answer_result.history
            history[-1][0] = query
            response = {"query": query,
                        "result": resp,
                        "source_documents": related_docs_with_score}
            yield response, history
            #yield response, history

    def update_file_from_vector_store(self,
                                      filepath: str or List[str],
                                      vs_path,
                                      docs: List[Document],):
        vector_store = load_vector_store(vs_path, self.embeddings)
        status = vector_store.update_doc(filepath, docs)
        return status

    def list_file_from_vector_store(self,
                                    vs_path,
                                    fullpath=False):
        vector_store = load_vector_store(vs_path, self.embeddings)
        docs = vector_store.list_docs()
        if fullpath:
            return docs
        else:
            return [os.path.split(doc)[-1] for doc in docs]


if __name__ == "__main__":
    # 初始化消息
    args = None
    args = parser.parse_args(args=['--model-dir', '/media/checkpoint/', '--model', 'chatglm-6b', '--no-remote-model'])

    args_dict = vars(args)
    shared.loaderCheckPoint = LoaderCheckPoint(args_dict)
    llm_model_ins = shared.loaderLLM()
    llm_model_ins.set_history_len(LLM_HISTORY_LEN)

    hibug_doc = LocalDocQA()
    hibug_doc.init_cfg(llm_model=llm_model_ins)
    query = "HiBug的参数量是多少"
    vs_path = "/hibugvector/locallib"
    last_print_len = 0
    # for resp, history in hibug_doc.get_knowledge_based_answer(query=query,
    #                                                              vs_path=vs_path,
    #                                                              chat_history=[],
    #                                                              streaming=True):
    for resp, history in hibug_doc.get_search_result_based_answer(query=query,
                                                                     chat_history=[],
                                                                     streaming=True):
        print(resp["result"][last_print_len:], end="", flush=True)
        last_print_len = len(resp["result"])
    source_text = [f"""出处 [{inum + 1}] {doc.metadata['source'] if doc.metadata['source'].startswith("http")
    else os.path.split(doc.metadata['source'])[-1]}：\n\n{doc.page_content}\n\n"""
                   # f"""相关度：{doc.metadata['score']}\n\n"""
                   for inum, doc in
                   enumerate(resp["source_documents"])]
    logger.info("\n\n" + "\n\n".join(source_text))
    pass
