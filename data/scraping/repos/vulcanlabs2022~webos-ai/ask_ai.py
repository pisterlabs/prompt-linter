import json
from loguru import logger
import os
import tornado
from llama_cpp import Llama
from tornado import ioloop
from tornado.options import define, options

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import langchain
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from tornado.escape import json_decode
from langchain.vectorstores import FAISS
from utils.conversation import conv_templates, get_default_conv_template, SeparatorStyle
langchain.verbose = False
import os
path1 = os.path.abspath('.')

def custom_text_splitter(text, chunk_size=128, chunk_overlap=16):
    chunks = []
    start = 0
    words = text.split()

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start = end - chunk_overlap
    return chunks
def split_text(text):
    # Split text into smaller chunks
    texts = custom_text_splitter(text)

    # Store chunks in a file
    with open('text_chunks.json', 'w') as f:
        json.dump(texts, f)


def load_faiss_db(db_path,embeddings):
    db = FAISS.load_local(db_path,embeddings = embeddings)
    return db

def merge_faiss_db(db_paths,embeddings):
    db0 = load_faiss_db(db_paths[0],embeddings = embeddings)
    [db0.merge_from(load_faiss_db(i,embeddings = embeddings)) for i in db_paths[1:]]
    return db0

def evaluate(query,history,text):
    conv = get_default_conv_template('conv_vicuna_v1_1').copy()
    if not text:
        if len(history) == 0:
            inp = query
            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
        else:
            for j, sentence in enumerate(history):
                for i in range(len(sentence)):
                    if i == 0:
                        role = "USER"
                        conv.append_message(role, sentence[0])
                    else:
                        role = "ASSISTANT"
                        conv.append_message(role, sentence[1])
            inp = query
            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    else:
        if len(history) == 0:
            inp = "Based on the known information below, please provide a concise and professional answer " \
                  "to the user's question.,If the answer cannot be obtained from the information provided, " \
                  "The known content: " + text + "\nquestion:" + query
            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
        else:
            for j, sentence in enumerate(history):
                for i in range(len(sentence)):
                    if i == 0:
                        role = "USER"
                        conv.append_message(role, sentence[0])
                    else:
                        role = "ASSISTANT"
                        conv.append_message(role, sentence[1])
            inp = query
            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    logger.debug(prompt)
    '''流式输出'''
    stream = llm(
        prompt,
        max_tokens=512,
        stop=["ASSISTANT:"],
        stream=True
    )
    # print(stream)
    return stream

class VicunaHandler(tornado.web.RequestHandler):
    executor = ThreadPoolExecutor(4)

    # @tornado.web.asynchronous
    # @tornado.gen.coroutine
    def post(self, *args, **kwargs):
        ret = {
            "ret": 1,
            "errcode": 1,
            "response": "",
        }
        try:
            data = json_decode(self.request.body)
            logger.debug("Assistant start")
            query = data.get("query", "")
            type = data.get("type","")
            path = data.get("path","")
            history = data.get("history",[])
            if type == "basic":
                for i in evaluate(query, history, ''):
                    # print(i)
                    # if i["choices"][0]["finish_reason"] == "stop":
                    #     continue
                    # print(i["choices"][0]["text"])
                    ret["response"] += i["choices"][0]["text"]
                    logger.debug(ret)
                    self.write(ret)
                    self.write('\n')
                    self.flush()
            elif type == "single_doc":
                if len(history) == 0:
                    logger.debug("%s process doc" % type)
                    # 判断是否已经建过索引
                    if os.path.exists("/data/save_index/%s_index" % path.replace("/", "_")):
                        logger.debug("%s index had been added"%path)
                        docsearch = load_faiss_db("/data/save_index/%s_index" % path.replace("/", "_"),embeddings)
                    else:
                        text = ""
                        for i in load_data_by_langchain(path):
                            text += i.page_content
                        # text = load_data_by_langchain(file_name)[0].page_content
                        split_texts = custom_text_splitter(text)
                        docs = []
                        for one_conent in split_texts:
                            # docs = load_file("microsoft.txt",sentence_size=SENTENCE_SIZE)
                            docs.append(Document(page_content=one_conent + "\n", metadata={"source": filepath}))
                        docsearch = FAISS.from_documents(docs, embeddings)
                        docsearch.save_local("/data/save_index/%s_index" % path.replace("/", "_"))
                        logger.debug("%s index has been added" % path)
                    docs = docsearch.similarity_search_with_score(query, k=1)
                    simdoc = ""
                    for doc in docs:
                        # if doc[1] <= 0.3:
                        #     continue
                        simdoc += doc[0].page_content
                    logger.debug("%s search doc done"%type)
                    if len(simdoc) == 0:
                        # todo 转成世界知识问答？？
                        ret["response"] = "Sorry, I can't get any useful information based on the question"
                        logger.debug(ret)
                        self.write(ret)
                        self.write('\n')
                        self.flush()
                    else:
                    # for i in generate_streaming_completion(query,simdoc):
                        history = []
                        for i in evaluate(query, history, simdoc):
                            # if i["choices"][0]["finish_reason"] == "stop":
                            #     continue
                            # print(i["choices"][0]["text"])
                            ret["response"] += i["choices"][0]["text"]
                            logger.debug(ret)
                            self.write(ret)
                            self.write('\n')
                            self.flush()
                    # self.write(ret)
                else:
                    for i in evaluate(query, history, ""):
                        # if i["choices"][0]["finish_reason"] == "stop":
                        #     continue
                        ret["response"] += i["choices"][0]["text"]
                        logger.debug(ret)
                        self.write(ret)
                        self.write('\n')
                        self.flush()

            elif type == "full_doc":
                if len(history) == 0:
                    logger.debug("%s process doc" % type)

                    files = os.listdir("/data/save_index")
                    new_files = []
                    for file in files:
                        new_path = '/data' + "/save_index/" + file
                        new_files.append(new_path)
                    logger.debug("the index num is %s"%len(new_files))
                    if len(new_files) == 0:
                        ret["response"] = "Sorry, I can't get any embedding files, please check the embedding file is existed?"
                        logger.debug(ret)
                        self.write(ret)
                        self.write('\n')
                        self.flush()
                    else:
                        db = merge_faiss_db(new_files,embeddings=embeddings)
                        # todo 增加相似得分的判断
                        docs = db.similarity_search_with_score(query, k=1)
                        simdoc = ""
                        for doc in docs:
                            if doc[1]<=0.3:
                                simdoc += doc[0].page_content
                        logger.debug("%s search doc done"%type)
                        if len(simdoc) == 0:
                            # todo 转成世界知识问答？？
                            ret["response"] = "Sorry, I can't get any useful information based on the question"
                            logger.debug(ret)
                            self.write(ret)
                            self.write('\n')
                            self.flush()
                        else:
                            # for i in generate_streaming_completion(query,simdoc):
                            history = []
                            for i in evaluate(query, history, simdoc):
                                # if i["choices"][0]["finish_reason"] == "stop":
                                #     continue
                                # print(i["choices"][0]["text"])
                                ret["response"] += i["choices"][0]["text"]
                                logger.debug(ret)
                                self.write(ret)
                                self.write('\n')
                                self.flush()
                    # self.write(ret)
                else:
                    no_index_answer = "Sorry, I can't get any embedding files, please check the embedding file is existed?"
                    if history[-1][1] == no_index_answer:
                        files = os.listdir("/data/save_index")
                        new_files = []
                        for file in files:
                            new_path = '/data' + "/save_index/" + file
                            new_files.append(new_path)
                        if len(new_files) == 0:
                            ret["response"] = "Sorry, I can't get any embedding files, please check the embedding file is existed?"
                            logger.debug(ret)
                            self.write(ret)
                            self.write('\n')
                            self.flush()
                        else:
                            db = merge_faiss_db(new_files, embeddings=embeddings)
                            # todo 增加相似得分的判断
                            docs = db.similarity_search_with_score(query, k=1)
                            simdoc = ""
                            for doc in docs:
                                if doc[1] <= 0.3:
                                    simdoc += doc[0].page_content
                            logger.debug("%s search doc done" % type)
                            if len(simdoc) == 0:
                                # todo 转成世界知识问答？？
                                ret["response"] = "Sorry, I can't get any useful information based on the question"
                                logger.debug(ret)
                                self.write(ret)
                                self.write('\n')
                                self.flush()
                            else:
                                # for i in generate_streaming_completion(query,simdoc):
                                history = []
                                for i in evaluate(query, history, simdoc):
                                    # if i["choices"][0]["finish_reason"] == "stop":
                                    #     continue
                                    # print(i["choices"][0]["text"])
                                    ret["response"] += i["choices"][0]["text"]
                                    logger.debug(ret)
                                    self.write(ret)
                                    self.write('\n')
                                    self.flush()
                    else:
                        for i in evaluate(query, history, ''):
                            # if i["choices"][0]["finish_reason"] == "stop":
                            #     continue
                            ret["response"] += i["choices"][0]["text"]
                            logger.debug(ret)
                            self.write(ret)
                            self.write('\n')
                            self.flush()
            else:
                logger.debug("type is not matched")
        except Exception as e:
            logger.debug(e)
            pass
        self.write(ret)
        logger.debug(ret)
        self.finish()

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.write({"status": 1})

def make_app():
    return tornado.web.Application(
        [
            (r"/", MainHandler),
            (r"/healthcheck", MainHandler),
            (r"/nlp/Vicuna_infer_v1", VicunaHandler),

        ],
        debug=False
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--cpp_model", type=str, default='')
    parser.add_argument("--embedding_model", type=str, default='')
    parser.add_argument("--style", type=str, default="simple",
                        choices=["simple", "rich"], help="Display style.")
    parser.add_argument("--n_threads", type=int,default=4)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--port", default=8087)
    args = parser.parse_args()

    from langchain.embeddings.huggingface import HuggingFaceInstructEmbeddings
    embeddings = HuggingFaceInstructEmbeddings(model_name=args.embedding_model)
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    llm = Llama(model_path=args.cpp_model,n_ctx=2048,n_threads=args.n_threads)
    define("port", default=args.port, help="run on the given port", type=int)

    app = make_app()
    logger.debug('web listen on %d' % options.port)
    app.listen(options.port, xheaders=True)
    ioloop.IOLoop.instance().start()