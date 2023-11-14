import logging
import bson
import json
from time import time
from uuid import uuid4
from copy import deepcopy
from datetime import datetime
from langchain.schema import Document, BaseRetriever
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
)
from elasticsearch_dsl import (
    UpdateByQuery,
    Search,
    Q,
    Boolean,
    Date,
    Integer,
    Document as ESDocumentBase,
    InnerDoc,
    Join,
    Keyword,
    Long,
    Nested,
    Object,
    Text,
    connections,
    DenseVector
)
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks import get_openai_callback
from app import app

class ObjID():
    def new_id():
        return str(bson.ObjectId())
    def is_valid(value):
        return bson.ObjectId.is_valid(value)


class NotFound(Exception): pass


connections.create_connection(
    hosts=f"http://{app.config['ES_HOST']}:{app.config['ES_PORT']}",
    # basic_auth=('elastic', '')
)


class ESDocument(ESDocumentBase):

    status = Integer()
    created = Date()
    modified = Date()

    @property
    def id(self):
        return self.meta.id

    @property
    def created_at(self):
        return int(self.created.timestamp() * 1000)

    def save(self, *args, **kwargs):
        if self.created is None:
            self.created = datetime.now()
        if self.status is None:
            self.status = 0
        self.modified = datetime.now()
        super().save(*args, **kwargs)


class User(ESDocument):
    openid = Keyword()
    name = Text(fields={"keyword": Keyword()})
    extra = Object()    # 用于保存 JSON 数据

    class Index:
        name = 'user'


class Collection(ESDocument):
    user_id = Keyword()  # 将字符串作为文档的 ID 存储
    name = Text(analyzer='ik_max_word')
    description = Text(analyzer='ik_max_word')  #知识库描述
    summary = Text(analyzer='ik_max_word')  #知识库总结

    class Index:
        name = 'collection'


#Documents区别于固有的Docunment
class Documents(ESDocument):
    collection_id = Keyword()  # 将字符串作为文档的 ID 存储
    type = Keyword()    #文档类型用keyword保证不分词
    path = Keyword()    #文档所在路径
    name = Text(analyzer='ik_max_word')
    chunks = Integer() #文档分片个数
    uniqid = Keyword()  #去重的唯一ID
    summary = Text(analyzer='ik_max_word')  #文档摘要
    version = Integer() #文档分片个数

    class Index:
        name = 'document'


class Embedding(ESDocument):
    document_id = Keyword()     #文件ID
    collection_id = Keyword()    #知识库ID
    chunk_index = Keyword()    #文件分片索引
    chunk_size = Integer()  #文件分片大小
    document = Text(analyzer='ik_max_word')       #分片内容
    embedding = DenseVector(dims=768, index=True, similarity="cosine")

    class Index:
        name = 'embedding'


class Bot(ESDocument):
    user_id = Keyword()  # 用户ID
    collection_id = Keyword()  # 知识库ID
    hash = Keyword()    #hash
    extra = Object()    #机器人配置信息

    class Index:
        name = 'bot'


def init():
    User.init()
    Collection.init()
    Documents.init()
    Collection.init()
    Embedding.init()
    Bot.init()


def get_user(user_id):
    user = User.get(id=user_id)
    if not user:
        raise NotFound()
    return user


def save_user(openid='', name='', **kwargs):
    s = Search(index="user").filter("term", status=0).filter("term", openid=openid)
    response = s.execute()
    if not response.hits.total.value :
        user = User(
            meta={'id': ObjID.new_id()},
            openid=openid,
            name=name,
            status = 0,
            extra=kwargs,
        )
        user.save(refresh='wait_for')
        return user
    else:
        user = User.get(id=response.hits[0].meta.id)
        extra = user.extra.to_dict() if user.extra else {}
        extra.update(kwargs)
        # 保存后，里面要用这里变量
        user.openid = openid
        user.name = name
        user.extra = extra
        user.update(
            refresh='wait_for',
            retry_on_conflict=3,
            openid=openid, name=name, extra=extra,
            modified=datetime.now(),
        )
        return user


def get_collections(user_id, page, size):
    s = Search(index="collection").filter(
        "term", user_id=user_id,
    ).filter(
        "term", status=0,
    ).extra(from_=page*size-size, size=size)
    # 执行查询
    response = s.execute()
    total = response.hits.total.value
    # 返回搜索结果（文档实例的列表）
    if total == 0:
        return  [],0
    return list(response), total


def get_collection_by_id(user_id, collection_id):
    collection = Collection.get(id=collection_id)
    if collection and user_id:
        if collection.user_id == user_id:
            return collection
        else:
            return None
    else:
        return collection


def save_collection(user_id, name, description, collection_id=None):
    collection_id = collection_id or ObjID.new_id()
    collection = Collection(
        meta={'id': collection_id},
        user_id=user_id,
        name=name,
        description=description,
        summary='',
        status=0,
        created=datetime.now(),
        modified=datetime.now(),
    )
    collection.save(refresh='wait_for')
    return collection_id


def get_relation_count_by_id(index, **kwargs):
    s = Search(index=index).extra(from_=0, size=0)
    for k, v in kwargs.items():
        s = s.filter("term", **{k: v})
    response = s.execute()
    return response.hits.total.value


def update_collection_by_id(user_id, collection_id, name, description):
    collection = get_collection_by_id(user_id, collection_id)
    if not collection:
        raise NotFound('collection not found')
    collection.name = name
    collection.description = description
    collection.save(refresh='wait_for')


def delete_collection_by_id(user_id, collection_id):
    collection = get_collection_by_id(user_id, collection_id)
    if not collection:
        raise NotFound('collection not found')
    collection.status = -1
    collection.save(refresh='wait_for')
    bots = Search(index="bot").filter("term", collection_id=collection_id).execute()
    for b in bots:
        bot = Bot.get(id=b.meta.id)
        bot.collection_id = ''
        bot.save(refresh='wait_for')


def get_document_id_by_uniqid(collection_id, uniqid):
    s = Search(index="document").filter(
        "term",
        collection_id=collection_id,
    ).filter(
        "term",
        uniqid=uniqid,
    ).filter(
        "term",
        status=0
    )
    response = s.execute()
    if response.hits.total.value == 0:
        return None  # 这里不抛出异常
        # raise NotFound()
    # 这里的格式是需要数组
    return [response[0].meta.id]


def get_documents_by_collection_id(user_id, collection_id, page, size):
    collection = get_collection_by_id(user_id, collection_id)
    assert collection, '找不到对应知识库'
    s = Search(index="document").filter("term", collection_id=collection_id).filter("term", status=0).extra(from_=page*size-size, size=size).sort({"created": {"order": "desc"}})
    response = s.execute()
    total = response.hits.total.value
    # 返回搜索结果（文档实例的列表）
    if total == 0:
        return [], 0
    return list(response), total


def remove_document_by_id(user_id, collection_id, document_id):
    collection = get_collection_by_id(user_id, collection_id)
    assert collection, '找不到对应知识库'
    doc = Documents.get(id=document_id)
    if doc:
        doc.status=-1
        doc.save(refresh='wait_for')
    embeddings = Search(index='embedding').filter("term", document_id=document_id).execute()
    for e in embeddings:
        embedding = Embedding.get(id=e.meta.id)
        embedding.update(status=-1)


def purge_document_by_id(document_id):
    doc = Documents.get(id=document_id)
    if doc:
        doc.delete()
    embeddings = Search(index='embedding').filter("term", document_id=document_id).execute()
    for e in embeddings:
        embedding = Embedding.get(id=e.meta.id)
        embedding.delete()


def set_document_summary(document_id, summary):
    doc = Documents.get(id=document_id)
    if doc:
        doc.summary = summary
        doc.save(refresh='wait_for')


def get_document_by_id(document_id):
    doc = Documents.get(id=document_id)
    return doc if doc else None


def save_document(collection_id, name, url, chunks, type, uniqid=None, version=0, document_id=None):
    did = document_id or ObjID.new_id()
    doc = Documents(
        meta={'id': did},
        collection_id=collection_id,
        type=type,
        name=name,
        path=url,
        chunks=chunks,
        uniqid=uniqid,
        summary='',
        version=version,
        status=0,
        created=datetime.now(),
        modified=datetime.now(),
    )
    doc.save(refresh='wait_for')
    return did

def save_embedding(collection_id, document_id, chunk_index, chunk_size, document, embedding):
    eid = ObjID.new_id()
    embedding = Embedding(
        meta={'id': eid},
        collection_id=collection_id,
        document_id=document_id,
        chunk_index=chunk_index,
        chunk_size=chunk_size,
        document=document,
        embedding=embedding,
        status=0,
        created=datetime.now(),
        modified=datetime.now(),
    )
    embedding.save()
    return eid

def get_bot_list(user_id, collection_id):
    s = Search(index="bot").filter(
        "term",
        user_id=user_id,
        collection_id=collection_id,
    ).filter(
        "terms",
        status=[0, 1]
    ).sort({"created": {"order": "desc"}})
    response = s.execute()
    total = response.hits.total.value
    # 返回搜索结果（文档实例的列表）
    if total == 0:
        return [], 0
    return list(response), total



def get_bot_by_hash(hash):
    bot = Search(index="bot").filter(
        "term",
        hash=hash,
    ).execute()
    if bot.hits.total.value == 0:
        raise NotFound()
    return bot[0]


def get_bot_by_hash(hash):
    bot = Search(index="bot").filter(
        "term",
        hash=hash,
    ).execute()
    if bot.hits.total.value == 0:
        raise NotFound()
    return bot[0]

def query_by_collection_id(collection_id, q, page, size, delta=None):
    from tasks import embed_query
    embed = embed_query(q)
    filter_ = [{
        "term": { "collection_id": collection_id },
    }, {
        "term": { "status": 0 },
    }]
    return _query_by_filter_and_embed(
        q, filter_, embed, page, size,
        delta=float(app.config['DELTA'] if delta is None else delta),
    )

def _query_by_filter_and_embed(q, filter_, embed, page, size, delta=0.5):
    s = Search().from_dict({
        "query": {
            "bool": {
                "must": [
                    {
                        "match": {
                            "document": {
                                "query": q,
                                "boost": 0.00001
                            }
                        },
                    },
                    {
                        "bool":{
                            "filter": filter_
                        }
                    }
                ]
            }
        },
        "knn": {
            "filter": filter_,
            "field": "embedding",
            "query_vector": embed,
            "k": size,
            "num_candidates": 50,
            "boost": 1.0
        },
        "size": size * 2,
        "explain": True,
        "_source": ["document", "document_id", "collection_id"],
    }).index("embedding")

    # app.logger.info("debug knn Search %r", s.to_dict())
    response = s.execute()
    total = response.hits.total.value
    """
    1. 如果关键词能命中topk以外的, total > k
       如果命中的比较多，total > k * 2
    2. 如果关键词不能命中（或者关键词命中的刚好在topk里面）  total == k
    """
    # TODO 排序+格式化
    response_map = {i.meta.id: i for i in response}
    # TODO 模拟排序
    explanation = []
    min, max = 10000, 0
    for i in response:
        details = i.meta.explanation.details
        topk = list(filter(lambda d: 'top k' in d.description, details))
        topk = topk[0].value if len(topk) > 0 else 0
        match = list(filter(lambda d: 'sum of' in d.description, details))
        match = match[0].value if len(match) > 0 else 0
        min = match if match < min else min
        max = match if match > max else max
        explanation.append({
            'id': i.meta.id,
            'topk': topk,
            'match': match,
        })

    for exp in explanation:
        exp['stand_score'] = (exp['match'] - min) / ((max - min) or 1)
        exp['score'] = delta * exp['topk'] + (1 - delta) * exp['stand_score']

    explanation = sorted(explanation, key=lambda e: e["score"], reverse=True)
    app.logger.info("debug score %r", explanation)
    result = []
    for exp in explanation:
        item = response_map[exp['id']]
        result.append((item, exp['score']))
    
    return result[:size], total

def get_collection_id_by_hash(hash):
    bot = Search(index="bot").filter(
        "term",
        hash=hash,
    ).filter(
        "term",
        status=0
    ).execute()
    try:
        if bot.hits.total.value == 0:
            raise NotFound()
        user = get_user(bot[0].user_id)
        extra = user.extra.to_dict()
        expires = extra.get('exp_time', extra.get('permission', {}).get('expires', 0))
        privilege = extra.get('active', extra.get('permission', {}).get('has_privilege', False))
        if privilege and int(expires or 0) > time():
            return bot[0].collection_id
    except Exception as e:
        return None

def get_hash_by_collection_id(collection_id):
    bot = Search(index="bot").filter(
        "term",
        collection_id=collection_id,
    ).filter(
        "range",
        status={"gte": 0}
    ).execute()
    if bot.hits.total.value == 0:
        raise NotFound()
    return bot[0].hash

def get_data_by_hash(hash, json):
    bot = Search(index="bot").filter(
        "term",
        hash=hash,
    ).filter(
        "range",
        status={"gte": 0}
    ).execute()
    if bot.hits.total.value == 0:
        raise NotFound()
    extra = bot[0].extra.to_dict()
    if extra:
        messages = json.get('messages', [])
        if extra.get('prompt', ''):
            messages = [{
                'role': 'system',
                'content': extra.get('prompt', '')
            }] + messages
        json.update(
            model=extra.get('model', 'gpt-3.5-turbo'),
            temperature=extra.get('temperature', 0.7),
            messages=messages,
        )
    return json

def create_bot(user_id, collection_id, **extra):
    hash = str(uuid4())
    bot = Bot(
        meta={'id': ObjID.new_id()},
        user_id=user_id,
        collection_id=collection_id,
        hash=hash,
        extra=extra,
        status=1,  # 启用
        created=datetime.now(),
        modified=datetime.now(),
    )
    bot.save(refresh='wait_for')
    return hash

def update_bot_by_hash(hash, action='', collection_id='', **extra):
    bot = Search(index="bot").filter(
        "term",
        hash=hash,
    ).filter(
        "range",
        status={"gte": 0}
    ).execute()
    if bot.hits.total.value == 0:
        raise NotFound()
    bot = bot[0]
    # 如果是action，可以只传一个参数。
    # 更新的时候和前面的创建接口使用类似的参数。
    # action=start/stop/remove/refresh
    if action == 'refresh':
        hash = str(uuid4())
        bot.hash = hash
        bot.save(refresh='wait_for')
        return hash
    elif action:
        if 'start' == action:
            status = 1
        elif 'remove' == action:
            status = -1
        else:
            status = 0
        bot.status = status
        bot.save(refresh='wait_for')
        return status
    elif collection_id:
        bot.collection_id = collection_id
        bot.extra = extra
        bot.save(refresh='wait_for')
        return True

def query_by_document_id(document_id, q, page, size, delta=None):
    from tasks import embed_query
    embed = embed_query(q)
    filter_ = [{
        "term": { "document_id": document_id },
    }, {
        "term": { "status": 0 },
    }]
    return _query_by_filter_and_embed(
        q, filter_, embed, page, size,
        delta=float(app.config['DELTA'] if delta is None else delta),
    )

def get_docs_by_document_id(document_id, page, size):
    s = Search(index="embedding").filter(
        "term", document_id=document_id
    ).filter(
        "term", status=0
    ).extra(from_=page*size-size, size=size).sort({"chunk_index": {"order": "asc"}})
    # 执行查询
    response = s.execute()
    total = response.hits.total.value
    # 返回搜索结果（文档实例的列表）
    if total == 0:
        return  [], 0
    return [(i, i.meta.score) for i in response], total


class Retriever(BaseRetriever):
    collection_id: str = ''
    similarity: float = 0
    limit: int = 4

    def get_relevant_documents(self, query: str):
        """Get texts relevant for a query.

        Args:
            query: string to find relevant texts for

        Returns:
            List of relevant documents
        """
        documents, total = query_by_collection_id(self.collection_id, query, 1, self.limit)
        # documents = list(filter(lambda i: i[1] > similarity, documents))
        app.logger.info("debug Documents %r", [documents, total, self.collection_id, query])
        app.logger.info("debug Documents %r", [(d.document, distance) for d, distance in documents])
        # 这里
        return [Document(
            page_content=document.document,
            metadata={
                'collection_id': self.collection_id,
                'document_id': document.document_id,
                'document_name': document.document_name,
                'distance': distance,
            }
        ) for document, distance in documents]

    async def aget_relevant_documents(self, query):
        # 这里不用真的异步
        return self.get_relevant_documents(query)


def chat_on_collection(
    collection_id,
    deployment_name=None,
    on_llm_new_token=None, stream=False,
    temperature=0.7,
    similarity=0.8,
    limit=4,
    messages=list(), **kwargs,
):
    retriever = Retriever(collection_id=collection_id, limit=limit)
    # TODO 这里需要一个简介的输出，可能需要调整模板
    system_template = """Use the following context to answer the user's question.
-----------
{{context}}
-----------
Question: {{question}}
Helpful Answer:"""

    assert len(messages) > 0, '问题为空'
    # 从原始的messages里面读取system message
    # TODO 如果用户传了system role，可能会被丢弃？
    # 取出除了system role之外的消息
    system_message = list(filter(lambda m: m.get('role') == 'system', messages))
    chat_history = list(filter(lambda m: m.get('role') != 'system', messages[:-1]))
    # 最后一条消息是提问消息
    assert messages[-1].get('role') == 'user', '问题为空'
    question = messages[-1].get('content')

    # 2023-06-22 客户想要使用system message，我们放弃使用langchain内置的prompt模式。而是直接更改chat_history，将context放进去就可以了?
    # 构建初始 messages 列表，这里可以理解为是 openai 传入的 messages 参数
    if len(system_message) > 0:
        chat_messages = [
          SystemMessagePromptTemplate.from_template(system_message[0]['content'], template_format="jinja2"),
          HumanMessagePromptTemplate.from_template(system_template, template_format="jinja2"),
        ]
    else:
        chat_messages = [
          SystemMessagePromptTemplate.from_template(system_template, template_format="jinja2"),
        ]

    # qa()函数传chat_history不起作用，将history放到prompt里面
    for m in chat_history:
        if m['role'] == 'assistant':
            message = AIMessagePromptTemplate.from_template(m['content'], template_format="jinja2")
        else:
            message = HumanMessagePromptTemplate.from_template(m['content'], template_format="jinja2")
        chat_messages.append(message)

    chat_messages.append(HumanMessagePromptTemplate.from_template('{{question}}', template_format="jinja2"))

    # 初始化 prompt 对象
    prompt = ChatPromptTemplate.from_messages(chat_messages)

    # 问答API一些配置信息
    app.config.setdefault('OPENAI_API_KEY', None)
    app.config.setdefault('OPENAI_API_BASE', None)
    app.config.setdefault('OPENAI_API_PROXY', '')
    openai_api_key = app.config['OPENAI_API_KEY']
    openai_api_base = app.config['OPENAI_API_BASE']
    openai_proxy = app.config['OPENAI_API_PROXY']
    # 这个是azure的版本号
    app.config.setdefault('OPENAI_API_VERSION', '2023-03-15-preview')
    openai_api_version = app.config['OPENAI_API_VERSION']

    params = deepcopy(kwargs)
    params.update(
        temperature=temperature,
        verbose=True,  # 调试信息
    )
    # 初始化问答API
    if stream:
        # 如果开启了stream，就增加一个callback_manager
        class StreamingCallback(BaseCallbackHandler):
            def on_llm_new_token(self, token: str, **kwargs):
                if on_llm_new_token:
                    on_llm_new_token(token)

        params.update(
            streaming=True,  # openai的接口使用的是stream，但是ChatOpenAI的参数名称是streaming
            callback_manager=CallbackManager([StreamingCallback()]),
            openai_api_key=openai_api_key,
            openai_api_base=openai_api_base,
            openai_proxy=openai_proxy,
        )
    else:
        params.update(
            openai_api_key=openai_api_key,
            openai_api_base=openai_api_base,
            openai_proxy=openai_proxy,
        )
    # Azure
    if deployment_name:
        params.update(
            deployment_name=deployment_name,
            openai_api_version=openai_api_version,
        )
        chat = AzureChatOpenAI(**params)
    else:
        chat = ChatOpenAI(**params)

    # 初始化问答链
    # qa = ConversationalRetrievalChain.from_llm(
    #     chat, retriever,
    #     # prompt=prompt,
    # )
    # result = qa({'question': question, 'chat_history': chat_history})
    # qa = RetrievalQA.from_chain_type(
    #     llm=chat,
    #     # prompt=prompt,  # chain_type_kwargs
    #     chain_type="stuff",
    #     retriever=retriever,
    #     return_source_documents=True,
    # )
    # 自定义模板
    qa = RetrievalQA.from_llm(
        llm=chat,
        prompt=prompt,  # chain_type_kwargs
        retriever=retriever,
        return_source_documents=True,
    )
    with get_openai_callback() as cb:
        result = qa({'query': question})
        app.logger.info("cb %r", cb)
        result['usage'] = {
            'prompt_tokens': cb.prompt_tokens,
            'completion_tokens': cb.completion_tokens,
            'total_tokens': cb.total_tokens,
        }
    # 这里的result['answer']就是回答
    return result


