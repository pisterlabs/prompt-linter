import os
import logging
import requests
from datetime import datetime
from hashlib import md5
from langchain.schema import Document
from models import get_user, get_collection_by_id, Search, purge_document_by_id, get_document_id_by_uniqid
from tasks import (
    celery,
    SitemapLoader, LOADER_MAPPING,
    NamedTemporaryFile,
    embedding_single_document, get_status_by_id, embed_query,
    LarkWikiLoader,
    LarkDocLoader,
    YuqueDocLoader,
    NotionDocLoader,
)


@celery.task()
def embed_documents(fileUrl, fileType, fileName, collection_id, openai=False, uniqid=None):
    # 从一个url获取文档，并且向量化
    # assert fileType in ['pdf', 'word', 'excel', 'markdown', 'ppt', 'txt', 'sitemap']
    uniqid = uniqid or md5(fileUrl.encode()).hexdigest()
    document_ids = []

    if fileType == 'sitemap':
        sitemap_loader = SitemapLoader(web_path=fileUrl)
        docs = sitemap_loader.load()
        for doc in docs:
            document_id = embedding_single_document(doc, fileUrl, fileType, fileName, collection_id, openai=openai, uniqid=uniqid)
            document_ids.append(document_id)

    elif fileType in ['feishudoc']:
        # 飞书文件导入
        collection = get_collection_by_id(None, collection_id)
        user = get_user(collection.user_id)
        extra = user.extra.to_dict()
        client = extra.get('client', {})
        loader = LarkDocLoader(fileUrl, None, **client)
        doc = loader.load()
        document_id = embedding_single_document(
            doc, fileUrl, fileType,
            doc.metadata.get('title'),
            collection_id,
            openai=openai,
            uniqid=doc.metadata.get('document_id'),
            version=doc.metadata.get('revision_id'),  # 当前只有飞书文档需要更新版本
        )
        document_ids.append(document_id)
    elif fileType in ['yuque']:
        # 语雀文件导入
        collection = get_collection_by_id(None, collection_id)
        user = get_user(collection.user_id)
        extra = user.extra.to_dict()
        yuque = extra.get('yuque', {})
        loader = YuqueDocLoader(fileUrl, **yuque)
        doc = loader.load()
        document_id = embedding_single_document(
            doc, fileUrl, fileType,
            doc.metadata.get('title'),
            collection_id,
            openai=openai,
            uniqid=doc.metadata.get('uniqid'),
            version=0,
        )
        document_ids.append(document_id)

    elif fileType in ['notion']:
        # notion文件导入
        collection = get_collection_by_id(None, collection_id)
        user = get_user(collection.user_id)
        extra = user.extra.to_dict()
        notion = extra.get('notion', {})
        loader = NotionDocLoader(fileUrl, **client)
        doc = loader.load()
        document_id = embedding_single_document(
            doc, fileUrl, fileType,
            doc.metadata.get('title'),
            collection_id,
            openai=openai,
            uniqid=doc.metadata.get('uniqid'),
            version=0
        )
        document_ids.append(document_id)


    elif fileType in ['pdf', 'word', 'excel', 'markdown', 'ppt', 'txt']:
        loader_class, loader_args = LOADER_MAPPING[fileType]
        # 全是文件，需要下载，再加载
        with NamedTemporaryFile(delete=False) as f:
            f.write(requests.get(fileUrl).content)
            f.close()
            loader = loader_class(f.name, **loader_args)
            docs = loader.load()
            os.unlink(f.name)
            # 这里只有单个文件
            merged_doc = Document(page_content='\n'.join([d.page_content for d in docs]), metadata=docs[0].metadata)
            document_id = embedding_single_document(merged_doc, fileUrl, fileType, fileName, collection_id, openai=openai, uniqid=uniqid)
            document_ids.append(document_id)

    return document_ids


@celery.task()
def embed_feishuwiki(collection_id, openai=False):
    # TODO 同步飞书知识库
    # 1. 获取documents列表
    # 2. 获取wiki中nodes列表
    # 3. 对比，然后区分移除或者新增文件，其中新增的添加到知识库，移除的就移除知识库
    collection = get_collection_by_id(None, collection_id)
    user = get_user(collection.user_id)
    extra = user.extra.to_dict()
    client = extra.get('client', {})
    loader = LarkWikiLoader(collection.space_id, **client)
    nodes = loader.get_nodes()
    current_document_ids = set([node['obj_token'] for node in loader.get_nodes()])
    logging.info("debug current_document_ids %r", current_document_ids)
    response = Search(index="document").filter(
        "term", type="feishudoc"
    ).filter(
        "term", status=0,
    ).filter(
        "term", collection_id=collection_id,
    ).extra(
        from_=0, size=10000
    ).sort({"modified": {"order": "desc"}}).execute()
    exists_document_ids = set([doc.uniqid for doc in response])
    logging.info("debug exists_document_ids %r", exists_document_ids)
    new_document_ids = current_document_ids - exists_document_ids
    deleted_document_ids = exists_document_ids - current_document_ids
    for uniqid in exists_document_ids:
        try:
            # 由于是定时任务，可能导致重复插入，检查出现重复的，移除掉
            document_ids = [doc.meta.id for doc in response if doc.uniqid == uniqid]
            if len(document_ids) > 1:
                for document_id in document_ids[:-1]:
                    purge_document_by_id(document_id)
        except Exception as e:
            logging.error(e)
    for document_id in deleted_document_ids:
        try:
            document_ids = get_document_id_by_uniqid(collection_id, document_id)
            for document_id in document_ids:
                purge_document_by_id(document_id)
        except Exception as e:
            logging.error(e)

    for document_id in new_document_ids:
        task = embed_documents.delay(
            f'https://feishu.cn/docx/{document_id}',
            'feishudoc',
            '', collection_id, False, uniqid=document_id
        )
        logging.info("debug add new document %r %r", document_id, task)
    return list(new_document_ids), list(exists_document_ids)


@celery.task()
def sync_feishuwiki(openai=False):
    collection_ids = []
    response = Search(index="collection").filter(
        "term", type="feishuwiki"
    ).filter(
        "term", status=0,
    ).extra(
        from_=0, size=10000
    ).sort({"modified": {"order": "desc"}}).execute()
    total = response.hits.total.value
    logging.info("debug sync_feishuwiki %r", total)
    for collection in response:
        collection_ids.append(collection.meta.id)
        task = embed_feishuwiki.delay(collection.meta.id, openai)
        logging.info("debug sync_feishuwiki collection %r %r", collection.meta.id, task)
    return collection_ids


@celery.task()
def sync_feishudoc(openai=False):
    document_ids = []
    response = Search(index="document").filter(
        "term", type="feishudoc"
    ).filter(
        "term", status=0,
    ).extra(
        from_=0, size=10000
    ).sort({"modified": {"order": "desc"}}).execute()
    total = response.hits.total.value
    logging.info("debug sync_feishudoc %r", total)
    for document in response:
        try:
            collection = get_collection_by_id(None, document.collection_id)
            user = get_user(collection.user_id)
            extra = user.extra.to_dict()
            client = extra.get('client', {})
            loader = LarkDocLoader(document.path, document_id=document.uniqid, **client)
            logging.info("debug version %r %r", document.path, loader.version)
            if hasattr(document, 'version') and loader.version > document.version:
                doc = loader.load()
                document_id = embedding_single_document(
                    doc, document.path, document.type,
                    doc.metadata.get('title'),
                    document.collection_id,
                    openai=openai,
                    uniqid=doc.metadata.get('document_id'),
                    version=doc.metadata.get('revision_id'),  # 当前只有飞书文档需要更新版本
                )
                document_ids.append(document_id)
                # 移除旧文档
                purge_document_by_id(document.meta.id)
        except Exception as e:
            logging.error('error to sync_feishudoc %r %r', document.path, e)

    logging.info("updated document_ids %r", document_ids)


@celery.task()
def sync_yuque(openai=False):
    document_ids = []
    response = Search(index="document").filter(
        "term", type="yuque"
    ).filter(
        "term", status=0,
    ).extra(
        from_=0, size=10000
    ).sort({"modified": {"order": "desc"}}).execute()
    total = response.hits.total.value
    logging.info("debug sync_yuque %r", total)
    for document in response:
        try:
            collection = get_collection_by_id(None, document.collection_id)
            user = get_user(collection.user_id)
            extra = user.extra.to_dict()
            yuque = extra.get('yuque', {})
            loader = YuqueDocLoader(document.path, **yuque)
            # 没有版本号，先load一遍，再按时间判断是否重新向量化入库
            doc = loader.load()
            if doc.metadata.get('modified') > datetime.fromisoformat(document.modified):
                document_id = embedding_single_document(
                    doc, document.path, document.type,
                    doc.metadata.get('title'),
                    document.collection_id,
                    openai=openai,
                    uniqid=doc.metadata.get('uniqid'),
                    version=0,  # 当前只有飞书文档需要更新版本
                )
                document_ids.append(document_id)
                # 移除旧文档
                purge_document_by_id(document.meta.id)
        except Exception as e:
            logging.error('error to sync_yuque %r %r', document.path, e)

    logging.info("updated document_ids %r", document_ids)

@celery.task()
def sync_notion(openai=False):
    document_ids = []
    response = Search(index="document").filter(
        "term", type="notion"
    ).filter(
        "term", status=0,
    ).extra(
        from_=0, size=10000
    ).sort({"modified": {"order": "desc"}}).execute()
    total = response.hits.total.value
    logging.info("debug sync_notion %r", total)
    for document in response:
        try:
            collection = get_collection_by_id(None, document.collection_id)
            user = get_user(collection.user_id)
            extra = user.extra.to_dict()
            notion = extra.get('notion', {})
            loader = NotionDocLoader(document.path, **notion)
            # 没有版本号，先load一遍，再按时间判断是否重新向量化入库
            doc = loader.load()
            if doc.metadata.get('modified') > document.modified:
                document_id = embedding_single_document(
                    doc, document.path, document.type,
                    doc.metadata.get('title'),
                    document.collection_id,
                    openai=openai,
                    uniqid=doc.metadata.get('uniqid'),
                    version=0,  # 当前只有飞书文档需要更新版本
                )
                document_ids.append(document_id)
                # 移除旧文档
                purge_document_by_id(document.meta.id)
        except Exception as e:
            logging.error('error to sync_notion %r %r', document.path, e)

    logging.info("updated document_ids %r", document_ids)
