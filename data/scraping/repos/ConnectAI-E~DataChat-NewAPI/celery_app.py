import os
import requests
from hashlib import md5
from langchain.schema import Document
from tasks import (
    celery,
    SitemapLoader, LOADER_MAPPING,
    NamedTemporaryFile,
    embedding_single_document, get_status_by_id, embed_query,
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


