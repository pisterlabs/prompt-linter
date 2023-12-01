import uuid

import faiss
from langchain.docstore import InMemoryDocstore
from langchain.vectorstores.faiss import FAISS

from app.transformers.fake import TransformerInterface


class MemoryKnowledgeBaseRepository:

    def __init__(self, transformer: TransformerInterface):
        self.data = {}
        # self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
        self.model = transformer

    def create(self) -> str:
        kb_id = str(uuid.uuid4())
        index = faiss.IndexFlatL2(self.model.dimension)
        self.data[kb_id] = FAISS(embedding_function=self.model.encode, index=index, docstore=InMemoryDocstore(),
                                 index_to_docstore_id={})
        return str(kb_id)

    def delete(self, kb_id: str) -> None:
        kb = self._get_obj(str(kb_id))
        del self.data[str(kb_id)]

    def add_document(self, kb_id: str, text: str) -> str:
        kb = self._get_obj(str(kb_id))
        doc_id = kb.add_texts([text])[0]
        return doc_id

    def delete_document(self, kb_id: str, doc_id: str) -> None:
        kb = self._get_obj(str(kb_id))
        is_ok = kb.delete([doc_id])
        if not is_ok:
            raise ValueError('...')

    def get_similar(self, kb_id: str, text: str, n_similar: int):
        kb = self._get_obj(str(kb_id))
        doc_n_scores = kb.similarity_search_with_score(text, n_similar)
        res = []
        for doc, score in doc_n_scores:
            d = {"text": doc.page_content, "score": score}
            res.append(d)
        return res

    def _get_obj(self, kb_id: str):
        kb: FAISS = self.data.get(str(kb_id))
        if not kb:
            raise ValueError('KB not found')
        return kb
