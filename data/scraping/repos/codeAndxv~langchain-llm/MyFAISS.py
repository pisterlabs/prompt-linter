from langchain.vectorstores import FAISS
from langchain.vectorstores.base import VectorStore
from langchain.vectorstores.faiss import dependable_faiss_import
from typing import Any, Callable, List, Dict, Tuple, Optional
from langchain.docstore.base import Docstore
from langchain.docstore.document import Document
import numpy as np
import copy
import os
from configs.model_config import *


class MyFAISS(FAISS, VectorStore):
    def __init__(
            self,
            embedding_function: Callable,
            index: Any,
            docstore: Docstore,
            index_to_docstore_id: Dict[int, str],
            normalize_L2: bool = False,
    ):
        super().__init__(embedding_function=embedding_function,
                         index=index,
                         docstore=docstore,
                         index_to_docstore_id=index_to_docstore_id,
                         normalize_L2=normalize_L2)
        self.score_threshold=VECTOR_SEARCH_SCORE_THRESHOLD
        self.chunk_size = CHUNK_SIZE
        self.chunk_conent = False

    def seperate_list(self, ls: List[int]) -> List[List[int]]:
        lists = []
        ls1 = [ls[0]]
        o_source = self.docstore.search(self.index_to_docstore_id[ls[0]]).metadata["source"]
        for i in range(1, len(ls)):
            if ls[i - 1] + 1 == ls[i] and o_source == self.docstore.search(self.index_to_docstore_id[i]).metadata["source"]:
                ls1.append(ls[i])
            else:
                o_source = self.docstore.search(self.index_to_docstore_id[ls[i]]).metadata["source"]
                lists.append(ls1)
                ls1 = [ls[i]]
        lists.append(ls1)
        return lists


    def similarity_search_with_score_by_vector(
            self,
            embedding: List[float],
            k: int = 4,
            filter: Optional[Dict[str, Any]] = None,
            fetch_k: int = 20,
            **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        faiss = dependable_faiss_import()
        vector = np.array([embedding], dtype=np.float32)
        if self._normalize_L2:
            faiss.normalize_L2(vector)
        scores, indices = self.index.search(vector, k if filter is None else fetch_k)
        docs = []
        id_set = set()
        store_len = len(self.index_to_docstore_id)
        for j, i in enumerate(indices[0]):
            if i == -1 or 0 < self.score_threshold < scores[0][j]:
                # This happens when not enough docs are returned.
                continue
            _id = self.index_to_docstore_id[i]
            doc = self.docstore.search(_id)
            if (not self.chunk_conent) or ("context_expand" in doc.metadata and not doc.metadata["context_expand"]):
                # 匹配出的文本如果不需要扩展上下文则执行如下代码
                if not isinstance(doc, Document):
                    raise ValueError(f"Could not find document for id {_id}, got {doc}")
                doc.metadata["score"] = int(scores[0][j])
                docs.append((doc, int(scores[0][j])))
                continue
            else:
                id_set.add(i)
                doc_len = len(doc.page_content)
                keys = list(self.index_to_docstore_id.keys())
                def generate_index(doc_index, s_len):
                    n = 1
                    while True:
                        if doc_index + n < s_len:
                            yield doc_index + n
                        if doc_index - n >= 0:
                            yield doc_index - n
                        if doc_index + n >= s_len and doc_index - n < 0:
                            break
                        n += 1
                for next_expend_i in generate_index(keys.index(i), store_len):
                    next_expend_i = keys[next_expend_i]
                    if next_expend_i not in id_set and next_expend_i in self.index_to_docstore_id:
                        _id0 = self.index_to_docstore_id[next_expend_i]
                        doc0 = self.docstore.search(_id0)
                        if doc_len + len(doc0.page_content) > self.chunk_size or doc0.metadata["source"] != doc.metadata["source"]:
                            break
                        else:
                            doc_len += len(doc0.page_content)
                            id_set.add(next_expend_i)
        if not id_set:      # 说明不需要扩展上下文
            return docs
        id_list = sorted(list(id_set))
        id_lists = self.seperate_list(id_list)
        for id_seq in id_lists:
            for id in id_seq:
                if id == id_seq[0]:
                    _id = self.index_to_docstore_id[id]
                    tem_doc = copy.deepcopy(self.docstore.search(_id))
                else:
                    _id0 = self.index_to_docstore_id[id]
                    doc0 = self.docstore.search(_id0)
                    tem_doc.page_content += " " + doc0.page_content
            if tem_doc:
                tem_scores = [scores[0][id] for id in [indices[0].tolist().index(i) for i in id_seq if i in indices[0]]]
                if tem_scores:
                    doc_score = min(tem_scores)
                    tem_doc.metadata["score"] = int(doc_score)
                    docs.append((tem_doc, int(doc_score)))
        return docs

    def delete_doc(self, source: str or List[str]):
        try:
            if isinstance(source, str):
                ids = [k for k, v in self.docstore._dict.items() if v.metadata["source"] == source]
                vs_path = os.path.join(os.path.split(os.path.split(source)[0])[0], "vector_store")
            else:
                ids = [k for k, v in self.docstore._dict.items() if v.metadata["source"] in source]
                vs_path = os.path.join(os.path.split(os.path.split(source[0])[0])[0], "vector_store")
            if len(ids) == 0:
                return f"docs delete fail"
            else:
                for id in ids:
                    index = list(self.index_to_docstore_id.keys())[list(self.index_to_docstore_id.values()).index(id)]
                    self.index_to_docstore_id.pop(index)
                    self.docstore._dict.pop(id)
                # TODO: 从 self.index 中删除对应id
                # self.index.reset()
                self.save_local(vs_path)
                return f"docs delete success"
        except Exception as e:
            print(e)
            return f"docs delete fail"

    def update_doc(self, source, new_docs):
        try:
            delete_len = self.delete_doc(source)
            ls = self.add_documents(new_docs)
            return f"docs update success"
        except Exception as e:
            print(e)
            return f"docs update fail"

    def list_docs(self):
        return list(set(v.metadata["source"] for v in self.docstore._dict.values()))
