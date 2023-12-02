import copy
from typing import Callable, Any, List, Dict, Optional

import numpy as np
from langchain.docstore.base import Docstore
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.vectorstores.base import VectorStore
from langchain.vectorstores.faiss import _default_relevance_score_fn, dependable_faiss_import

from config import *



#定义一个类，实现向量库的增删改查
class MyFAISS(FAISS,VectorStore):
	def __init__(
			self,
			embedding_function:Callable,
			index:Any,
			docstore:Docstore,
			index_to_docstore_id:Dict[int,str],
			#用于做相关性之间的匹配
			relevance_score_fn: Optional[
				Callable[[float], float]
			] = _default_relevance_score_fn,
			normalize_L2: bool = False,

	):
		super().__init__(
			embedding_function=embedding_function,
			index=index,
			docstore=docstore,
			index_to_docstore_id=index_to_docstore_id,
			relevance_score_fn=relevance_score_fn,
			normalize_L2 = normalize_L2
		)
		self.score_threshold = VECTOR_SEARCH_SCORE_THRESHOLD
		self.chunk_size = CHUNK_SIZE
		self.chunk_conent = False

	def seperate_list(self, ls: List[int]) -> List[List[int]]:
		# TODO:增加是否属于同一文档的判断
		# TODO:该函数的作用是将一个整数列表分成多个连续的子列表，其中每个子列表中的整数相邻且连续。

		lists = []
		ls1 = [ls[0]]
		for i in range(1, len(ls)):
			if ls[i - 1] + 1 == ls[i]:
				ls1.append(ls[i])
			else:
				lists.append(ls1)
				ls1 = [ls[i]]
		lists.append(ls1)
		return lists


	def similarity_search_with_score_by_vector(
			self,
			embedding:List[float],
			k:int = VECTOR_SEARCH_TOP_K,
			filter:Optional[Dict[str, Any]] = None,
			fetch_k:int=20,
			**kwargs: Any,
	)->List[Document]:
		#TODO:重写了similarity_search_with_score_by_vector方法，该方法不再返回score,增加了文档判断
		faiss = dependable_faiss_import()
		#TODO:相当于import faiss
		vector = np.array([embedding], dtype=np.float32)
		#TODO:根据embedding特征生成矩阵
		if self._normalize_L2:
			faiss.normalize_L2(vector)
		scores, indices = self.index.search(vector, k)
		print(scores)
		#cores 表示每个查询向量的相似度得分（即余弦相似度），indices 表示与每个查询向量最相似的 k 个向量的索引号。
		docs=[]
		id_set=set()
		store_len=len(self.index_to_docstore_id)
		rearrange_id_list=False
		for j,i in enumerate(indices[0]):
			if i==-1 or 0<scores[0][j]<self.score_threshold:
				#TODO:向量编号 i 取值范围为 [0, num_vectors - 1]，其中 num_vectors 表示向量索引中的向量总数。当搜索结果中的向量编号 i 为 -1 时，表示索引未能找到与查询向量相似的向量。
				continue

			if i in self.index_to_docstore_id:
				_id=self.index_to_docstore_id[i]
			else:
				continue
			doc=self.docstore.search(_id)
			if (not self.chunk_conent) or ("context_expand" in doc.metadata and not  doc.metadata["context_expand"]):
				#不需要进行上下文的扩展
				if not isinstance(doc,Document):
					raise ValueError(f"Could not find document for id {_id}, got {doc}")
				doc.metadata["score"]=int(scores[0][j])
				docs.append(doc)
				continue
			id_set.add(i)
			docs_len=len(doc.page_content)
			for k in range(1,max(i,store_len-1)):
				break_flag=False
				if "context_expand_method" in doc.metadata and doc.metadata["context_expand_method"] == "forward":
					expand_range = [i + k]
				elif "context_expand_method" in doc.metadata and doc.metadata["context_expand_method"] == "backward":
					expand_range = [i - k]
				else:
					expand_range = [i + k, i - k]
				for l in expand_range:
					if l not in id_set and 0 <= l < len(self.index_to_docstore_id):
						_id0 = self.index_to_docstore_id[l]
						doc0 = self.docstore.search(_id0)
						if docs_len + len(doc0.page_content) > self.chunk_size or doc0.metadata["source"] != \
								doc.metadata["source"]:
							break_flag = True
							break
						elif doc0.metadata["source"] == doc.metadata["source"]:
							docs_len += len(doc0.page_content)
							id_set.add(l)
							rearrange_id_list = True
				if break_flag:
					break
		if(not self.chunk_conent) or (not  rearrange_id_list):
			return docs
		if len(id_set)==0 and self.score_threshold>0:
			return []
		id_list=sorted(list(id_set))
		id_lists=self.seperate_list(id_list)
		for id_seq in id_lists:
			for id in id_seq:
				if id==id_seq[0]:
					_id=self.index_to_docstore_id[id]
					doc=copy.deepcopy(self.docstore.search(_id))
				else:
					_id0=self.index_to_docstore_id[id]
					doc0=self.docstore.search(_id0)
					doc.page_content+=" "+doc0.page_content
			if not isinstance(doc,Document):
				raise ValueError(f"Could not find document for id {_id}, got {doc}")
			doc_score=min([scores[0][id] for id in [indices[0].tolist().index(i) for i in id_seq if i in indices[0]]])
			doc.metadata["score"]=int(doc_score)
			docs.append(doc)
		return docs
	def delete_doc(self,source: str or List[str]):
		try:
			if isinstance(source,str):
				ids=[k for k,v in self.docstore._dict.item() if v.metadata["source"]==source]
				db_path=os.path.join(os.path.split(os.path.split(source)[0])[0], "vector_store")
			else:
				ids=[k for k, v in self.docstore._dict.items() if v.metadata["source"] in source]
				db_path = os.path.join(os.path.split(os.path.split(source[0])[0])[0], "vector_store")
			if len(ids)==0:
				return f"docs delete fail"
			else:
				for id in ids:
					index=list(self.index_to_docstore_id.keys())[list(self.index_to_docstore_id.values()).index((id))]
					self.index_to_docstore_id.pop(index)
					self.docstore._dict.pop(id)
				self.save_local(db_path)
				return f"docs delete success"
		except Exception as e:
			print(e)
			return f"docs delete fail"


	def update_doc(self,source,new_docs):
		try:
			self.delete_doc(source)
			self.add_documents(new_docs)
			return f"docs update success"
		except Exception as e:
			print(e)
			return f"docs update fail"
	def list_docs(self):
		#TODO:列出文档的source来源
		return list(set(v.metadata["source"] for v in self.docstore._dict.values()))


