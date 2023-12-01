
import pickle
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
import os
import numpy as np
import faiss
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.base import AddableMixin, Docstore
from langchain.docstore.document import Document
from langchain.docstore.in_memory import InMemoryDocstore
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.base import VectorStore
from langchain.vectorstores.utils import maximal_marginal_relevance
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

from langchain import PromptTemplate, LLMChain


def dependable_faiss_import() -> Any:
    """Import faiss if available, otherwise raise error."""
    try:
        import faiss
    except ImportError:
        raise ValueError(
            "Could not import faiss python package. "
            "Please it install it with `pip install faiss` "
            "or `pip install faiss-cpu` (depending on Python version)."
        )
    return faiss


class FAISS(VectorStore):
    """Wrapper around FAISS vector database.
    To use, you should have the ``faiss`` python package installed.
    Example:
        .. code-block:: python
            from langchain import FAISS
            faiss = FAISS(embedding_function, index, docstore)
    """

    def __init__(
        self,
        embedding_function: Callable,
        index: Any,
        docstore: Docstore,
        index_to_docstore_id: Dict[int, str],
    ):
        """Initialize with necessary components."""
        self.embedding_function = embedding_function
        self.index = index
        self.docstore = docstore
        self.index_to_docstore_id = index_to_docstore_id

    def add_texts(
        self, texts: Iterable[str], metadatas: Optional[List[dict]] = None
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.
        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        if not isinstance(self.docstore, AddableMixin):
            raise ValueError(
                "If trying to add texts, the underlying docstore should support "
                f"adding items, which {self.docstore} does not"
            )
        # Embed and create the documents.
        embeddings = [self.embedding_function(text) for text in texts]
        documents = []
        for i, text in enumerate(texts):
            metadata = metadatas[0] if metadatas else {}
            documents.append(Document(page_content=text, metadata=metadata))
        # Add to the index, the index_to_id mapping, and the docstore.
        starting_len = len(self.index_to_docstore_id)
        self.index.add(np.array(embeddings, dtype=np.float32))
        # Get list of index, id, and docs.
        full_info = [
            (starting_len + i, str(uuid.uuid4()), doc)
            for i, doc in enumerate(documents)
        ]
        # Add information to docstore and index.
        self.docstore.add({_id: doc for _, _id, doc in full_info})
        index_to_id = {index: _id for index, _id, _ in full_info}
        self.index_to_docstore_id.update(index_to_id)
        return [_id for _, _id, _ in full_info]

    def similarity_search_with_score_by_vector(
        self, embedding: List[float], k: int = 4
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query.
        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
        Returns:
            List of Documents most similar to the query and score for each
        """
        scores, indices = self.index.search(np.array([embedding], dtype=np.float32), k)
        docs = []
        for j, i in enumerate(indices[0]):
            if i == -1:
                # This happens when not enough docs are returned.
                continue
            _id = self.index_to_docstore_id[i]
            doc = self.docstore.search(_id)
            # if not isinstance(doc, Document):
            #     raise ValueError(f"Could not find document for id {_id}, got {doc}")
            docs.append((doc, scores[0][j]))
        return docs

    def similarity_search_with_score(
        self, query: str, k: int = 4
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query.
        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
        Returns:
            List of Documents most similar to the query and score for each
        """
        embedding = self.embedding_function(query)
        docs = self.similarity_search_with_score_by_vector(embedding, k)
        return docs

    def similarity_search_by_vector(
        self, embedding: List[float], k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Return docs most similar to embedding vector.
        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
        Returns:
            List of Documents most similar to the embedding.
        """
        docs_and_scores = self.similarity_search_with_score_by_vector(embedding, k)
        return [doc for doc, _ in docs_and_scores]

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Return docs most similar to query.
        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
        Returns:
            List of Documents most similar to the query.
        """
        docs_and_scores = self.similarity_search_with_score(query, k)
        return [doc for doc, _ in docs_and_scores]

    def max_marginal_relevance_search_by_vector(
        self, embedding: List[float], k: int = 4, fetch_k: int = 20
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.
        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.
        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        _, indices = self.index.search(np.array([embedding], dtype=np.float32), fetch_k)
        # -1 happens when not enough docs are returned.
        embeddings = [self.index.reconstruct(int(i)) for i in indices[0] if i != -1]
        mmr_selected = maximal_marginal_relevance(
            np.array([embedding], dtype=np.float32), embeddings, k=k
        )
        selected_indices = [indices[0][i] for i in mmr_selected]
        docs = []
        for i in selected_indices:
            if i == -1:
                # This happens when not enough docs are returned.
                continue
            _id = self.index_to_docstore_id[i]
            doc = self.docstore.search(_id)
            # if not isinstance(doc, Document):
            #     raise ValueError(f"Could not find document for id {_id}, got {doc}")
            docs.append(doc)
        return docs

    def max_marginal_relevance_search(
        self, query: str, k: int = 4, fetch_k: int = 20
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.
        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.
        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        embedding = self.embedding_function(query)
        docs = self.max_marginal_relevance_search_by_vector(embedding, k, fetch_k)
        return docs

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) :
        """Construct FAISS wrapper from raw documents.
        This is a user friendly interface that:
            1. Embeds documents.
            2. Creates an in memory docstore
            3. Initializes the FAISS database
        This is intended to be a quick way to get started.
        Example:
            .. code-block:: python
                from langchain import FAISS
                from langchain.embeddings import OpenAIEmbeddings
                embeddings = OpenAIEmbeddings()
                faiss = FAISS.from_texts(texts, embeddings)
        """
        faiss = dependable_faiss_import()
        embeddings = embedding.embed_documents(texts)
        index = faiss.IndexFlatL2(len(embeddings[0]))
        index.add(np.array(embeddings, dtype=np.float32))
        documents = []
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas else {}
            documents.append(Document(page_content=text, metadata=metadata))
        index_to_id = {i: str(uuid.uuid4()) for i in range(len(documents))}
        docstore = InMemoryDocstore(
            {index_to_id[i]: doc for i, doc in enumerate(documents)}
        )
        return cls(embedding.embed_query, index, docstore, index_to_id)

    def save_local(self, folder_path: str) -> None:
        """Save FAISS index, docstore, and index_to_docstore_id to disk.
        Args:
            folder_path: folder path to save index, docstore,
                and index_to_docstore_id to.
        """
        path = Path(folder_path)
        path.mkdir(exist_ok=True, parents=True)

        # save index separately since it is not picklable
        faiss = dependable_faiss_import()
        faiss.write_index(self.index, str(path / "index.faiss"))

        # save docstore and index_to_docstore_id
        with open(path / "index.pkl", "wb") as f:
            pickle.dump((self.docstore, self.index_to_docstore_id), f)

    @classmethod
    def load_local(cls, folder_path: str, embeddings: Embeddings) :
        """Load FAISS index, docstore, and index_to_docstore_id to disk.
        Args:
            folder_path: folder path to load index, docstore,
                and index_to_docstore_id from.
            embeddings: Embeddings to use when generating queries
        """
        path = Path(folder_path)
        # load index separately since it is not picklable
        faiss = dependable_faiss_import()
        index = faiss.read_index(str(path / "index.faiss"))

        # load docstore and index_to_docstore_id
        with open(path / "index.pkl", "rb") as f:
            docstore, index_to_docstore_id = pickle.load(f)
        return cls(embeddings.embed_query, index, docstore, index_to_docstore_id)





class DocHandler:
  
 
  doc_map_suffix = "document_mapping.pkl"

  def __init__(self, save_dir,c_name):
    self.save_dir = save_dir+"/"+c_name
    # self.mdb_client = mdb_client
    # self.coll = c_name
    # self.index = self._get_index()

    self.embed_model = OpenAIEmbeddings()
    self.splitter =  TokenTextSplitter( chunk_size=200, chunk_overlap=10)
    self.docstore =  InMemoryDocstore({})
    self.index = faiss.IndexFlatL2(1536)
    self.index_to_docstore_id = {}
    self.document_mapping = {}
    self.vecstore = FAISS(self.embed_model.embed_query, self.index, self.docstore, self.index_to_docstore_id)
    # self.index_to_docstore_id = self.vecstore.index_to_docstore_id
    save_path_bool = Path(self.save_dir).mkdir(exist_ok=True)
    index_path_bool = os.path.isfile(self.save_dir+'/index.faiss')
   
    if not save_path_bool and  index_path_bool:
      print("Loading previous vector store")
      self._reload()

    else:
      
      print(f"Initializing DocHandler Obj with save to {self.save_dir}")
      # self._initialize()
      # self._update()
 
  def update(self, filename:str, text:str=''):
    if filename in self._get_all_indexed_filenames():
      self._remove_document(filename)
      if not text=='':
        self.add_document(filename,text)
      else:
        print("Deleted file, since no text was set")
    return

  def _remove_document(self, filename): # delete filename. Make sure it exists in db
    if filename not in self._get_all_indexed_filenames():
      print(f"File with filename {filename}, does not exists inside index ! WIll not do anything")
      return None
    print(f"Removing document {filename}")
    assert len(self.document_mapping) > 0 , "No files to remove"
    
    # filter function
    def my_filtering_function(pair):
      key, value = pair
      if value['filename'] == filename:
          return True # filter pair out of the dictionary
      else:
          return False # keep pair in the filtered dictionary
    file_dict = dict(filter(my_filtering_function, self.document_mapping.items()))
    doc_mapping = list(file_dict.values())[0]['mapping']
    doc_id = list(file_dict.keys())[0]
    str_ids,ids = list(doc_mapping.keys()), list(doc_mapping.values())
    # ids need to be shifted 
    new_str_to_ids = {j:i for i,j in self.vecstore.index_to_docstore_id.items() if j not in str_ids}
    # remove from index
    action =  self.index.remove_ids(np.array(ids))
    print(action)
    if action > 0 :
   
      for i,j in zip(str_ids,ids):
        # print(i)
        jk = self.vecstore.docstore.pop(i)      # docstore value removal
    else:
      print(f"Didnt delete {filename}")
      #action not sucessful 
      return 

    all_keys = list(self.document_mapping.keys())
    # print("all keys before delete ", all_keys)
    del self.document_mapping[doc_id]

    self.vecstore.index_to_docstore_id = {e:k[0] for e,k in enumerate(new_str_to_ids.items())}
    new_str_to_ids = {j:i for i,j in self.vecstore.index_to_docstore_id.items()}

    # update document mapping
    all_keys = list(self.document_mapping.keys())

    for e,i in enumerate(all_keys):
        if e != i:
          old_mapping = list(self.document_mapping[i]['mapping'].keys())
          filename = list(self.document_mapping[i]['filename'])
          new_map = {}
          for map in old_mapping:
            prop_index = new_str_to_ids[map]
            new_map[map] = prop_index
          self.document_mapping[e] = {'filename':''.join(filename), 'mapping':new_map}
          del self.document_mapping[i]
    cur_doc_list_len = 0 if self.document_mapping == None else len(self.document_mapping.keys())
    
    print("No. of docs: ", cur_doc_list_len)
    self._save_all()
    return
    
  


  def add_document(self,filename:str, text:str):

    if filename in self._get_all_indexed_filenames():
      print(f"File with filename {filename}, already indexed ! Will not do anything")
      return None
    print(f"Adding document {filename}")
    # filepath = self.data_dir + "/" + filename
    # doc = self._load_doc_by_filepath(filepath)
    if text != None and type(text)==str:
      new_index_to_docstr= self._run_texts(text, [{'source':filename}])
      
      cur_doc_list_len = 0 if self.document_mapping == None else len(self.document_mapping.keys())
      print("No. of docs: ", cur_doc_list_len+1)
      mapping = {cur_doc_list_len: {'filename':filename,'mapping':new_index_to_docstr}}
      if type(self.document_mapping)==dict:
          self.document_mapping.update(mapping)
      else:
        self.document_mapping = mapping
      # self.index_to_docstore_id = self.vecstore.index_to_docstore_id
      self._save_all()
    # print(mapping)
    return 

  def _run_texts(self,text,meta):
    #return new ids inserted
    texts=self.splitter.split_text(text)
    new_doc_indexes = self.vecstore.add_texts(texts,meta)
    new_doc_indexes = {j:i for i,j in self.vecstore.index_to_docstore_id.items() if j in new_doc_indexes}
    return new_doc_indexes

  def _get_all_indexed_filenames(self):
    if self.document_mapping!={}:
      filenames = [i['filename'] for i in self.document_mapping.values()]
      return filenames
    else:
      return []

  def _save_all(self):
    self.vecstore.save_local(self.save_dir)

    doc_map_path = self.save_dir +"/"+self.doc_map_suffix
    with open(doc_map_path, 'wb') as b:
      pickle.dump(self.document_mapping, b)
  
  def _reload(self):
   
    doc_map_path = self.save_dir +"/"+self.doc_map_suffix

    with open(doc_map_path, 'rb') as f:
      self.document_mapping = pickle.load(f)
    self.vecstore=FAISS.load_local(self.save_dir, self.embed_model)
    # self.index_to_docstore_id = self.vecstore.index_to_docstore_id
  
  def max_marginal_relevance_search(self,query: str, k: int = 4, fetch_k: int = 20):
    return self.vecstore.max_marginal_relevance_search(query, k, fetch_k)

  def similarity_search_with_score(self, query:str, k: int = 4):
    return self.vecstore.similarity_search_with_score(query, k)

  def similarity_search(self, query: str, k: int = 4, **kwargs):
    return self.vecstore.similarity_search(query, k, **kwargs)
  
  def get_document_id(self, id:int):
    if self.document_mapping != {}:
      for i,j in self.document_mapping.items():
         indexes = list(j['mapping'].values())
         if id in indexes:
           return i, j['filename']
    
  

  
  class SearchDocstoreGPT4:
    def __init__(self, docstore,prompt=None, cherry_pick=True):
        self.docstore = docstore
        self.fetch_k = 30
        self.send_k = 15
        self.cherry_pick = cherry_pick
        self.chain = self.init_llm_chain(prompt)

 

    def run(self,query:str, k = None, scores=True):
        if len(self.get_all_filenames())>0:
            if k is None:
                docs = self.docstore.similarity_search_with_scores(query,self.send_k)
            else:
                docs = self.docstore.similarity_search_with_scores(query,k)
        
            curated_docs=""
            for doc,score in docs:
                if scores:
                    curated_docs += f"Source: {doc.metadata['source']}::\n {doc.page_content} :: Query Embedding Match Source : {score}\n\n"
                else:
                    curated_docs += f"Source: {doc.metadata['source']}::\n {doc.page_content} \n\n"
            if self.cherry_pick:
                input = {"sources":curated_docs,"query":query}
                res = self.chain.run(**input)
                return str(res)
            else: 
                return curated_docs
        else:
            return "[]::NO DOCUMENTS INDEXED"
        # return curated_docs
    def get_all_filenames(self):
        return self.docstore._get_all_indexed_filenames()

    def init_llm_chain(self,prompt):
        
        
        system_template= """
        INSTRUCTION:

        You are an Information Extractor. You Have to filter out the sources and cite information that helps answer 
        the user query. The user is knowledgeable and needs the AIs help to extract sentences from the sources that are 
        even minutely related to the user query. Provide as much usable information as you can without changing the source . 
        If the sources provided are not relevant to the user query, then say "NO SOURCES". If the SOURCES section is blank, return "No Documents"

        RESPONSE FORMAT:
        (Case 1: If relevant information extractable from sources)
        [1st source name]:: [all useful sentences];;
        [2nd source name]:: [all useful sentences];;

        (Case 2: If no relevant information is extractable from Sources)
        []:: NO INFORMATION FOUND  

        SOURCES:
        {sources}

        Begin!
        """
        messages = [
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template("Dig out information for: {query}")
        ]
        chat_prompt = ChatPromptTemplate.from_messages(messages)
        llm = ChatOpenAI(temperature=0, model='gpt-4',max_tokens=800)
        chain =  LLMChain(llm=llm, prompt=chat_prompt)

        return chain

