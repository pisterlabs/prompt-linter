from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type


from langchain.embeddings.base import Embeddings
from langchain.vectorstores.base import VectorStore
from pydantic import BaseModel, Field
from pgvector.django import L2Distance,CosineDistance,MaxInnerProduct
from customuser.models import UserProfile
from pdfpaper.models import PDFChunk, PDFFile
from paperchat.models import  ChatTransaction

class DjangoDBDocument(BaseModel):
    page_content: str 
    pdf_name: str 
    page_number: str 
    metadata: dict = Field(default_factory=dict)



class DjangoVectorStore(VectorStore):
    """Wrapper around Django vector database.

    """

    def __init__(
        self,
        userprofile: UserProfile,
        transaction: ChatTransaction,
        embedding_function: Callable,
        search_in: List = None,
        strategy: Optional(str) = "cosine",
        mode: str = "GENERAL",
        **kwargs: Any,
    ):
        """Initialize with necessary components."""
        self.embedding_function = embedding_function
        self.userprofile = userprofile
        self.transaction = transaction
        self.strategy = strategy
        self.search_in = search_in
        self.mode = mode




    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> List[DjangoDBDocument]:

        embedding = self.embedding_function.embed_query(text=query)

        return self.similarity_search_by_vector(
            embedding=embedding,
            k=k
        )
    


    def similarity_search_with_score_by_vector(
            self,
            embedding: List[float],
            k: int = 4 
    ) -> List[Tuple[DjangoDBDocument, float]]:
        
        queryset = self.get_queryset(embedding=embedding)

        res = queryset[:k]

        print(f'running query set {k}')

        print(embedding)

        result = [(DjangoDBDocument(
            page_content=x.text,
            page_number=x.page_number,
            pdf_name=x.pdf.title
        ), i) for i,x in enumerate(res)]

        print(result)
        return result

    def similarity_search_by_vector(self, embedding: List[float], k: int = 4, **kwargs: Any) -> List[DjangoDBDocument]:

        docs_and_scores = self.similarity_search_with_score_by_vector(
            embedding=embedding,
            k=k
        )

        results = [x for x,_ in docs_and_scores]

        return results


    def get_queryset(self, embedding):

        pre_base = PDFChunk.objects.filter(pdf__company__id=self.userprofile.company.id)

        if self.mode == "LAW":
            base = pre_base.filter(pdf__pdf_type="LAW") 

        if self.mode == "SCIENCE":
            base = pre_base.filter(pdf__pdf_type="SCIENCE") 

        else:
            base = pre_base
        if self.search_in is not None:
            print(self.search_in)
            print(type(self.search_in))
            ids = [x.id for x in self.search_in] if self.search_in is not None else []
            query = base.filter(pdf__id__in=ids)


        else:
            query = base

        
        if self.strategy == "cosine":

            final = query.order_by(CosineDistance('embedding', embedding))

        elif self.strategy == "l2distance":

            final = query.order_by(L2Distance('embedding', embedding))

        elif self.strategy == "maxdistance":

            final = query.order_by(MaxInnerProduct('embedding', embedding))

        else:
            final = query.order_by(CosineDistance('embedding', embedding))

        return final
    


    
    def add_texts(self, texts: Iterable[str],userprofile: UserProfile, metadatas: List[dict] | None = None, **kwargs: Any) ->  List[str]:
        title = "llm pdf texts"
        text = " ".join(texts)

        pdf = PDFFile.objects.create(title=title, text=text, company= userprofile.company)
        print(f'add texts added :  {text}')
        # created = []
        # for chunk in texts:
        #     chunkembedd = self.embedding_function.embed_documents(chunk)
        #     chu = PDFChunk.objects.create(
        #         pdf= pdf,
        #         text = chunk,
        #         embedding = chunkembedd
        #     )
        #     created.append(chu)

        # return[x.text for x in created]


    @classmethod
    def from_texts(cls, 
                   texts: List[str], 
                   embedding: Embeddings, 
                   userprofile: UserProfile,
                   transaction: ChatTransaction,
                   metadatas: List[dict] | None = None, **kwargs: Any) -> Type[DjangoVectorStore]:


        #store = 

        #store.add_texts(texts, userprofile, metadatas)
        vector_store = cls(
            userprofile=userprofile,
            transaction=transaction,
            embedding_function=embedding,  # I assume the embedding function is the one needed here
            **kwargs,
        )

        # Add the texts to the vector store
        vector_store.add_texts(texts, userprofile, metadatas)

        return vector_store
        


