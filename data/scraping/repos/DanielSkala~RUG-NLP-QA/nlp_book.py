from algorithm.ir_system import IRSystem
from algorithm.document_operator import PDFDocumentOperator
from algorithm.embedding_operator import OpenAIEmbeddingOperator
from algorithm.models import Document
from algorithm.embedding_factory import ESEmbeddingFactory
from algorithm.document_factory import ESDocumentFactory
from algorithm.caching_strategy import PDFChunkingCachingStrategy
from algorithm.answer_strategy import SentenceTransformerAnswerStrategy
from utils.path import get_absolute_path

es_client_params = {
    "hosts": "http://localhost:9200",
}

embedding_index_name = 'example_embedding_index_2'
document_index_name = 'example_document_index_2'

if __name__ == '__main__':
    doc = Document(
        id='nlp_book',
        data=get_absolute_path("../samples/nlp_book.pdf")
    )

    caching_strategy = PDFChunkingCachingStrategy(
        document_factory=ESDocumentFactory(es_client_params, index_name=document_index_name),
        embedding_factory=ESEmbeddingFactory(es_client_params, embedding_size=512,
                                             index_name=embedding_index_name),
        embedding_operator=OpenAIEmbeddingOperator("text-embedding-ada-002"),
        document_operator=PDFDocumentOperator(),
        chunk_size=5,
        sentence_word_count=(10, 20)
    )

    ir_system = IRSystem(
        caching_strategy=caching_strategy,
        answer_strategy=SentenceTransformerAnswerStrategy("../artifacts/gpt2")
    )

    ir_system.index_document(doc)
