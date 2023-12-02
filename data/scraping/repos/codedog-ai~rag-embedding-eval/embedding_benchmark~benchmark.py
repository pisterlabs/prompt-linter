from langchain.schema import Document
from langchain.schema.embeddings import Embeddings
from langchain.schema.language_model import BaseLanguageModel

from embedding_benchmark.embedding import Case, DocumentEmbeddingBuilder


class Benchmark:
    def __init__(
        self, cases: list[Case], llm: BaseLanguageModel, embedding: Embeddings
    ):
        self.cases = cases
        self.builder = DocumentEmbeddingBuilder(llm=llm, embedding=embedding)
        self.doc_ebds = {}
        self.query_ebds = {}
        self.result = []

    def run_embedddings(self):
        for case in self.cases:
            key = case.topic
            doc = case.doc()
            pq = case.positive_query_docs()
            nq = case.negative_query_docs()
            self.builder.add_doc(key, doc)
            self.builder.add_queries(key, pq + nq)

        doc_embeddings = self.builder.run_doc_embedding()
        query_embeddings = self.builder.run_query_embedding()

        self.doc_ebds = doc_embeddings
        self.query_ebds = query_embeddings

    def calculate(
        self,
        doc_embeddings: dict[str, dict[str, list[tuple[Document, list[float]]]]],
        query_embeddings: dict[
            str, dict[str, dict[str, list[tuple[Document, list[float]]]]]
        ],
    ):
        self._data = []

        for key, doc_ebd in doc_embeddings.items():
            query_ebd = query_embeddings[key]

            for _d_type, _d_type_ebds in doc_ebd.items():
                for _q_type, _q_type_ebds in query_ebd.items():
                    for _q_d_type, _q_d_type_ebds in _q_type_ebds.items():
                        min_dis, max_dis = self._calc_min_max_distance(
                            _d_type_ebds, _q_d_type_ebds
                        )
                        gt, query_type = _q_type.split("|")
                        record = {
                            "topic": key,
                            "doc_embedding_type": _d_type,
                            "query_embedding_type": _q_d_type,
                            "query_type": query_type,
                            "label": gt,
                            "min_score": min_dis,
                            "max_score": max_dis,
                        }
                        self.result.append(record)

    def _calc_min_max_distance(
        self, _d_type_ebds, _q_d_type_ebds
    ) -> tuple[float, float]:
        min_dis = 100.0
        max_dis = 0.0

        for _d in _d_type_ebds:
            for _q in _q_d_type_ebds:
                distance = self._cosine(_d[1], _q[1])
                if distance <= min_dis:
                    min_dis = distance
                if distance >= max_dis:
                    max_dis = distance

        return min_dis, max_dis

    def _cosine(self, a: list[float], b: list[float]) -> float:
        _sum = 0.0
        for i in range(len(a)):
            _sum += a[i] * b[i]

        _sum_a = 0.0
        _sum_b = 0.0
        for i in range(len(a)):
            _sum_a += a[i] * a[i]
            _sum_b += b[i] * b[i]

        return _sum / (_sum_a * _sum_b) ** 0.5
