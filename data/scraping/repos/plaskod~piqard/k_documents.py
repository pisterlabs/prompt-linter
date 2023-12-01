from benchmark_evaluator import BenchmarkEvaluator
from utils import set_env_variables
from piqard.PIQARD import PIQARD
from piqard.information_retrievers import BM25Retriever, FAISSRetriever
from piqard.information_retrievers.annoy_retriever import AnnoyRetriever
from piqard.language_models.bloom_176b_api import BLOOM176bAPI
from piqard.language_models.cohere_api import CohereAPI
from piqard.utils.data_loaders import DatabaseLoaderFactory
from piqard.utils.io import save_results
from piqard.utils.prompt_template import PromptTemplate


if __name__ == "__main__":
    set_env_variables()

    database_loader = DatabaseLoaderFactory("openbookqa")
    benchmark = database_loader.load_questions(
        "../assets/benchmarks/openbookqa/test.jsonl"
    )

    language_models = [CohereAPI(stop_token="\n"), BLOOM176bAPI(stop_token="\n")]
    information_retrievers = [
        AnnoyRetriever(
            "openbookqa",
            database_path="../assets/benchmarks/openbookqa/corpus.jsonl",
            database_index="../assets/benchmarks/openbookqa/corpus_annoy_index_384.ann",
        ),
        BM25Retriever(
            "openbookqa",
            database_path="../assets/benchmarks/openbookqa/corpus.jsonl",
            database_index="../assets/benchmarks/openbookqa/corpus_bm25_index.pickle",
        ),
        FAISSRetriever(
            "openbookqa",
            database_path="../assets/benchmarks/openbookqa/corpus.jsonl",
            database_index="../assets/benchmarks/openbookqa/corpus_faiss_index.pickle",
        ),
    ]

    prompting_templates_dir = "../assets/prompting_templates/openbookqa/k_documents/"

    for language_model in language_models:
        for information_retriever in information_retrievers:
            for k in range(1, 4):
                information_retriever.k = k
                piqard = PIQARD(
                    PromptTemplate(
                        f"{prompting_templates_dir}5_shot_{k}_documents.txt"
                    ),
                    language_model,
                    information_retriever,
                )
                benchmark_evaluator = BenchmarkEvaluator(piqard)
                results = benchmark_evaluator.evaluate(
                    benchmark,
                    f"./result/openbookqa/k_documents/{language_model}/{information_retriever}/{k}_documents_checkpoint.jsonl",
                )
                save_results(
                    f"./result/openbookqa/k_documents/{language_model}/{information_retriever}/{k}_documents.json",
                    results,
                )

        piqard = PIQARD(
            PromptTemplate(f"{prompting_templates_dir}5_shot_{0}_documents.txt"),
            language_model,
        )
        benchmark_evaluator = BenchmarkEvaluator(piqard)
        results = benchmark_evaluator.evaluate(
            benchmark,
            f"./result/openbookqa/k_documents/{language_model}/0_documents_checkpoint.jsonl",
        )
        save_results(
            f"./result/openbookqa/k_documents/{language_model}/0_documents.json",
            results,
        )
