from collections import defaultdict

from benchmark_evaluator import BenchmarkEvaluator
from utils import set_env_variables
from piqard.PIQARD import PIQARD
from piqard.information_retrievers import FAISSRetriever
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
        FAISSRetriever(
            "openbookqa",
            database_path="../assets/benchmarks/openbookqa/corpus.jsonl",
            database_index="../assets/benchmarks/openbookqa/corpus_faiss_index.pickle",
        )
    ]
    prompting_tempates_dir = "../assets/prompting_templates/openbookqa/permutations/"

    for language_model in language_models:
        for information_retriever in information_retrievers:
            results_agg = {}
            for n in range(1, 7):
                piqard = PIQARD(
                    PromptTemplate(f"{prompting_tempates_dir}permutation_{n}.txt"),
                    language_model,
                    information_retriever,
                )
                benchmark_evaluator = BenchmarkEvaluator(piqard)
                results = benchmark_evaluator.evaluate(
                    benchmark,
                    f"./result/openbookqa/permutations/{language_model}/{information_retriever}/permutations_{n}_checkpoint.jsonl",
                )
                save_results(
                    f"./result/openbookqa/permutations/{language_model}/{information_retriever}/permutations_{n}.json",
                    results,
                )

                results_agg[f"{n}_shot"] = results["report"]

            consistency = defaultdict(int)
            for i in range(len(benchmark)):
                consistency[
                    len(
                        set(
                            [
                                question[i]["predicted_answer"]
                                for question in results_agg.values()
                            ]
                        )
                    )
                ] += 1
            save_results(
                f"./result/openbookqa/permutations/{language_model}/{information_retriever}/consistency_report.json",
                {
                    "percent_of_consistent_answers": consistency[1] / len(benchmark),
                    "numbers_of_unique_answers": consistency,
                },
            )
