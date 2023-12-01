from benchmark_evaluator import BenchmarkEvaluator
from utils import set_env_variables
from piqard.PIQARD import PIQARD
from piqard.information_retrievers import FAISSRetriever
from piqard.language_models.bloom_176b_api import BLOOM176bAPI
from piqard.language_models.cohere_api import CohereAPI
from piqard.utils.io import save_results
from piqard.utils.prompt_template import PromptTemplate
from piqard.utils.data_loaders import DatabaseLoaderFactory


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
    prompting_tempates_dir = "../assets/prompting_templates/openbookqa/few_shot/"

    for language_model in language_models:
        for information_retriever in information_retrievers:
            for n in range(0, 6):
                piqard = PIQARD(
                    PromptTemplate(f"{prompting_tempates_dir}{n}_shot_1_documents.txt"),
                    language_model,
                    information_retriever,
                )
                benchmark_evaluator = BenchmarkEvaluator(piqard)
                results = benchmark_evaluator.evaluate(
                    benchmark,
                    f"./result/openbookqa/n_shot/{language_model}/{information_retriever}/{n}_shot_checkpoint.jsonl",
                )
                save_results(
                    f"./result/openbookqa/n_shot/{language_model}/{information_retriever}/{n}_shot.json",
                    results,
                )
