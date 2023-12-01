from utils import set_env_variables
from piqard.utils.data_loaders import DatabaseLoaderFactory
from piqard.utils.io import save_results

from benchmark_evaluator import BenchmarkEvaluator
from piqard.PIQARD import PIQARD
from piqard.language_models.bloom_176b_api import BLOOM176bAPI
from piqard.language_models.cohere_api import CohereAPI
from piqard.utils.prompt_template import PromptTemplate


if __name__ == "__main__":
    set_env_variables()

    database_loader = DatabaseLoaderFactory("openbookqa")
    benchmark = database_loader.load_questions(
        "../assets/benchmarks/openbookqa/test.jsonl"
    )

    language_models = [CohereAPI(stop_token="|||"), BLOOM176bAPI(stop_token="|||")]
    prompting_templates_dir = (
        "../assets/prompting_templates/openbookqa/chain_of_thought/"
    )

    for language_model in language_models:
        for n in [1, 3, 5]:
            piqard = PIQARD(
                PromptTemplate(
                    f"{prompting_templates_dir}/cot_{n}_shot.txt",
                    fix_text="So the final answer is:",
                ),
                language_model,
            )
            benchmark_evaluator = BenchmarkEvaluator(piqard)
            results = benchmark_evaluator.evaluate(
                benchmark,
                f"./result/openbookqa/chain_of_thought/{language_model}/{n}_shot_checkpoint.jsonl",
            )
            save_results(
                f"./result/openbookqa/chain_of_thought/{language_model}/{n}_shot.json",
                results,
            )
