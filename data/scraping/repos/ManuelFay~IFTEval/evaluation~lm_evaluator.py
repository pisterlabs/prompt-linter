import json
import re

import numpy as np
from illuin_llm_tools import OpenAIConnector, Prompt

from evaluation import GENERAL_PROMPT_TEMPLATE as PROMPT_TEMPLATE
from evaluation.base_evaluator import BaseEvaluator


class LMEvaluator(BaseEvaluator):
    def __init__(self, **kwargs):
        """Initialize the evaluator with a list of metrics to compute

        clf_metrics: List[str] = ["accuracy", "f1", "precision", "recall"]
        qa_metrics: List[str] = ["squad"]
        summarization_metrics: List[str] = ["rouge1", "rouge2", "rougeL", "bleu"]
        translation_metrics: List[str] = ["bleu"]
        """
        super().__init__(**kwargs)
        self.connector = OpenAIConnector(
            model=kwargs.get("model", "gpt-4"),
            max_tokens=1024,
            temperature=0.1,
            cache_path=kwargs.get("cache_path", ".cache"),
        )
        self.max_req = int(kwargs.get("max_req_per_s", 2))
        self.model_name = kwargs.get("model", "gpt-4")
        self.name = f"{self.model_name}_evaluator"

    def _compute(self, *args, **kwargs):
        predictions = args[0]
        references = args[1]
        prompts = args[2]

        formatted_prompts = []
        for i, (pred, ref, prompt) in enumerate(zip(predictions, references, prompts)):
            formatted_prompts.append(
                Prompt(id=i, text=PROMPT_TEMPLATE.format(prompt=prompt, response1=pred, response2=ref))
            )

        responses = self.connector.multi_requests(
            formatted_prompts,
            max_requests_per_second=self.max_req,
            progress_desc=f"Prompting {self.model_name} model",
            max_catching_retries=10,
            clean_cache_at_end=False,
        )
        # print([response.text for response in responses])

        def parse_response(response):
            try:
                response = re.sub(r"[^,\d]", "", response.text)
            except:
                print(response.text)
                return [np.nan, np.nan]
            response = response.split(",")
            if (
                len(response) != 2
                or float(response[0]) < 0
                or float(response[1]) < 0
                or float(response[0]) > 10
                or float(response[1]) > 10
            ):
                return [np.nan, np.nan]
            return [float(x) for x in response]

        responses = [parse_response(response) for response in sorted(responses, key=lambda x: x.id)]
        responses = np.array(responses)
        return {
            f"{self.model_name}_lm_score": np.nanmean(responses, axis=0)[0],
            f"{self.model_name}_mean_score_ref": np.nanmean(responses, axis=0)[1],
            f"{self.model_name}_mean_score_ratio": np.nanmean(responses, axis=0)[0] / np.nanmean(responses, axis=0)[1],
            f"{self.model_name}_responses": [
                {
                    "prediction": pred,
                    "ref": ref,
                    "score_pred": int(sp) if not np.isnan(sp) else "NA",
                    "score_ref": int(sr) if not np.isnan(sr) else "NA",
                }
                for pred, ref, sp, sr in zip(predictions, references, list(responses[:, 0]), list(responses[:, 1]))
            ],
        }


def run_cli(predictions_file: str, predictions_key: str, reference_key: str, **kwargs):
    evaluator = LMEvaluator(**kwargs)
    scores = evaluator.compute_from_jsonl(predictions_file, predictions_key, reference_key)
    with open(predictions_file.replace(".jsonl", "_scores.json"), "w") as f:
        f.write(json.dumps(scores, indent=4))
    print(
        f"Mean score prediction: {scores['mean_score_pred']}, Mean score reference: {scores['mean_score_ref']}, Mean score ratio: {scores['mean_score_ratio']}"
    )


if __name__ == "__main__":
    import fire

    fire.Fire(run_cli)
