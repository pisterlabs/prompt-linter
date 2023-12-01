from typing import Optional, List, Any
import logging

from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.schema import Generation, LLMResult
from langchain.llms.utils import enforce_stop_tokens
from datasets import Dataset
from transformers.pipelines.pt_utils import KeyDataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


def data(dataset):
    for row in dataset:
        yield row["prompt"]


class BatchedHuggingFacePipeline(HuggingFacePipeline):
    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        prompts = Dataset.from_dict({"prompt": prompts})

        text_generations: List[str] = []

        for out in tqdm(
            self.pipeline(
                KeyDataset(prompts, "prompt"),
                batch_size=self.batch_size,
                return_full_text=False,
                **self.pipeline_kwargs,
            ),
            total=len(prompts),
        ):
            text_generations.append(out[0]["generated_text"])
            # try:
            #     from transformers.pipelines.text_generation import ReturnType

            #     remove_prompt = (
            #         self.pipeline._postprocess_params.get("return_type")
            #         != ReturnType.NEW_TEXT
            #     )
            # except Exception as e:
            #     logger.warning(
            #         f"Unable to extract pipeline return_type. "
            #         f"Received error:\n\n{e}"
            #     )
            #     remove_prompt = True
            # if remove_prompt:
            #     text = out["generated_text"]
            # else:
            #     text = response["generated_text"]

        # for i in range(0, len(prompts), self.batch_size):
        #     batch_prompts = prompts[i : i + self.batch_size]
        #     # Process batch of prompts
        #     responses = self.pipeline(batch_prompts)

        #     # Process each response in the batch
        #     for j, response in enumerate(responses):
        #         if isinstance(response, list):
        #             # if model returns multiple generations, pick the top one
        #             response = response[0]

        #         if self.pipeline.task == "text-generation":
        #             try:
        #                 from transformers.pipelines.text_generation import ReturnType

        #                 remove_prompt = (
        #                     self.pipeline._postprocess_params.get("return_type")
        #                     != ReturnType.NEW_TEXT
        #                 )
        #             except Exception as e:
        #                 logger.warning(
        #                     f"Unable to extract pipeline return_type. "
        #                     f"Received error:\n\n{e}"
        #                 )
        #                 remove_prompt = True
        #             if remove_prompt:
        #                 text = response["generated_text"][len(batch_prompts[j]) :]
        #             else:
        #                 text = response["generated_text"]
        #         elif self.pipeline.task == "text2text-generation":
        #             text = response["generated_text"]
        #         elif self.pipeline.task == "summarization":
        #             text = response["summary_text"]
        #         else:
        #             raise ValueError(
        #                 f"Got invalid task {self.pipeline.task}, "
        #                 f"currently only {VALID_TASKS} are supported"
        #             )
        #         if stop:
        #             # Enforce stop tokens
        #             text = enforce_stop_tokens(text, stop)

        #         # Append the processed text to results
        #         text_generations.append(text)

        return LLMResult(generations=[[Generation(text=text)] for text in text_generations])
