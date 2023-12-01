import random
from typing import Any, Mapping

try:
    import guidance
except ImportError:
    print("Could not import guidance. This is OK unless using LLamaAugmenter.")
import polars as pl
from tqdm import tqdm

from common_lit_kaggle.framework import table_io
from common_lit_kaggle.framework.task import Task
from common_lit_kaggle.settings.config import Config
from common_lit_kaggle.tables import AugmentedLlamaTrainTable
from common_lit_kaggle.tasks.data_split.data_blocks import data_blocks_generator

# pylint: disable=invalid-name
imported_auto_gptq = False
try:
    from auto_gptq import AutoGPTQForCausalLM
    from transformers import AutoTokenizer

    imported_auto_gptq = True

except ImportError as excp:
    print("Import error!", excp)


class LlamaAugmenterTask(Task):
    def __init__(self, name: str | None = None) -> None:
        super().__init__(name)

    def run(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        assert imported_auto_gptq, "Cannot run llama augmenter task without autogptq!"

        train_data: pl.DataFrame = context["train_unified_text_data"]

        config = Config.get()

        tokenizer, model = self._load_llama(config.llama_path)
        llama = guidance.llms.transformers.Vicuna(model=model, tokenizer=tokenizer)
        number_of_few_shot_examples = 2

        new_data_points = []
        # Consume generator so we know how many blocks we have for tqdm

        old_augmented = None
        augmented = None
        try:
            old_augmented = table_io.read_table(AugmentedLlamaTrainTable())
        # pylint: disable=broad-exception-caught
        except Exception:
            pass
        program = guidance(
            """{{#system~}}{{prompt}}{{~/system}}
{{#assistant~}}TOPIC TITLE: {{gen 'topic_title' stop='\n' max_tokens=10}}
REFERENCE TEXT: {{gen 'reference_text' max_tokens=400}}
QUESTION: {{gen 'question' stop='\n' max_tokens=40}}{{~/assistant}}
{{#user~}}STUDENT ANSWER: {{gen 'student_answer' max_tokens=400 stop='Q'}}{{~/user}}
"""
        )
        blocks = list(data_blocks_generator(train_data))
        number_of_samples_per_block = 100
        for data_block in tqdm(blocks):
            for _ in tqdm(range(number_of_samples_per_block)):
                sample = data_block.sample(
                    min(number_of_few_shot_examples, len(data_block))
                )
                if len(sample) < 1:
                    continue

                sample_content_mean = sample["content"].mean()
                sample_wording_mean = sample["wording"].mean()
                prompt = self._samples_to_guidance_prompt(sample)

                output = program(prompt=prompt, llm=llama)
                topic_title = output["topic_title"]
                reference_text = output["reference_text"]
                question = output["question"]
                student_answer = output["student_answer"]

                # pylint: disable=line-too-long
                output = f"TOPIC TITLE: {topic_title}\nREFERENCE TEXT: {reference_text}\nQUESTION: {question}\nSTUDENT ANSWER: {student_answer}"
                print("Generated datapoint:", topic_title, "\n", student_answer)
                new_data_points.append(
                    {
                        "student_id": f"SYNTHETIC_DATA_{random.randint(0, 999999999)}",
                        "content": sample_content_mean,
                        "wording": sample_wording_mean,
                        "text": student_answer,
                        "prompt_id": f"SYNTHETIC_DATA_{random.randint(0, 999999999)}",
                        "prompt_title": topic_title,
                        "prompt_question": question,
                        "prompt_text": reference_text,
                    }
                )

                new_data_points_df = pl.DataFrame(new_data_points)  # type: ignore

                if old_augmented is not None:
                    augmented = pl.concat([old_augmented, new_data_points_df])  # type: ignore
                else:
                    augmented = new_data_points_df

                table_io.write_table(augmented, AugmentedLlamaTrainTable())  # type: ignore

        return {"llama_augmented": augmented}

    def _samples_to_guidance_prompt(self, sample: pl.DataFrame) -> str:
        # Available columns to use in prompt
        # student_id = pl.Utf8
        # prompt_id = pl.Utf8
        # content = pl.Float64
        # wording = pl.Float64
        # unified_text = pl.Utf8
        # unified_labels = pl.Utf8

        unified_text_list = sample.select("unified_text").to_numpy().tolist()
        # Flatten list
        unified_text_list = [text[0] for text in unified_text_list]
        prompt = "\n".join(unified_text_list)

        return (
            "You are an advanced Artificial Intelligence simulating student summarizing essays. "
            + "Here's an example of what you do.\n"
            + prompt
            + "\nYour turn. Create some new questions answers following the template. "
            + "\nDo your best to follow the previous student style and writing skills."
        )

    def _load_llama(self, model_directory):
        config = Config.get()
        model = AutoGPTQForCausalLM.from_quantized(
            model_directory, device=config.device, use_safetensors=True, use_triton=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_directory, use_fast=True)
        return tokenizer, model
