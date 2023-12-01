import os
import jsonlines
from tqdm import tqdm
import torch
from peft import PeftModel

from typing import Optional

from datasets import load_dataset
from langchain import PromptTemplate, FewShotPromptTemplate
from transformers import AutoModelForCausalLM, T5Tokenizer, AutoTokenizer, AutoConfig
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

# from src.few_shot.utils import save_to_disk
# from src.few_shot.together import infer


BASE_PATH = "/storage/ukp/work/sachdeva/research_projects/exp_calibration/"


def save_to_disk(data, file_name):
    with jsonlines.open(file_name, "a") as writer:
        for example in tqdm(data, total=len(data), desc="Saving samples ... "):
            writer.write(example)


class LLMClient:
    def __init__(
        self,
        template,
        model_name_or_path: str = None,
        tokenizer_name_or_path: Optional[str] = None,
        data_path: str = None,
        threshold: float = 0.5,
        task: str = "qg",
        max_new_tokens: int = 50,
        temperature: float = 0.01,
        top_p: float = 1,
        top_k: int = 40,
        repetition_penalty: float = 1.0,
        save_results: bool = True,
        max_samples: int = None,
        stop: str = "\n",
    ):
        self.base_model = model_name_or_path
        self.tokenizer = tokenizer_name_or_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.data_path = data_path
        self.task = task
        self.threshold = threshold
        self.save_results = save_results
        self.max_samples = max_samples
        self.stop = stop.split(";")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.lora_weights = f"{BASE_PATH}alpaca-cot-13b"

    def _load_model(self):
        tokenizer = LlamaTokenizer.from_pretrained(self.base_model)
        model = LlamaForCausalLM.from_pretrained(
            self.base_model,
            load_in_8bit=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model, self.lora_weights, torch_dtype=torch.float16,
        )
        model.to(self.device)
        # unwind broken decapoda-research config
        model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2
        return model, tokenizer

    def _create_zero_shot_prompt(self, context, question, answer):
        prompt = qg_template.format(context=context, question=question, answer=answer)
        return prompt

    def generate(self):
        c = 0
        skipped_instances = 0
        examples = []
        model, tokenizer = self._load_model()

        model_identifier = self.base_model.split("/")[-1]
        save_path = (
            BASE_PATH
            + f"src/data/squad/{model_identifier}_{self.task}_pipeline_temp_0.7"
        )
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # load squad data
        dataset = load_dataset("squad", "plain_text")
        train_data = dataset["train"]
        squad_data = [
            sample
            for sample in tqdm(
                train_data, total=len(train_data), desc="Loading SQuAD data ... "
            )
        ]

        current_files = []
        file_names = []
        file_path = BASE_PATH + f"src/data/squad/{self.data_path}"
        file_names = [file_path]

        model.eval()
        for file_name in file_names:
            with jsonlines.open(file_name) as reader:
                for example in tqdm(reader):
                    try:
                        id = example["id"].split("_")[0]
                        context = example["context"]
                        orig_example = [
                            sample for sample in squad_data if sample["id"] == id
                        ][0]
                        # print(orig_example)
                        orig_context = orig_example["context"]
                        orig_question = orig_example["question"]
                        orig_answer = orig_example["answers"]
                        c += 1

                        if self.max_samples:
                            if c == self.max_samples:
                                break

                        prompt = self._create_zero_shot_prompt(
                            context=orig_context,
                            question=orig_question,
                            answer=orig_answer["text"][0],
                        )
                        # print(prompt)

                        inputs = tokenizer(prompt, return_tensors="pt").to(self.device)

                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=self.max_new_tokens,
                            temperature=self.temperature,
                            top_p=self.top_p,
                            top_k=self.top_k,
                            repetition_penalty=self.repetition_penalty,
                            do_sample=True,
                            # num_return_sequences=1,
                            early_stopping=True,
                            pad_token_id=tokenizer.eos_token_id,
                        )
                        output = tokenizer.decode(outputs[0], skip_special_tokens=True)
                        # remove the context from the output
                        output = output[len(prompt) :]

                        result = {"id": example["id"], "context": context}

                        print("Context:", orig_context)
                        print("Question:", orig_question)
                        print("Answer:", orig_answer["text"][0])
                        print("Explanation:", output)
                        print("-" * 100)
                        #
                        # examples.append(result)
                        # if self.save_results:
                        #     if c % self.threshold == 0:
                        #         save_to_disk(
                        #             examples,
                        #             f"{save_path}/counterfactual_samples_{model_identifier}_{c}.jsonl"
                        #         )
                        #         examples = []
                    except Exception as e:
                        # print(outputs)
                        skipped_instances += 1
                        print(f"Skipped instance {c} due to error: {e}.")
                        continue

                    if c == 5:
                        break

            # save the remaining examples
            # if self.save_results:
            #     if examples:
            #         save_to_disk(
            #             examples,
            #             f"{save_path}/counterfactual_samples_{model_identifier}_{c}.jsonl"
            #         )
            break


if __name__ == "__main__":

    qg_template = """
As an answer explainer, your job is to give a rationale for the answer to the following question given the context it is derived from. 
The rationale should express a clear thought of reasoning that led to the answer.

Context: {context}

Question: {question}

Answer: {answer}

Rationale: Let's think step by step,

""".strip()

    model = "decapoda-research/llama-13b-hf"
    client = LLMClient(
        template=qg_template,
        model_name_or_path=model,
        tokenizer_name_or_path=model,
        task="zero-shot-cot",
        data_path="squad_counterfactuals_28_03.jsonl",
        threshold=1000,
        max_new_tokens=128,
        temperature=0.7,
        top_p=1,
        top_k=40,
        repetition_penalty=1.0,
        save_results=True,
        max_samples=None,
    )

    client.generate()
