import json
import os
import pathlib
import random
import warnings

import torch
import openai
import datasets
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from eval.args import RunnerArguments, HFArguments, OAIArguments, GenerationArguments
from eval.evaluator import HFEvaluator, OAIEvaluator, _WARNING
from eval.tasks import ALL_TASKS, get_task

from dotenv import load_dotenv

load_dotenv()  # loads the api keys in .env

transformers.logging.set_verbosity_error()
datasets.logging.set_verbosity_error()

runner_args = RunnerArguments()
hf_args = HFArguments()
oai_args = OAIArguments()
gen_args = GenerationArguments()
args = HfArgumentParser([runner_args, hf_args, oai_args, gen_args]).parse_args()

args.output_dir = pathlib.Path(os.getcwd()).parent / args.output_dir
args.save_generations_raw_path = args.output_dir / args.save_generations_raw_path
args.save_generations_prc_path = args.output_dir / args.save_generations_prc_path
args.save_references_path = args.output_dir / args.save_references_path
args.save_results_path = args.output_dir / args.save_results_path
args.save_generations_raw_path.parent.mkdir(parents=True, exist_ok=True)
args.save_generations_prc_path.parent.mkdir(parents=True, exist_ok=True)
args.save_references_path.parent.mkdir(parents=True, exist_ok=True)
args.save_results_path.parent.mkdir(parents=True, exist_ok=True)


task_name = "proofwriter-neurosymbolic-2shot"
my_parent_task = get_task(task_name)


# this is a very bad engineering practice. don't do that at home!
class CustomSubTask(type(my_parent_task)):
    def __init__(self, *args, **kwargs):
        super(CustomSubTask, self).__init__(*args, **kwargs)
        self.stop_words.append("</OUTPUT>")

    def get_prompt(self, doc):
        """
        Builds the prompt for the LM to generate from.
        :param doc: dict[str: str]
            sample from the test dataset
        :return: str
        """
        instructions = self.get_instructions()
        train = self.fewshot_examples()
        test = self.format_test_example(doc)
        prompt = "\n".join([instructions, train, test])
        return prompt

    def get_extra_context_prompt(self, generation):
        instruction_and_train = self.get_instruction_and_extra_context_examples()
        test = self.format_extra_context_example(generation)
        return "\n".join([instruction_and_train, test])

    def format_extra_context_example(self, generation):
        # todo maybe we should also include the doc itself (similar to the format_example?)
        return f"""
<INPUT>
<PREMISES>
{generation}
</PREMISES>
</INPUT>
<OUTPUT>
"""

    def get_instruction_and_extra_context_examples(self):
        return f"""
You will be given the premises for a first-order logic (FOL) problem.
The problem is to identify additional premises that are implicitly common sense from the ones given.
The premises are given in the form of a set of first-order logic sentences.
The task is to generate new common sense premises, text and FOL pairs, that would be common sense to someone reading the original premises.
Expressions should be adhere to the format of the Python NLTK package logic module. Here are a couple examples:
        
<INPUT>
<PREMISES>
Premise: When a person reads a book, that person gains knowledge.
FOL: all x. all y. (Person(x) & Reads(x, y) & Book(y) -> Gains(x, Knowledge))
Premise: Harry read the book "Walden" by Henry Thoreau.
FOL: Reads(Harry, Walden)
</PREMISES>
</INPUT>
<OUTPUT>
Premise: Harry is a human.
FOL: Person(Harry)
Premise: Walden is a book.
FOL: Book(Walden)
</OUTPUT>

<INPUT>
<PREMISES>
Premise: Heinrich Schmidt was a Nazi German politician.
FOL: NaziGermanPolitician (HeinrichSchmidt)
</PREMISES>
</INPUT>
<OUTPUT>
Premise: Heinrich Schmidt was a Nazi
FOL: Nazi(HeinrichSchmidt)
Premise: Heinrich Schmidt was a German
FOL: German(HeinrichSchmidt)
Premise: Heinrich Schmidt was a Politician
FOL: Politician(HeinrichSchmidt)
</OUTPUT>

<INPUT>
<PREMISES>
Premise: Famine is bad
FOL: Bad(Famine)
</PREMISES>
</INPUT>
<OUTPUT>
Premise: Bad is not good.
FOL: Bad -> -Good
Premise: Cold is not warm
FOL: Cold -> -Warm
</OUTPUT>

Do not limit on how many additional premises you generate, so long as they are common sense.
"""


class CustomOAIEvaluator(OAIEvaluator):
    def generate_text(self, task):
        dataset = task.get_dataset()
        n_tasks = self.args.limit if self.args.limit else len(dataset)
        prompts = [task.get_prompt(dataset[i]) for i in range(n_tasks)]
        stops = [task.stop_words for _ in range(n_tasks)]

        # todo add this fancy stuff back later
        # with ThreadPoolExecutor() as executor:
        #     res = executor.map(self.get_completion, prompts, stops)

        # todo instead of choosing over the sample use all of them
        initial_generations_raw = [self.get_completion(prompt, stop) for prompt, stop in zip(prompts, stops)]

        # todo change n_sample of here to 1
        generation_prompts = [task.get_extra_context_prompt(random.choice(generation)) for generation in initial_generations_raw]
        generations_raw = [self.get_completion(prompt, stop) for prompt, stop in zip(generation_prompts, stops)]
        if self.args.postprocess:
            generations_prc = [
                [
                    task.postprocess_generation(
                        generations_raw[i][j], i, completion_only=True
                    )
                    for j in range(self.args.n_samples)
                ]
                for i in range(n_tasks)
            ]
        else:
            generations_prc = generations_raw
        references = [task.get_reference(dataset[i]) for i in range(n_tasks)]
        return generations_prc, generations_raw, references


    # uses task instead of task_name
    def evaluate(self, task):
        if task.requires_execution and not self.allow_code_execution:
            raise ValueError(_WARNING)

        generations_prc, generations_raw, references = self.generate_text(task)
        if len(generations_prc[0]) != self.args.n_samples:
            generations_prc = [l[: self.args.n_samples] for l in generations_prc]
            warnings.warn(
                "Number of tasks wasn't proportional to number of devices, we removed extra predictions"
            )

        if not hasattr(self, "accelerator") or self.accelerator.is_main_process:
            if not self.args.generations_path:
                if self.args.save_generations_raw:
                    with open(self.args.save_generations_raw_path, "w") as fp:
                        json.dump(generations_raw, fp)
                        print("raw generations were saved")
                if self.args.save_generations_prc:
                    with open(self.args.save_generations_prc_path, "w") as fp:
                        json.dump(generations_prc, fp)
                        print("processed generations were saved")
                if self.args.save_references:
                    with open(self.args.save_references_path, "w") as fp:
                        json.dump(references, fp)
                        print("references were saved")

            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            if self.allow_code_execution and task.requires_execution:
                os.environ["HF_ALLOW_CODE_EVAL"] = "1"
            results = task.process_results(generations_prc, references)
            return results

    # uncomment this if you don't have access to LLM or you want to manually debug...
    # def get_completion(self, prompt, stop):
    #     print("please query this and tell me the answer:")
    #     print("stop words: ", stop)
    #     print(prompt)
    #     print("paste here and type end at the end:")
    #     res = ""
    #     while True:
    #         s = input()
    #         if s.strip() == "end":
    #             break
    #         elif s.strip():
    #             res += s.strip() + "\n"
    #     return [res] * self.args.n_samples  # copy instead of asking n times...


if __name__ == "__main__":
    args.max_length_generation = 4096
    args.openai_api_env_keys = ['OPENAI_API_KEY']
    args.model = 'gpt-3.5-turbo'  # hard coded model here
    args.allow_code_execution = True
    args.n_samples = 1

    is_chat = True  # todo change this if you change model
    args.limit = 1  # hard coded so that we don't run all examples

    # todo uncomment this for checking the comparison
    # task = my_parent_task
    # evaluator = OAIEvaluator(args, chat=is_chat)
    # result = evaluator.evaluate(task_name)

    task = CustomSubTask()
    evaluator = CustomOAIEvaluator(args, chat=is_chat)
    result = evaluator.evaluate(task)

    print(result)
