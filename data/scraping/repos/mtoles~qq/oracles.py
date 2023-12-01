from transformers import AutoTokenizer, T5ForConditionalGeneration, AutoModelForCausalLM
import torch
import math
import random
import openai
import time
import string

from metrics import get_metrics
from prepare_data import prepare_inputs_hp
from tqdm import tqdm

# from nltk.corpus import stopwords

# sw_set = set(stopwords.words("english"))


class Oracle:
    """abstract method for oracles"""

    def __init__(self):
        print("subclass this method")

    def process(self, ds, q2_masking_scheme):
        new_ds = ds.map(
            lambda x: self.forward(x, q2_masking_scheme),
            load_from_cache_file=False,
            batched=True,
            batch_size=1,  # batching happens internally
        )
        return new_ds

    def forward():
        print("subclass this method")


class T5_Bool_Oracle(Oracle):
    def __init__(
        self,
        model_size="base",
        batch_size=1,
        # raw_val_dataset=None,
    ):
        self.batch_size = batch_size
        self.model_size = model_size
        self.model_name = f"google/flan-t5-{self.model_size}"
        self.tk = AutoTokenizer.from_pretrained(
            self.model_name, cache_dir="./.model_cache"
        )
        self.model = T5ForConditionalGeneration.from_pretrained(
            self.model_name, cache_dir="./.model_cache"
        ).cuda()
        self.model.eval()

    def forward(self, example, q2_masking_scheme):
        """Perform forward pass on a single example. Not sure what happens with padding if you pass multiple examples."""
        with torch.no_grad():
            q2 = example[f"q2_{q2_masking_scheme}"][0]
            masked_sentence = example["masked_sentence"][0]
            masked_sentence_title = example["masked_sentence_title"][0]
            # Build the corpus
            # First answer is correct. The rest are distractor.
            # corpus_strs = masked_sentence + [
            #     distractor
            #     for sublist in example["context_distractor"][0]["sentences"]
            #     for distractor in sublist
            # ]
            cs_template = "%s: %s"
            corpus_strs = [
                cs_template % (masked_sentence_title, masked_sentence)
            ]  # make sure a2 is always at index 0
            # add distractors
            for i, sublist in enumerate(example["context_distractor"][0]["sentences"]):
                for distractor in sublist:
                    title = example["context_None"][0]["title"][i]
                    corpus_str = cs_template % (title, distractor)
                    # corpus_str = distractor
                    corpus_strs.append(corpus_str)
            # add supporting facts
            for i, sublist in enumerate(example["context_supporting"][0]["sentences"]):
                for supporting in sublist:
                    title = example["context_None"][0]["title"][i]
                    corpus_str = cs_template % (title, supporting)
                    # corpus_str = supporting
                    if corpus_str != masked_sentence:
                        corpus_strs.append(corpus_str)
            # drop the masked sentence from corpus_strs
            input_strs = [
                f"question: {q2}\ncontext: {cs}\nprompt: Does the context answer the question, yes or no?"
                for cs in corpus_strs
            ]

            input_ids = self.tk(
                input_strs, return_tensors="pt", padding=True
            ).input_ids.cuda()

            c = len(corpus_strs)

            # copy input_ids for each possible answer
            label_strs = ["yes", "no"]
            label_encoding = self.tk(label_strs, return_tensors="pt", padding=True)
            max_answer_len = 1  # must change if label_strs is edited
            label_ids = label_encoding.input_ids[:, :-1].cuda()
            label_attention_masks = label_encoding.attention_mask[
                :, :-1
            ].cuda()  # different from bloom

            # process logits in batches
            example = self._fw0(
                c, example, input_ids, label_ids, corpus_strs, q2_masking_scheme
            )
            # example1 = self._fw1(c, example, input_ids, label_ids, corpus_strs, q2_masking_scheme)
            return example

    def _fw0(self, c, example, input_ids, label_ids, corpus_strs, q2_masking_scheme):
        # normal
        num_batches = math.ceil(c / self.batch_size)
        probs = []
        for i in range(num_batches):
            start = i * self.batch_size
            end = min((i + 1) * self.batch_size, c)
            # batch_logits = self.model(
            #     input_ids=input_ids[start:end], labels=label_ids
            # ).logits
            batch_logits = self.model.generate(
                input_ids[start:end],
                max_new_tokens=1,
                output_scores=True,
                return_dict_in_generate=True,
            ).scores[0]
            yn_scores = batch_logits[:, label_ids.T.squeeze()].softmax(dim=1)
            probs.append(yn_scores)
        probs = torch.cat(probs, dim=0)
        assert len(probs) == len(corpus_strs)

        best_index = probs[:, 0].argmax()
        best_prob = probs[:, 0].max()
        # always answer mode
        oracle_answer = corpus_strs[best_index]
        oracle_answer_is_correct = bool(best_index == 0)

        example[f"a2_{q2_masking_scheme}"] = [oracle_answer]
        example[f"a2_is_correct_{q2_masking_scheme}"] = [oracle_answer_is_correct]
        return example


class OpenAI_Oracle(Oracle):
    def __init__(self, model):
        # model_size and batch_size are unused but are here for compatibility with T5_Oracle
        assert model in ["gpt-3.5-turbo", "gpt-4"]
        self.model = model

    def forward(self, example, q2_masking_scheme):
        q2 = example[f"q2_{q2_masking_scheme}"][0]
        masked_sentence = example["masked_sentence"][0]
        masked_sentence_title = example["masked_sentence_title"][0]
        # Build the corpus
        # First answer is correct. The rest are distractor.
        cs_template = "%s: %s"
        corpus_strs = [
            cs_template % (masked_sentence_title, masked_sentence)
        ]  # make sure a2 is always at index 0
        # add distractors
        for i, sublist in enumerate(example["context_distractor"][0]["sentences"]):
            for distractor in sublist:
                title = example["context_None"][0]["title"][i]
                # corpus_str = cs_template % (title, distractor)
                corpus_str = distractor
                corpus_strs.append(corpus_str)
        # add supporting facts
        for i, sublist in enumerate(example["context_supporting"][0]["sentences"]):
            for supporting in sublist:
                title = example["context_None"][0]["title"][i]
                # corpus_str = cs_template % (title, supporting)
                corpus_str = supporting
                corpus_strs.append(corpus_str)
        input_strs = [
            f"question: {q2}\ncontext: {cs}\nprompt: Does the context answer the question, yes or no?"
            for cs in corpus_strs
        ]
        ### end copy paste from the other oracle ###

        # shuffle the strings to remove possible bias from ChatGPT
        # but remember the place where the correct answer is
        correct_answer = corpus_strs[0]
        random.shuffle(corpus_strs)
        correct_index = corpus_strs.index(correct_answer)

        answers_block = ""
        for i, cs in enumerate(corpus_strs):
            answers_block += f"{i}. {cs}\n"

        prompt = f"Question: {q2}\n\n{answers_block}\n\nWhich answer is correct? Only say the number of the answer, nothing else."

        def call_oai_api(prompt):
            while True:
                try:
                    response = openai.ChatCompletion.create(
                        model=self.model,
                        messages=[
                            {"role": "user", "content": prompt},
                        ],
                    )
                    break
                except Exception as e:
                    print(e)
                    print("Retrying...")
                    # pause a second
                    time.sleep(1)
                    continue

            a1 = response["choices"][0]["message"]["content"].strip()
            self.oai_model_id = response.model
            idx = f"{self.oai_model_id} {prompt}"
            return a1

        generation = call_oai_api(prompt)
        # strip punctuation from generation
        generation = "".join(char for char in generation if char in string.digits)
        generation_idx = None
        try:
            generation_idx = int(generation)
            assert generation_idx < len(corpus_strs)
        except (ValueError, AssertionError):
            print("OpenAI returned a non-integer answer. Returning random index.")
            generation_idx = random.randint(0, len(corpus_strs) - 1)
        oracle_answer = corpus_strs[generation_idx]
        oracle_answer_is_correct = bool(generation_idx == correct_index)
        example[f"a2_{q2_masking_scheme}"] = [oracle_answer]
        example[f"a2_is_correct_{q2_masking_scheme}"] = [oracle_answer_is_correct]
        return example
