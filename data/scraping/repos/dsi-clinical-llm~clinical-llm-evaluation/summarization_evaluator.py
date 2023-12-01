from typing import List

import evaluate
from langchain.text_splitter import RecursiveCharacterTextSplitter

from prompt_templates.prompt_abstract import Prompt, NestedPrompt
from evaluations.causal_llm_evaluators import CausalLanguageModelEvaluator
from prompt_templates.summarization.summarization_prompt import SummarizationNestedPrompt, SummarizationBasePrompt


class SummarizationEvaluator(CausalLanguageModelEvaluator):

    def __init__(
            self,
            chunk_size,
            chunk_overlap,
            num_of_words,
            *args,
            **kwargs
    ):
        super(SummarizationEvaluator, self).__init__(*args, **kwargs)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.num_of_words = num_of_words
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        self.get_logger().info(
            f'chunk_size: {chunk_size}\n'
            f'chunk_overlap: {chunk_overlap}\n'
            f'num_of_words: {num_of_words}\n'
        )

    def get_prompt_classes(self) -> List[NestedPrompt]:
        return [
            SummarizationNestedPrompt
        ]

    def generate_prompts(
            self,
            record,
            few_shot_records
    ) -> List[NestedPrompt]:
        identifier = record['id'] if 'id' in record else None
        article = record['article']
        abstract = record['abstract']
        prompts = []
        for nested_prompt_class in self.get_prompt_classes():
            base_prompt_class = nested_prompt_class.get_base_prompt_class()
            chunks = self.text_splitter.split_text(article)
            chunk_prompts = []
            for chunk in chunks:
                num_of_words = int((len(chunk) / len(article)) * self.num_of_words)
                prompt = base_prompt_class(
                    ground_truth=abstract,
                    record_id=identifier,
                    data={'article': chunk, 'num_of_words': num_of_words}
                )
                chunk_prompts.append(prompt)
            prompts.append(
                SummarizationNestedPrompt(
                    ground_truth=abstract,
                    record_id=identifier,
                    prompts=chunk_prompts
                )
            )
        return prompts

    def get_metrics(self) -> dict:
        # Return the regular metrics for the QA task
        rouge = evaluate.load('rouge')
        bleu = evaluate.load('bleu')
        meteor = evaluate.load('meteor')
        return {'rouge': rouge, 'bleu': bleu, 'meteor': meteor}, {}
