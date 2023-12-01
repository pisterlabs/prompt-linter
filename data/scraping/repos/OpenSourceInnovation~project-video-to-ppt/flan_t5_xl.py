import os
import torch
import accelerate
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline, AutoModelForCausalLM

model_id = 'google/flan-t5-large'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id, load_in_8bit=True)

pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=512,
)


class templates:
    """
    This class provides methods to generate a summary and a title for a given text using a pre-trained T5 model.
    """

    def __init__(self) -> None:
        """
        Initializes the templates class with summaryPipe and TitlePipe set to None.
        """
        self.summaryPipe = None
        self.TitlePipe = None

    def ChunkSummarizer(self, text, custom_instruction: str = None, **kwargs):
        """
        Generates a summary for the given text using a pre-trained T5 model.

        Args:
            text (str): The input text for which summary needs to be generated.
            custom_instruction (str, optional): Custom instruction to generate the summary. Defaults to None.
            **kwargs: Additional keyword arguments to be passed to the pipeline.

        Returns:
            str: The generated summary for the input text.
        """
        default_instruction = "generate a perfect title for the following text in 6 words: "
        instructions = custom_instruction if custom_instruction is not None else default_instruction
        pipe = self.summaryPipe

        max_length = kwargs.get("max_length", 400)

        if pipe is None:
            pipe = pipeline(
                "text2text-generation",
                model=model,
                tokenizer=tokenizer,
                max_length=max_length,
            )

        return pipe(instructions + text)[0]["generated_text"]

    def ChunkTitle(self, text, custom_instruction: str = None, **kwargs):
        """
        Generates a title for the given text using a pre-trained T5 model.

        Args:
            text (str): The input text for which title needs to be generated.
            custom_instruction (str, optional): Custom instruction to generate the title. Defaults to None.
            **kwargs: Additional keyword arguments to be passed to the pipeline.

        Returns:
            str: The generated title for the input text.
        """
        default_instruction = "generate a perfect title for the following text in 6 words: "
        instructions = custom_instruction if custom_instruction is not None else default_instruction
        pipe = self.summaryPipe

        max_length = kwargs.get("max_length", 60)

        if pipe is None:
            pipe = pipeline(
                "text2text-generation",
                model=model,
                tokenizer=tokenizer,
                max_length=max_length,
            )

        return pipe(instructions + text)[0]["generated_text"]

    @staticmethod
    def model():
        """
        Returns the pre-trained T5 model.
        """
        return pipe
