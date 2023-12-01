from transformers import BartForConditionalGeneration, BartTokenizer
import torch
from langchain import OpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from config import Settings



class SummarizationUtils:
    # def __init__(self):
        # self.summerizer_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
        # self.summerizer_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

    # def perform_summarization(self, text):
    #     input_ids = self.summerizer_tokenizer.encode(text, truncation=True, max_length=1024, return_tensors="pt")
    #     summary_ids = self.summerizer_model.generate(input_ids, num_beams=4, max_length=150, early_stopping=True)
    #     summary_result = self.summerizer_tokenizer.decode(summary_ids.squeeze(), skip_special_tokens=True)
    #     return summary_result

    @staticmethod
    def perform_summarization(text):
        """
        Generates a summary for the provided text using the Large Language Model (LLM).

        Parameters:
            llm (LLM): An instance of the Large Language Model (LLM) for generating responses.
            text (str): The text that needs to be summarized.

        Returns:
            str: A summarized version of the input text.
        """
        
        # Create the template
        template_string = '''Summarize the following text into a detailed, clear, and formal concise version:

        Input Text:
        "{input_text}"

        Prompt Template:
        "Provide a comprehensive summary of the given text while ensuring clarity and formality. The summary should capture the most important points and maintain a formal tone. Pay attention to details and structure the summary logically."

        Summary:
        '''
        llm = OpenAI(openai_api_key=Settings().GPT_API_KEY, max_tokens=-1)
        # LLM call
        prompt_template = ChatPromptTemplate.from_template(template_string)
        chain = LLMChain(llm=llm, prompt=prompt_template)
        response = chain.run({"input_text" : text})

        return response
