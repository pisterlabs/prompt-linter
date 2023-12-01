import os
from dotenv import load_dotenv

from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain

load_dotenv()

# This script generates answers to the questions asked using generative AI models hosted on Hugging Face Hub.

models = [  
            "google/flan-t5-xxl",
            "salesforce/codegen25-7b-multi",
            "Open-Orca/OpenOrca-Preview1-13B",
            "stabilityai/FreeWilly2",
            "ai-forever/ruGPT-3.5-13B",
            "upstage/llama-30b-instruct-2048"
        ]

class QuestionAnswerGenerator:
    def __init__(self, model="google/flan-t5-xxl"):
        self.repo_id = model
        self.llm = HuggingFaceHub(
            repo_id=self.repo_id,
            model_kwargs={
                "temperature": 0.5,
                "max_length": 264
            }
        )
        self.prompt = self.generate_prompt()

    def generate_prompt(self):
        """
        Generates a prompt template for the question and answer format.
        """
        template = """Question: {question}

        Answer: """

        prompt = PromptTemplate(template=template, input_variables=["question"])

        return prompt

    def generate_answer(self, question):
        """
        Generates an answer for the given question using the LLMChain and the prompt.
        """
        llm_chain = LLMChain(prompt=self.prompt, llm=self.llm)

        answer = llm_chain.run(question)

        return answer

    def main(self):
        """
        Main function to run the QuestionAnswerGenerator. Continuously takes user input for questions and generates answers.
        """
        while True:
            question = input("\nQuestion: ")

            answer = self.generate_answer(question=question)

            print(f"Answer: {answer}")


if __name__ == "__main__":
    qag = QuestionAnswerGenerator(model=models[0])
    qag.main()