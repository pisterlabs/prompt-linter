# Class to generate queries to the Looker API using LangChain from natural language questions
import json
from webbrowser import Chrome
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema import BaseLanguageModel


class QueryConverter:
    def prompt_template(self):
        prompt_template = """
        Given an input question, first create a syntactically correct JSON. The JSON is Looker SDK's run_inline_query function's models.WriteQuery argument. Do not use "fields": ["*"] in the JSON. Field names must include the view name. For example, fields: ["pet.id"]. The JSON must include the view name. For example, "view": "pet".

        # LookML Reference

        ```
        {context}
        ```

        # Question
        {question}"""

        return PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

    def __init__(self, looker_model_name: str, docsearch: Chrome, llm: BaseLanguageModel):
        self.model_name = looker_model_name
        chain_type_kwargs = {"prompt": self.prompt_template()}
        self.qa = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff", retriever=docsearch.as_retriever(), chain_type_kwargs=chain_type_kwargs)

    def run(self, question):
        response = self.qa.run(question)
        # extract the contents within {} from the response string
        print(f"raw response: {response}")
        response = response[response.find("{"): response.rfind("}") + 1]
        print(f"split response: {response}")

        response_json = json.loads(response)
        # Delete only model field from json
        if "model" in response_json:
            del response_json["model"]
        response_json["model"] = self.model_name

        print(f"response_json: {response_json}")
        self.response_json = response_json

        return self.response_json
