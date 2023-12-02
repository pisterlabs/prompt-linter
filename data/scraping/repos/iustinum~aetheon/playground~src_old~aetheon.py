import openai
import os
from dotenv import load_dotenv

from model import GPT
import repo_reader


class Aetheon():

    def __init__(self, model_name) -> None:
        load_dotenv(".env")
        openai.api_key = os.environ["OPENAI_API_KEY"]
        self.model = GPT(model=model_name)

    def run(self, repo_name, branch, author):
        repo_reader.generateDataFile(author, repo_name, branch=branch)

        allNames = repo_reader.generateFileNames()
        desc = repo_reader.generateDescriptions(allNames)

        self.model.giveContext(desc)
        text = self.model.createMDText()


        print(text)


    def load_repo(self, repo_name, branch, author):
        pass

    def generateInlineDocs(self):
        pass
