import os
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain.chat_models import ChatAnthropic
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from typing import Optional, List
import json

load_dotenv(".env")


class ContributionsByDay(BaseModel):
    busiest_day: Optional[str]
    typical_days: Optional[str]
    no_contributions: Optional[str]


class RepositoriesCreated(BaseModel):
    total: Optional[int]
    last_3_months: Optional[int]
    popular_languages: Optional[List[str]]


class PullRequests(BaseModel):
    opened_last_3_months: Optional[int]


class Stars(BaseModel):
    total: Optional[int]


class Activity(BaseModel):
    contributions_last_year: Optional[int]
    contributions_by_day: Optional[ContributionsByDay]
    repositories_created: Optional[RepositoriesCreated]
    pull_requests: Optional[PullRequests]
    stars: Optional[Stars]


class BasicInfo(BaseModel):
    name: Optional[str]
    url: Optional[str]
    description: Optional[str]
    location: Optional[str]
    company: Optional[str]
    followers: Optional[int]
    following: Optional[int]


class GithubAgent(BaseModel):
    basic_info: Optional[BasicInfo]
    activity: Optional[Activity]


class GithubProfileAnalyzer:
    def __init__(self, anthropic_api_key, serper_api_key):
        self.anthropic_api_key = anthropic_api_key
        self.serper_api_key = serper_api_key
        self.model = ChatAnthropic(
            model="claude-2.0", max_tokens_to_sample=3000, temperature=0.7
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2500, chunk_overlap=300, length_function=len
        )
        self.template_string = """You are an expert Github Analyst and experiece in leading top projects and an awesome project manager.Your job is to analyse this github profile : {docs} and extract key contents in json {format_instructions} \n Make sure your responce is complete in proper json. \n If you cannot find the information needed for the output schema add None.Don't add ```json``` in the output. \n"""

    def clean_text(self, text):
        text = text.replace("\n", " ").replace("\xa0", " ")
        return " ".join(text.split())

    def analyze(self, github_url):

        loader = WebBaseLoader(github_url)
        github_profile = loader.load()[0].page_content
        github_profile = self.clean_text(github_profile)
        texts = self.text_splitter.create_documents([github_profile])
        pydantic_parser = PydanticOutputParser(pydantic_object=GithubAgent)
        format_instructions = pydantic_parser.get_format_instructions()
        prompt = ChatPromptTemplate.from_template(template=self.template_string)
        messages = prompt.format_messages(
            docs=github_profile, format_instructions=format_instructions
        )
        output = self.model(messages)
        output = output.content.replace("```json\n", '"').replace("\n```" , '"')
        print(output)
        return output


