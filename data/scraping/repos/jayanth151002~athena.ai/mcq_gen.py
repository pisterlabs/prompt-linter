import os
import re 
import json 
import pickle
from tqdm import tqdm 
from typing import Optional
from dotenv import load_dotenv

from langchain.schema import HumanMessage
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate


class MCQgenChain:
    def __init__(self, project_name : str, artifact_folder : str, openai_key : Optional[str] = None, openai_temp : int = 0) -> None:
        load_dotenv(dotenv_path='.env/openai.env')    
        
        self.project_name = project_name
        self.artifact_folder = artifact_folder
        self.openai_key = os.getenv("OPENAI_API_KEY") if openai_key is None else openai_key

        self.story_summary_path = os.path.join(artifact_folder, project_name, "story_summary.pkl")
        self.story_summary = pickle.load(
            open(self.story_summary_path, "rb")
        )
        self.chunks = self.story_summary['summary_chunks']
        self.response_schemas = [
            ResponseSchema(name="question", description="A question generated from input text snippet."),
            ResponseSchema(name="options", description="Possible choices of the multiple choice question."),
            ResponseSchema(name="answer", description="Correct answer for question.")
        ]
        self.output_parser = StructuredOutputParser.from_response_schemas(self.response_schemas)
        self.format_instructions = self.output_parser.get_format_instructions() 
        self.chat_model = ChatOpenAI(temperature=openai_temp, model_name = "gpt-3.5-turbo", openai_api_key=self.openai_key)
        
        self.instruction  ="""
        Given a text input from the user, generate multiple choice questions 
        from it along with the correct answer. \n{format_instructions}\n{user_prompt}"""

        self.prompt = ChatPromptTemplate(
            messages=[
                HumanMessagePromptTemplate.from_template(self.instruction)
            ], 
            input_variables=["user_prompt"], 
            partial_variables={"format_instructions": self.format_instructions}
        )
    
    def post_process_string(self, markdown_text):
        json_string = re.search(r'```json\n(.*?)```', markdown_text, re.DOTALL).group(1)
        python_list = json.loads(f'[{json_string}]')
        
        for lst in python_list:
            lst['options'] = lst['options'].split('\n')
        return python_list
    
    def get_chunkwise_output(self):
        chunk_wise_output = []

        for idx, chunk in tqdm(enumerate(self.chunks[1:2]), total=len(self.chunks[1:2])):
            user_query = self.prompt.format_prompt(user_prompt = chunk)
            user_query_output = self.chat_model(user_query.to_messages())
            chunk_wise_output.append(user_query_output.content)
        
        return chunk_wise_output
    
    def generate_chunkwise_mcq(self):
        mcqs_list_chunkwise = []
        for chunk in self.get_chunkwise_output():
            try:
                mcqs_list_chunkwise.append(self.post_process_string(chunk))
            except:
                continue
        return mcqs_list_chunkwise
    
    def generate_mcqs(self):
        all_mcqs = {}
        for chunk_number, chunk_wise_mcq in enumerate(self.generate_chunkwise_mcq()):
            sub_chunks = []
            for mcqs in chunk_wise_mcq:
                sub_chunks.append({
                    "question": mcqs['question'],
                    "options": mcqs['options'][0].split('.')[:4],
                    "answer": mcqs['answer']
                })
            all_mcqs[f"chunk_{chunk_number}"] = sub_chunks
        return all_mcqs