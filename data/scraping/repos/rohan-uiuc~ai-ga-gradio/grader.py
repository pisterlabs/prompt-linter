import asyncio
import csv
import glob
import json
import os
import shutil
from datetime import datetime
from typing import Optional

from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, UnstructuredHTMLLoader
from langchain.output_parsers import PydanticOutputParser
from pathvalidate import sanitize_filename
from pydantic import BaseModel, Field
from tqdm import tqdm


class Grader:
    def __init__(self, model):
        print("Setting up environment for grading")
        os.environ["LANGCHAIN_TRACING"] = "true"
        self.title = None
        self.model = model
        self.rubric_file = 'docs/rubric_data.json'
        self.discussions_file_path = "docs/discussion_entries.json"
        self.fieldnames = ['student_name', 'total_score', 'score_breakdown', 'grader_comments', 'student_feedback',
                           'summary']
        self.docs = self.get_html_files()
        self.llm = ChatOpenAI(temperature=0, model_name=model)
        self.parser: PydanticOutputParser = self.create_parser()
        self.rubric_text = self.create_rubric_text()
        self.prompt = self.create_prompt()
        self.splitter = None
        self.tokens = self.get_num_tokens()
        self.llm_chain = self.create_llm_chain(model)
        self.csv = self.get_csv_file_name()
        self.outputs = []
        self.completed = 0
        self.lock = asyncio.Lock()

    class ToolArgsSchema(BaseModel):
        student_name: Optional[str] = Field(description="The name of the student")
        total_score: int = Field(description="The grade of the student's answer")
        score_breakdown: Optional[str] = Field(description="The grade split breakup based on rubric")
        grader_comments: Optional[str] = Field(
            description="The grade split breakup based on rubric added as grader's one liner customized comments to explain how the grade was calculated for that particular student's answer")
        student_feedback: Optional[str] = Field(
            description="The developmental feedback from Grader's point of view to the student, some examples are: 'Great work, ...', 'Although, your submission is relevant to the question, it doesn't answer the question entirely...'. Give customized feedback based on student's answer")
        summary: Optional[str] = Field(
            description="The overall summary of the student's answer outlining key points from the student's answer based on the rubric which can be used as a portion of a vectorstore, used to answer summary based questions about all the discussions")

        class Config:
            schema_extra = {
                "required": ["student_name", "total_score", "score_breakdown", "grader_comments", "student_feedback",
                             "summary"]
            }

    def create_parser(self):
        # print("in parser")
        return PydanticOutputParser(pydantic_object=self.ToolArgsSchema)

    def create_rubric_text(self):
        with open(self.rubric_file, 'r') as file:
            rubric = json.load(file)
        rubric_text = []
        self.title = None  # Initialize title
        for r in rubric:
            if 'description' in r and 'ratings' in r:
                rubric_text.append(f"RUBRIC CATEGORY: {r['description']}\n" + "\n".join(
                    [f"POINTS: {rating['points']} CRITERIA: {rating['description']}" for rating in r['ratings']]))
            elif 'points_possible' in r:
                rubric_text.append(f"MAX POINTS POSSIBLE: {r['points_possible']}")
                print("added points_possible")
            elif 'title' in r:  # Check if title exists in rubric
                self.title = r['title']  # Save title for later use
                rubric_text.append(f"TITLE: {self.title}")
            elif 'instruction' in r:
                rubric_text.append(f"DISCUSSION INSTRUCTIONS: {r['instruction']}")

        rubric_text = "\n".join(rubric_text)
        # print(rubric_text) Add this to log when moving to application
        return rubric_text


    def create_map_prompt(self):
        map_template_string = f"""I am an expert concise Canvas Discussion Summarizer! I am here to concisely summarize the following sections of a long canvas discussion responses of this student on the basis of instructions and rubric provided.
        The aim is to capture the important and key points on the basis of instructions and rubric provided and create a short summary, so that grading can be done on all the summarized sections of canvas discussion of a student's response.
        --------------------
        Following is the canvas instruction and rubric:
        {self.rubric_text}
        --------------------
        I will summarize this extracted part of a long canvas discussion: 
        {{input_documents}}
        """
        return PromptTemplate(template=map_template_string, input_variables=["input_documents"])

    def create_reduce_prompt(self):
        reduce_template_string = f"""I am a Canvas Discussion Grader! I am here to grade the following summarized sections of canvas discussion responses of the student on the basis of instructions and rubric provided.
        --------------------
        To grade student discussion, I will use the discussion instructions and rubric below. I will not deviate from the grading scheme.
        {self.rubric_text}
        --------------------
        I will be able to identify each student by name, their key interests, key features pertinent to the discussion intruction and rubric.
        I will be able to summarize the entire discussion in concise manner including key points from each student's answer.
        --------------------
        I will grade the following summarized canvas discussion: {{input_documents}}
        --------------------
        My grading results will ALWAYS be in following format:
        Format instructions: {{format_instructions}}
        """
        return PromptTemplate(
            template=reduce_template_string,
            input_variables=["input_documents"],
            output_parser=self.parser,
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )

    def create_map_llm_chain(self):
        print("Ready to grade!")
        map_llm_chain = LLMChain(
            llm=self.llm,
            prompt=self.map_prompt,
            verbose=True,
        )
        return map_llm_chain

    def create_reduce_llm_chain(self):
        reduce_llm_chain = LLMChain(
            llm=self.llm,
            prompt=self.reduce_prompt,
            verbose=True,
        )
        return reduce_llm_chain

    async def process_file(self, file, pbar):
        if self.model == 'gpt-4':
            await asyncio.sleep(10)  # Add a 3-second delay before each request
        result = await self.llm_chain.arun(file)
        output: self.ToolArgsSchema = self.parser.parse(result)
        async with self.lock:
            self.completed += 1
            pbar.update(1)
        return result

    async def run_chain(self):
        print("Grading Started! Now sit back and get a coffee \u2615")
        total = len(self.docs)
        pbar = tqdm(total=total)
        # if model is gpt-4, batch size is 2, else batch size is 5
        batch_size = 2 if self.model == 'gpt-4' else 5
        batches = [self.docs[i:i + batch_size] for i in range(0, len(self.docs), batch_size)]
        for batch in batches:
            tasks = [self.process_file(file, pbar) for file in batch]
            results = await asyncio.gather(*tasks)
            for result in results:
                output: self.ToolArgsSchema = self.parser.parse(result)
                self.outputs.append(output)
            if self.model == 'gpt-4':
                await asyncio.sleep(3)  # Add a delay between each batch
        pbar.close()
        self.save_csv()
        return True

    def create_csv(self):
        # remove existing csvs in output folder
        if os.path.exists('output'):
            shutil.rmtree('output')

        os.mkdir('output')
        now = datetime.now()  # current date and time
        date_time = now.strftime("%m-%d-%Y_%H-%M-%S")
        if self.title:  # If title exists, use it in the filename
            file_name = f"{self.title}-{self.llm.model_name}-{date_time}.csv"
        else:  # If title doesn't exist, use 'output' in the filename
            file_name = f"output-{self.llm.model_name}-{date_time}.csv"

        # Sanitize the entire filename
        sanitized_file_name = sanitize_filename(file_name)
        sanitized_file_name = os.path.join('output', sanitized_file_name)

        with open(sanitized_file_name, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writeheader()
        return sanitized_file_name

    def save_csv(self):
        # Use the filename created in create_csv method
        self.csv = self.create_csv()
        with open(self.csv, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            rows = [output.dict() for output in self.outputs]  # Convert each output to a dictionary
            writer.writerows(rows)  # Write all rows to the CSV
            print(f"Saved grades for {len(self.outputs)} students in {self.csv}")
            return True
        return False

    def get_html_files(self):
        loader = DirectoryLoader('docs', glob="**/*.html", loader_cls=UnstructuredHTMLLoader, recursive=True)
        document_list = loader.load()
        for document in document_list:
            document.metadata["name"] = document.metadata["source"].split("/")[-1].split(".")[0]
            break
        return document_list

    def create_prompt(self):
        # print("in prompt")
        prompt_template = f"""I am a Canvas Discussion Grader! I am here to grade the following canvas discussion on the basis of instructions and rubric provided.
        To grade student discussion, I will follow the rubric below. I will not deviate from the grading scheme.
        {self.rubric_text}

        I will be able to identify each student by name, identify their key interests, key features of the responses pertinent to the discussion intruction and rubric.
        I will be able to summarize the entire discussion in concise manner including key points from each student's answer.
        I will grade the following canvas discussion: {{input_documents}}

        My grading results will ALWAYS be in following format:
        Format instructions: {{format_instructions}}
        """
        return PromptTemplate(template=prompt_template, input_variables=["input_documents"], output_parser=self.parser,
                              partial_variables={"format_instructions": self.parser.get_format_instructions()})

    def create_llm_chain(self, model):
        print("Ready to grade!")

        return LLMChain(
            llm=self.llm,
            prompt=self.prompt,
        )

    def get_num_tokens(self):
        total_tokens = 0
        for doc in self.docs:
            summary_prompt = self.prompt.format(input_documents=doc)

            num_tokens = self.llm.get_num_tokens(summary_prompt)
            total_tokens += num_tokens

            # summary = self.llm(summary_prompt)

            # print (f"Summary: {summary.strip()}")
            # print ("\n")
        return total_tokens

    def get_csv_file_name(self):
        output_dir = 'output'
        if os.path.exists(output_dir):
            csv_files = glob.glob(os.path.join(output_dir, '*.csv'))
            if csv_files:
                return csv_files[0]  # return the first csv file found
        return None


def run(model):
    grader = Grader(model)
    asyncio.run(grader.run_chain())
    print("Grading successful")
