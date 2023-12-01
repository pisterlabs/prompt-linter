
import sys
from typing import List
import os
from .prompts import PromptBook
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

import json

from .utils import Utilities


class Developer():
    def __init__(self, requirement: str, root_dir: str, openai_api_key: str, model: str):
        self.requirement = requirement
        self.root_dir = root_dir
        self.api_key = openai_api_key
        self.files_written = []
        self.brain = ChatOpenAI(
            model=model,
            openai_api_key=self.api_key,
            temperature=0.1,
            streaming=False
        )

    def brain_storm(self, prompt: str, step_name="prompt") -> str:
        messages = [
            SystemMessage(content="you are a senior software developer"),
            HumanMessage(content=prompt)
        ]
        aiMessage: AIMessage = self.brain(messages=messages)
        if not os.path.exists("./thoughts"):
            os.makedirs("./thoughts")
        if os.path.exists(f"./thoughts/{step_name}.md"):
            os.remove(f"./thoughts/{step_name}.md")
        Utilities.write_to_file(
            f"{prompt}\n\n{aiMessage.content}", f"./thoughts/{step_name}.md")
        return aiMessage.content

    def get_doubts(self):
        prompt = PromptBook.expand_requirements(self.requirement)
        doubts = self.brain_storm(prompt, 'clear-doubts')
        doubt_list: List[str] = doubts.split("\n")
        doubt_list = [doubt.strip()
                      for doubt in doubt_list if doubt.strip() != ""]
        return doubt_list

    def get_clarifications(self, doubts: List[str], answers: List[str]):
        clarifications = ""
        for i in range(len(doubts)):
            clarifications = f"{clarifications}\n\n{i+1}. {doubts[i]}\n Ans: {answers[i]}"

        compressed_clarrifications = self.brain_storm(
            PromptBook.get_compressed_text(clarifications), "compressed_clarrifications")
        self.clarifications = compressed_clarrifications

        return compressed_clarrifications

    def clear_doubts(self):
        doubt_list = self.get_doubts()
        print("""
Hey there! 😄 It's Lazy Dev, your friendly neighborhood programmer, here to make your awesome project's dreams come true! 🎉 
But before I dive into coding magic, I have a few fun and important questions for you. 
So, grab a cup of coffee ☕️, sit back, and let's clarify some details, shall we? Here we go! 🚀
        """)
        answer_list = []
        for doubt in doubt_list:
            answer = input(f"{doubt}\n>>")
            answer_list.append(answer)

        print("""
Thats all I need! 😄

Sit back and relax while I work my coding magic for you! ✨✨✨

🚀 I've got this! 🎉

Cheers! 👨‍💻
        """)
        return doubt_list, answer_list

    def plan_project(self):
        prompt = PromptBook.plan_project(self.requirement, self.clarifications)
        plannings: str = self.brain_storm(prompt, 'plan-project')
        compressed_plan = self.brain_storm(
            PromptBook.get_compressed_text(plannings), "compressed_plan")
        self.plannings = compressed_plan
        return compressed_plan

    def generate_folder_structure(self):
        prompt = PromptBook.design_folder_structure(
            question=self.requirement,
            plan=self.plannings,
            clarifications=self.clarifications
        )
        retry_count = 3
        while retry_count > 0:
            try:

                folder_tree_str: str = self.brain_storm(
                    prompt, "generate-filders")
                folder_tree: dict = json.loads(
                    folder_tree_str.strip().strip("`"))
                break
            except:
                print("Opps messed up the json format, let me try again")
                retry_count = retry_count-1

        if retry_count == 0:
            print("Sorry I was not able to create the folder structure in json correct format, check my instructions and try to refine it sothat i can understan the task better")
            sys.exit()
        self.root_folder_name, self.file_paths = Utilities.generate_files_and_folders(
            structure=folder_tree, root_dir=self.root_dir)
        return self.root_folder_name, self.file_paths

    def prioratize_files(self):
        prompt = PromptBook.prioritise_file_list(
            self.file_paths, self.plannings)
        retry_count = 3
        while retry_count > 0:
            try:
                file_paths_str = self.brain_storm(prompt, 'prioratize_files')
                break
            except:
                print("Opps messed up the json format, let me try again")
                retry_count = retry_count-1
        if retry_count == 0:
            print("Sorry I was not able to create the file list in correct format, check my instructions and try to refine it sothat i can understan the task better")
            sys.exit()
        self.file_paths = file_paths_str.split("\n")
        return self.file_paths

    def write_file_content(self, file_path, review_iteration: int = 1):
        prompt = PromptBook.write_file(
            question=self.requirement,
            clarifications=self.clarifications,
            plan=self.plannings,
            files_written=self.files_written,
            file_path_to_write=file_path,
            file_paths=self.file_paths
        )
        filename = file_path.split("/")[-1]
        code = self.brain_storm(prompt, f'code-{filename}')
        if (review_iteration > 1):
            print(f"Reviewing the code {filename}")
            for i in range(review_iteration-1):
                review_prompt = PromptBook.get_code_feedback(
                    draft=code,
                    question=self.requirement,
                    clarifications=self.clarifications,
                    plan=self.plannings,
                    files_written=self.files_written,
                    file_path_to_write=file_path,
                    file_paths=self.file_paths
                )
                response = self.brain_storm(
                    review_prompt, f'code-{file_path.split("/")[-1]}')
                response = response.strip('."\'`-\n')
                nonecheck_response = response.split("\n")[0].split(" ")[0]
                if (nonecheck_response.strip('."\'`-\n') == "NONE"):
                    break
                code = response

        Utilities.write_to_file(code, file_path=file_path)
        # compress the code
        compressed_code = self.brain_storm(PromptBook.get_compressed_code(
            code), f'code-compressed-{file_path.split("/")[-1]}')
        self.files_written.append((
            file_path,
            compressed_code
        ))

    def final_instruction(self):
        prompt = PromptBook.generate_instruction(
            question=self.requirement,
            clarifications=self.clarifications,
            plan=self.plannings,
            files_written=self.files_written,
            file_paths=self.file_paths
        )
        instruction = self.brain_storm(prompt, f'instructions')
        return instruction

    def develop(self):
        # clearing all doubts
        doubts, answers = self.clear_doubts()
        self.clarifications = self.get_clarifications(
            doubts=doubts, answers=answers)
        # planning the project
        print("Brainstorming ideas...")
        self.plan_project()
        print(self.plannings)
        print("\n\n")
        # creating files and folders for the project
        print("Creating folder structure...")
        root_folder_name, file_paths = self.generate_folder_structure()

        self.prioratize_files()
        self.files_written = []
        for file_path in self.file_paths:
            file_name = file_path.split("/")[-1]
            if (file_name.split(".")[-1] in ["png", "jpg", "jpeg", "bimp", "lock"]):
                continue
            print(f"\nWriting Code for :{file_name}")
            self.write_file_content(file_path, review_iteration=1)
        print("\n\nI am done!!!\n\n")
        print(self.final_instruction())
