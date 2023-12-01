import logging
import os
import random
from pathlib import Path
from typing import Union

from dotenv import load_dotenv
from langchain import OpenAI, PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
from jonbot import logger


class PaperSummary(BaseModel):
    title: str = Field("", description="The title of the research article")
    author_year: str = Field("", description="The author and year of the research article (e.g. 'Smith et al. 2020')")
    citation: str = Field("", description="The citation to the research article")
    abstract: str = Field("", description="A copy-paste of the abstract of the research article")
    detailed_summary: str = Field("",
                                  description="A detailed summary/overview of the major points of the paper in a bulleted outline format ( with * to denote bullet points) with 1-2 sentence top level bullet points and second-level bullet points")
    short_summary: str = Field("", description="A short (2-3 sentence) summary of the paper")
    very_short_summary: str = Field("", description="A very short one sentence summary of the research article")
    extremely_short_summary: str = Field("", description="An extremely short 6-10 word summary of the research article")
    basic_methodology: str = Field("",
                                   description="A basic description of the methodology used in the research article (e.g. information about tools used, species studied, analytical techniques, etc.)")
    summary_title: str = Field("",
                               description="A summary title made by combining the `author_year` field with the `extremely_short_summary` field, like this: ['author_year'] - ['extremely_short_summary']")
    tags: str = Field("", description="A list of tags formatted using #kebab-case-lowercase")
    backlinks: str = Field("",
                           description="The same as the 'tags' field, but with double brackets around the tag name and without the # symbol, so '#tag-name' becomes '[[tag-name]]'")

    def __str__(self):
        tags = "\n".join(self.tags.split(" "))
        backlinks = "\n".join(self.backlinks.split(" "))
        return f"""
# {self.summary_title}\n
## Title\n
{self.title}\n\n
## Citation:\n
{self.citation}\n\n
## Abstract\n
{self.abstract}\n\n
## Basic Methodology\n
{self.basic_methodology}\n\n
## Detailed Summary\n
{self.detailed_summary}\n\n
## Short Summary\n
{self.short_summary}\n\n
## Very Short Summary\n
{self.very_short_summary}\n\n
## Extremely Short Summary\n
{self.extremely_short_summary}\n\n
## Tags\n
{tags}\n\n
## Backlinks\n
{backlinks}
"""


PAPER_SUMMARY_TEMPLATE_STRING = """
Use these instructions to convert the input text into a paper summary:\n 
{format_instructions} \n\n
Input text: \n
___\n
{input_text}\n\n
___\n
DO NOT MAKE ANYTHING UP. ONLY USE TEXT FROM THE INPUT TEXT. \n
IF YOU DO NOT HAVE ENOUGH INFORMATION TO FILL OUT A FIELD SAY 'COULD NOT FIND IN INPUT TEXT'
"""


class UnstructuredTextParser:
    def __init__(self,
                 prompt_template_string: str = PAPER_SUMMARY_TEMPLATE_STRING,
                 pydantic_object: BaseModel = PaperSummary, ):
        self._llm = OpenAI(model_name="text-davinci-003",
                           temperature=0,
                           max_tokens=-1,
                           streaming=True,
                           callbacks=[StreamingStdOutCallbackHandler()],
                           )

        self._parser = PydanticOutputParser(pydantic_object=pydantic_object)

        self._prompt_template = PromptTemplate(
            template=prompt_template_string,
            input_variables=["input_text"],
            partial_variables={"format_instructions": self._parser.get_format_instructions()}
        )

    def parse_input(self, input_text: str) -> PaperSummary:
        _input_text = self._prompt_template.format_prompt(input_text=input_text)
        _output = self._llm(_input_text.to_string())
        response = self._parser.parse(_output)
        return response

    async def aparse_input(self, input_text: str) -> PaperSummary:
        response = self.parse_input(input_text=input_text)
        return response


async def parse_green_check_messages(overwrite: bool = False,
                                     save_to_json: bool = True,
                                     collection_name: str = "green_check_messages"):
    parser = UnstructuredTextParser()

    collection = mongo_database.get_collection(collection_name)
    all_entries = await collection.find().to_list(length=None)
    random.shuffle(all_entries)
    logger.info("Parsing green check messages")

    for entry in all_entries:

        print(
            "=====================================================================================================")
        print(f"Green Check Messages for student: {entry['_student_initials']}\n")
        print(
            "=====================================================================================================")

        messages = entry["green_check_messages"]
        if len(messages) == 0:
            raise ValueError(f"Student {entry['_student_initials']} has no green check messages")

        messages = "\n".join(messages)
        parsed_output = await parser.aparse_input(input_text=messages)

        student_query = {"_student_initials": entry["_student_initials"],
                         "_student_uuid": entry["_student_uuid"], }
        await mongo_database.upsert(
            collection=collection_name,
            query=student_query,
            data={"$set": {**parsed_output.dict(),
                           "_parsed_output_string": str(parsed_output),
                           "messages": messages,
                           }}
        )
        save_green_check_entry_to_markdown(base_summary_name="green_check_paper_summaries",
                                           text=str(parsed_output),
                                           file_name=f"{parsed_output.author_year}_{parsed_output.title}_{entry['_student_initials']}",
                                           backlinks=parsed_output.backlinks,
                                           tags=parsed_output.tags,
                                           )

        print(f"Student: {entry['_student_initials']}: \n"
              f"Messages with green check: \n{messages}\n"
              f"Parsed output: \n{parsed_output}")

    if save_to_json:
        await mongo_database.save_json(collection_name=collection_name)


def save_green_check_entry_to_markdown(base_summary_name: str,
                                       text: str,
                                       file_name: str,
                                       subfolder: str = None,
                                       save_directory: Union[str, Path] = None,
                                       backlinks: str = None,
                                       tags: str = None,
                                       ):
    if not save_directory:
        load_dotenv()
        save_directory = Path(
            os.getenv(
                "PATH_TO_COURSE_DROPBOX_FOLDER")) / "course_data" / "chatbot_data" / base_summary_name
    if subfolder:
        save_directory = save_directory / subfolder

    summaries_directory = save_directory / "paper_summaries"
    summaries_directory.mkdir(parents=True, exist_ok=True)

    clean_file_name = file_name.replace(":", "_").replace(".", "_").replace(" ", "_")
    clean_file_name += ".md"
    summary_file_path = summaries_directory / clean_file_name

    with open(str(summary_file_path), 'w', encoding='utf-8') as f:
        f.write(text)

    print(f"Markdown file generated and saved at {str(summary_file_path)}.")

    if backlinks:
        backlinks_split = backlinks.split(" ")
        for backlink in backlinks_split:
            backlinks_clean = backlink.replace("[", "").replace("]", "")
            backlink_md = backlinks_clean + ".md"
            tag_path = summaries_directory.parent / "tags"
            tag_path.mkdir(parents=True, exist_ok=True)
            backlink_path = tag_path / backlink_md

            if tags:
                tags_split = tags.split(" ")
                backlink_path.write_text("\n".join(tags_split), encoding="utf-8")
            else:
                backlink_path.touch(exist_ok=True)

        print(f"Backlinks generated and saved at {str(tag_path)}.")


if __name__ == "__main__":
    import asyncio

    asyncio.run(grab_green_check_messages(server_name="Neural Control of Real World Human Movement 2023 Summer1",
                                          overwrite=True,
                                          save_to_json=True,
                                          ))

    asyncio.run(parse_green_check_messages(collection_name="green_check_messages",
                                           overwrite=True,
                                           save_to_json=True,
                                           ))
