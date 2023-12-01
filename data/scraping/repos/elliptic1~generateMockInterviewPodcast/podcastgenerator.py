from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import OutputParserException

from data.company import get_company
from data.interview_types import get_interview_type
from data.job_titles import get_job_title
from data.senior_job_titles import get_senior_job_title
from data.voices import get_interview_voices
from keys import OpenAI_API_KEY
from music.add_metadata import add_metadata
from music.add_music import add_intro_outro_music
from speech_util import create_qa_audio_files, create_intro_audio_files, create_outro_audio_files
from util import remove_files, concatenate_audio_files, copy_to_google_drive, get_article

remove_files()

max_content_tokens = 2000  # Reserve some tokens for the other messages and instructions.

chat = ChatOpenAI(
    openai_api_key=OpenAI_API_KEY,
    model_name="gpt-4",
)

# This is to get a random tech job title
company = get_company()
interviewer_title = get_senior_job_title()
job_post_title = get_job_title()
interviewee_voice_name, interviewer_voice_name = get_interview_voices()
interviewee_old_company = get_company()
interviewee_old_job = get_job_title()
num_questions_per_response = 1
total_num_questions = 1
interview_type = get_interview_type()

title = f"Today, {interviewer_voice_name} will act as {interviewer_title}, and give a {interview_type}" \
        + f" interview to {interviewee_voice_name} for the job of {job_post_title} at {company}." \
        + f" {interviewee_voice_name} currently works at {interviewee_old_company} as " \
        + f"{get_article(interviewee_old_job)} {interviewee_old_job}."

print(title)

# Commented out IPython magic to ensure Python compatibility.
response_schemas = [
    ResponseSchema(name="Introduction", description="This is an introduction to the interview"),
    ResponseSchema(name="Outro", description="This is an outro for the interview"),
    ResponseSchema(name="Wrap", description="This is a short prompt by the interviewer to wrap up the interview"),
    ResponseSchema(name="Guest Intro", description="This is an introduction given by the guest"),
    ResponseSchema(name="Guest Outro", description="This is an outro given by the guest"),
]

# How you would like to parse your output
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

system_template_shared = f"""
This is a podcast called "Tech Star Podcast" where we give mock interviews to guests.
You are {interviewer_voice_name}, the podcast host, and you will also act as a {interviewer_title} from {company}. 
You are conducting a {interview_type} interview of {interviewee_voice_name} for the job of {job_post_title}.
{interviewee_voice_name} currently works at {interviewee_old_company} as a {interviewee_old_job}.
Research the company {company}, its business practices, ideology, and its technologies, and use that
context to create your questions.
Research the company {interviewee_old_company}, its business practices, ideology, and its technologies, and use that
context to create your answers.
"""

system_template = f"""
The intro should contain information about {company} and thank the guest, whose name is {interviewee_voice_name}.
The outro should thank the guest, {interviewee_voice_name}, and thank the audience, and ask people to subscribe to the podcast.
The guest intro should be a short introduction of the guest, {interviewee_voice_name}, as a mock interviewee.
The guest outro should be a short outro by the guest, {interviewee_voice_name}, that responds to the interviewer's outro.
The wrap should be a short prompt by {interviewer_voice_name} to wrap up the interview.
"""

system_message_prompt = SystemMessagePromptTemplate.from_template(system_template_shared + system_template)

human_template = """

Generate the intro, outro, and wrap for the interview. 

{format_instructions}

Your RFC8259 compliant JSON Response:
"""

human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

_input = chat_prompt.format_prompt(
    format_instructions=output_parser.get_format_instructions(),
    interviewer_title=interviewer_title,
    interviewee_voice=interviewee_voice_name,
    interviewer_voice=interviewer_voice_name,
    company=company,
    job_post_title=job_post_title,
)

got_intro = False
intro_outro_response_json = {}

while not got_intro:

    print("Asking for intro and outro...")
    output = chat(_input.to_messages())

    try:
        intro_outro_response_json = output_parser.parse(output.content)
        got_intro = True
    except OutputParserException:
        print(output.content)
        print("Failed to parse intro and outro. Trying again...")
        intro_outro_response_json = {}

response_schemas = [
    ResponseSchema(name="Question1", description="This is question 1"),
    ResponseSchema(name="Answer1", description="This is answer 1"),
    ResponseSchema(name="Response1", description="This is response 1"),
]

# How you would like to parse your output
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

system_template = f"""
Question Instructions:
Ask {interview_type} interview questions related to the job of {job_post_title} at {company}, 
from the point of view of a {interviewer_title} from {company}.
For each question, preface it by an introduction of the question so {interviewee_voice_name} has the full context.
The question should be written in a normal speaking style and sound like something a real person would say.

Answer Instructions:
The answer should be based on {interviewee_voice_name}'s experience at {interviewee_old_company} as a {interviewee_old_job}.
Generate a thorough and detailed answer from {interviewee_voice_name} for each question using the STAR method.
The STAR method is a technique you can use to answer interview questions. 
STAR stands for situation, task, action and result.
Do not mention the STAR method or "situation", "task", "action", "result" in your answer.
The answer should be written in a normal speaking style and sound like something a real person would say.

Answer Response Instructions:
For each answer, generate a thoughtful response from {interviewer_voice_name}.
The response should not contain a question.
The response should be one sentence long and should be a statement.
The response should not be overly positive or flattering.
The response should be written in a normal speaking style and sound something a real interviewer would say.

General Instructions:
There should be no markdown, no formatting, no newlines, no quotation marks, and no tabs in your output.
Generate {num_questions_per_response} questions and answers and responses.
Do not include any explanations, only provide a RFC8259 compliant JSON response.
Do not prefix the output with 'json' or anything else.
"""

system_message_prompt = SystemMessagePromptTemplate.from_template(system_template_shared + system_template)

human_template = """

{format_instructions}

Your RFC8259 compliant JSON response:
"""

human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

_input = chat_prompt.format_prompt(
    format_instructions=output_parser.get_format_instructions(),
    interviewer_title=interviewer_title,
    company=company,
    job_post_title=job_post_title,
    num_questions=num_questions_per_response,
)

filenames = []
batch = 0

filenames += create_intro_audio_files(intro_outro_response_json, interviewee_voice_name, interviewer_voice_name)

while (len(filenames) - 5) / 3 < total_num_questions:
    batch = batch + 1

    got_json = False

    response_json = {}
    while not got_json:
        print(f"sending input for batch {batch} to chat")
        output = chat(_input.to_messages())
        print("got output from chat")

        try:
            response_json = output_parser.parse(output.content)
            got_json = True
        except OutputParserException:
            print(output.content)
            print("Failed to parse json. Trying again...")
            response_json = {}

    # Create the audio files
    print(f"creating audio files for batch {batch}")
    filenames += create_qa_audio_files(response_json, interviewee_voice_name, num_questions_per_response,
                                       interviewer_voice_name, batch)

filenames += create_outro_audio_files(intro_outro_response_json, interviewee_voice_name, interviewer_voice_name)

# Concatenate the audio files
print("concatenating audio files")
concatenate_audio_files(filenames)

# Add intro and outro music
print("adding intro and outro music")
add_intro_outro_music()

# Add metadata
print("adding metadata")
add_metadata(company, job_post_title, interviewee_voice_name, interviewee_old_job, interviewee_old_company)

print("uploading to google drive")
copy_to_google_drive()
