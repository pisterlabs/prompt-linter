import json
from dotenv import load_dotenv
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chat_models import ChatOpenAI

load_dotenv()

# Load job listings from the JSON file
with open('norway_construction_data.json', 'r', encoding='utf-8') as file:
    job_listings = json.load(file)

# Initialize the ChatOpenAI class
chat = ChatOpenAI()

instructions1 = """Create a very concise job summary in English about the job listing. It must be max 90 words. Focus on the most critical details and omit less important information. Use the format below:

*Company name*: *Job title*
Experience Required:
Education:
Skills:
*brief summary here*

Job listing:
"""

instructions2 = """###
Remember, max 90 words!
"""

# Create a PromptTemplates
human_message_template = HumanMessagePromptTemplate.from_template(
    "{instructions1}\n{employer_name}\n{title}\n{description}\n{instructions2}"
)
system_message = "You create very concise job summaries from Norwegian job listings."
system_message_template = SystemMessagePromptTemplate.from_template(system_message)
chat_prompt = ChatPromptTemplate.from_messages([
    system_message_template,
    human_message_template
])

def create_messages_for_job(job):
    title = job['title']
    employer_name = job['employer']['name']
    description = job['description']
    
    # Use the PromptTemplate to format the human_message
    chat_prompt_value = chat_prompt.format_prompt(
        instructions1=instructions1,
        instructions2=instructions2,
        employer_name=employer_name,
        title=title, 
        description=description
    )
    
    return chat_prompt_value

# Iterate over the job listings and update them with summaries
for index, job in enumerate(job_listings["content"]):
    chat_prompt_value_result = create_messages_for_job(job)
    messages = chat_prompt_value_result.to_messages()

    generation = chat(messages)
    summary = generation.content

    # Add the generated summary to the job listing
    job['summary'] = summary

    # Print the loop number for progress indication
    print(f"Processed job listing {index + 1}/{len(job_listings['content'])}")

    # Write the updated job listings to the JSON file after each iteration
    with open('norway_construction_data.json', 'w', encoding='utf-8') as outfile:
        json.dump(job_listings, outfile, ensure_ascii=False, indent=4)

print("All job listings processed and summaries added.")

