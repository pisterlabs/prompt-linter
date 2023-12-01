import os

from dotenv import load_dotenv
import guidance

load_dotenv()

def extract_params(search):
    # set the default language model used to execute guidance programs
    guidance.llm = guidance.llms.OpenAI("text-davinci-003", token=os.getenv("OPENAI_API_KEY"))

    program = guidance("""Extract the following from this job search query, if there is no location write NONE:

    EXAMPLE
    {{example_input}}
    QUERY: {{example_query}}
    LOCATION: {{example_location}}

    UPDATED
    {{input}}
    QUERY: {{gen 'query' stop='\\n'}}
    LOCATION: {{gen 'location'}}""")

    # execute the program on a specific proverb
    executed_program = program(
        example_input="san francisco remote data jobs",
        example_query="remote data jobs",
        example_location="san francisco",
        input=search,
    )

    query = executed_program["query"].strip()
    location = None if executed_program["location"].strip() == "NONE" else executed_program["location"].strip()

    return {"extracted_query": query, "extracted_location": location}