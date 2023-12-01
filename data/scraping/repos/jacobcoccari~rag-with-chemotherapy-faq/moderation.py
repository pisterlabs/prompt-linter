from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

def harmful_content_check(message):
    client = OpenAI()
    response = client.moderations.create(input=message)
    output = response.results[0]
    flagged_problems = ""
    if output.flagged == True:
        attributes = vars(output.categories)
        for key in attributes:
            if attributes[key] == True:
                flagged_problems += key + ", "
        return "This message has been flagged for the following issues: " + flagged_problems
    else:
        return 