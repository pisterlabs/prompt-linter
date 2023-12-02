from config import OPEN_AI_API_KEY
from langchain.llms import OpenAI

llm = OpenAI(openai_api_key=OPEN_AI_API_KEY, model="text-davinci-002")


# class JSONOutputParser(BaseOutputParser):
#     """Parse the output of an LLM call to a string."""

#     def parse(self, stringifiedObj: str) -> str:
#         resultObj: dict = json.loads(stringifiedObj)
#         return resultObj.get("result")


# def generatePrompt(threads_list: list):
#     # template = """
#     # You will be given a list of social media posts text from a user,\
#     # and then you should perform personality test for this user based on the post contents given.\
#     # Generate your response in the form of a stringified JSON with only one field: 'result'\
#     # Example: \{"result": "Some personality analysis"\}

#     # posts:
#     # ```
#     # {threads}
#     # ```
#     # """
#     # prompt = PromptTemplate.from_template(template)
#     # prompt.format(threads=threads_list)

#     return prompt


def analyze_personality_by_threads(threads_list: list) -> str:
    stringified_threads = str(threads_list)
    template = f"Analyze the MBTI personality of the author of the following social media posts.\
        Use 'You' as the subject in your response, as if the author is me.\
        Posts: \
            ```{stringified_threads}```\
        "
    output = llm.predict(template)
    return output
