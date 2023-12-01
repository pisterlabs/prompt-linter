import json
from langchain.prompts.prompt import PromptTemplate
from ..utils.file import FileUtils

class HollisPrompt():
    def __init__(self):
        self.file_utils = FileUtils()
        # https://github.com/langchain-ai/langchain/blob/3d74d5e24dd62bb3878fe34de5f9eefa6d1d26c7/libs/langchain/langchain/chains/api/prompt.py#L4
        self.hollis_template = """\n\nHuman:
            You are given a user question asking for help to find books on certain subject available at select libraries at Harvard.\n
            Based on the question, create a JSON object with properties, 'keywords' with a list of keywords and 'libraries' with list of the Library Codes for the requested libraries.\n
            Please follow these instructions inside the <instructions> XML tags to create the JSON object.\n
            <instructions>\n
            To create the keywords list:\n
            The keywords property must contain a list of keywords relevant to the question.\n
            If you cannot find any keywords in the user question, do not make them up, the keywords list should be empty.\n
            Please make a distinction between the subject of the books, how the user intends to use the books, and the description of the libraries.
            Exclude keywords related to how the user intends to use the books e.g. 'research' or 'study'.\n
            Exclude keywords related to the description of the libraries e.g. 'the science library' should not include the keyword 'science'.\n
            Exclude any keywords that could be considered harmful, offensive, or inappropriate in a professional and educational environment.
            Please follow these instructions to create the libraries list:\n
            The 'libraries' property must contain a list of ALL the three-letter Library Codes from the 'libraryCode' property in the Libraries JSON file.\n\n
            If and ONLY IF the user mentions certain libraries in the question, the list must have ONLY the Library Codes mentioned.\n
            Use both the 'primoDisplayName' and 'howUsersMayRefer' properties in the Libraries JSON file to find the corresponding library codes based on the user question.\n
            If the user does not mention any libraries, you MUST include ALL the Library Codes.\n
            Libraries JSON file:{libraries_json}\n
            Please follow these instructions to create the JSON object result:\n
            You must return a single valid json object ONLY and nothing more. Do not return any additional text.\n
            The response must be a valid JSON object.\n
            Do not include any explanations, ONLY provide a RFC8259 compliant JSON response:{example_query_result_json}\n
            </instructions>\n
            Here are some example responses inside the <example> XML tags:\n
            <example>{example_1_json}</example>\n
            <example>{example_2_json}</example>\n
            <example>{example_3_json}</example>\n
            You must generate your response based on the following the user question inside the <user_question> XML tags:\n
            <user_question>{human_input_text}</user_question>\n
            \n\nAssistant:
            """

        self.example_1_json = {"keywords":[],"libraries":["AJP","BAK","CAB","DES","DIV","FAL","FUN","GUT","HYL","KSG","LAW","LAM","MUS","SEC","TOZ","WID"]}
        self.example_2_json = {"keywords":["economics"],"libraries":["BAK","LAM","WID"]}
        self.example_3_json = {"keywords":["copyrights"],"libraries":["LAW"]}

        self.hollis_no_keywords_template = """\n\nHuman:
            You are a friendly assistant whose purpose is to carry on a conversation with a user, in order to help them find books at libraries.\n
            Please follow these instructions inside the <instructions> XML tags to create the JSON object.\n
            <instructions>\n         
            You MUST answer the user question to the best of your ability.\n
            If the user did provide enough information about the books they want, suggest this specific example template question.\n
            Example template question inside the <example_question> XML tags: <example_question>I'm looking for books about "subjects" available at "one or more of the Harvard libraries".</example_question>\n
            </instructions>\n
            Please use the current conversation inside the <current_conversation> XML tags as your reference for the current conversation:\n
            <current_conversation>{history}</current_conversation>\n
            You must generate your response based on the following the user question inside the <user_question> XML tags:\n
            <user_question>{input}</user_question>\n
            Please do not include XML tags in your response.\n
            \n\nAssistant:
            """

        self.example_query_result_json = {"keywords":["string"],"libraries":["string"]}
        self.hollis_prompt_template = PromptTemplate.from_template(template=self.hollis_template)
        self.hollis_no_keywords_prompt_template = PromptTemplate(input_variables=['input', 'history'], template=self.hollis_no_keywords_template)

    async def get_hollis_prompt_formatted(self, human_input_text):
        self.libraries_json = await self.file_utils.get_libraries_csv()
        # format the prompt to add variable values
        hollis_prompt_formatted: str = self.hollis_prompt_template.format(
            human_input_text=human_input_text,
            libraries_json=json.dumps(self.libraries_json),
            example_query_result_json=json.dumps(self.example_query_result_json),
            example_1_json=json.dumps(self.example_1_json),
            example_2_json=json.dumps(self.example_2_json),
            example_3_json=json.dumps(self.example_3_json)
        )
        return hollis_prompt_formatted

    async def get_hollis_no_keywords_prompt(self):
      return self.hollis_no_keywords_prompt_template
