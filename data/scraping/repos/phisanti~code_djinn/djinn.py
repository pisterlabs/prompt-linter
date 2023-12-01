from typing import Optional
from dotenv import dotenv_values, set_key
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import DeepInfra
from .utils import get_os_info

class djinn():
    """
    The djinn class is the main class of the codedjinn package. It is used to interact with the DeepInfra API and generate commands.
    """
    def __init__(self, 
                 os_fullname: Optional[str] = None, 
                 shell: Optional[str] = None, 
                 api: Optional[str] = None):
        """
        The constructor for the djinn class. It takes the following parameters:
        os_fullname: The full name of the operating system. If not provided, it will be automatically detected.
        shell: The shell used by the user. If not provided, it will be automatically detected.
        api: The API key for the DeepInfra API. If not provided, it will be automatically detected from the .env file.
        """
        
        if os_fullname is None or shell is None:
            os_fullname, shell = get_os_info()
        if api is not None:
            config=dotenv_values()
            api=config['DEEPINFRA_API_TOKEN']
        self.os_fullname = os_fullname
        self.shell = shell
        self.llm = DeepInfra(model_id="mistralai/Mistral-7B-Instruct-v0.1", 
                             deepinfra_api_token=api)

        return None
        
    def _build_prompt(self, explain: bool = False):
        """
        This function builds the prompt for the DeepInfra API. It takes the following parameters:
        explain: A boolean value that indicates whether the user wants to provide an explanation of how the command works. If True, the prompt will include a description of the command.
        """

        explain_text = ""
        format_text = "Command: <insert_command_here>"
        os_fullname = self.os_fullname
        shell = self.shell

        if explain:
            explain_text = "Also, provide a brief and concise description of how the command works."
            format_text += "\nDescription: <insert_description_here>"
        format_text += "\nDo not enclose the command with extra quotes or backticks."

        template = f"""Instructions: Write a CLI command that does the following: {{wish}}. Make sure the command is correct and works on {os_fullname} using {shell}. {explain_text}Format: {format_text} \nPlease adhere strictly to the format provided above."""
        prompt_variables = ["wish"]
        prompt = PromptTemplate(template=template, input_variables=prompt_variables)
        return prompt

    def test_prompt(self, wish: str, explain: bool = False):
        """
        This function builds the prompt for the DeepInfra API. It takes the following parameters:
        wish: The command the user wants to generate.
        explain: A boolean value that indicates whether the user wants to provide an explanation of how the command works. If True, the prompt will include a description of the command.
        """
        prompt = self._build_prompt(explain)
        promt_text = prompt.format(wish = wish)

        return promt_text

    def ask(self, wish: str, explain: bool = False, llm_verbose: bool = False):

        """
        This function generates a command using the DeepInfra API. It takes the following parameters:
        wish: The command the user wants to generate.
        explain: A boolean value that indicates whether the user wants to provide an explanation of how the command works. If True, the prompt will include a description of the command.
        llm_verbose: A boolean value that indicates whether the user wants to see the output of the LLM model. If True, the output of the LLM model will be printed.
        """
        
        if explain:
            max_tokens = 1000
        else:
            max_tokens = 250
        
        self.llm.model_kwargs = {'temperature': 0.7, 'repetition_penalty': 1.2, 
                            'max_new_tokens': max_tokens, 'top_p': 0.9}
        prompt = self._build_prompt(explain)

        llm_chain = LLMChain(prompt=prompt,llm=self.llm, verbose=llm_verbose)
        response = llm_chain.run(wish)
        responses_items = response.strip().split("\n")

        command = None
        description = None
        for element in responses_items:
            if "command:" in element.lower():
                command = element
                command.replace("Command: ", "").strip()

            elif "description:" in element.lower():
                description = element
                description.replace("Description: ", "").strip()

            if command is not None and description is not None:

                return response

        return command, description