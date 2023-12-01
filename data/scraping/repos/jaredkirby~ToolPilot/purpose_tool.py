from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)

from .base_tool import BaseTool


class PurposeTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="Prompt Purpose",
            model="gpt-3.5-turbo",
            temperature=0.75,
            uploads=None,
            inputs=[
                {
                    "input_label": "Prompt",
                    "example": "Write and edit scripts for Youtube videos",
                    "button_label": "Purpose",
                    "help_label": "The Prompt Purpose tool helps by reviewing a given prompt and generating a summary of the prompt's purpose.",
                }
            ],
        )

    def execute(self, chat, inputs):
        purpose_gen_template = f"""
You are a natural language processing researcher explaining the techniques for 
prompting large language models. 
Please write a short overview of the purpose of the following large language model 
prompting tool: 

--
"{inputs}"
--

Do not answer the prompt. 
Analyze the purpose of the prompt and develop a concise summary explanation.
    """

        user_prompt = HumanMessagePromptTemplate.from_template(purpose_gen_template)
        chat_prompt = ChatPromptTemplate.from_messages([user_prompt])
        formatted_prompt = chat_prompt.format_prompt(user_input=inputs).to_messages()
        llm = chat
        result = llm(formatted_prompt)
        return result.content
