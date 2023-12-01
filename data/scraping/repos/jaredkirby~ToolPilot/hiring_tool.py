from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)

from .base_tool import BaseTool


class HiringTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="Resume Analysis",
            model="gpt-3.5-turbo",
            temperature=0.75,
            file_inputs=[
                {
                    "input_label": "Upload Resume",
                    "help_label": "Upload a resume to analyse.",
                },
            ],
            inputs=None,
        )

    def execute(self, chat, inputs):
        template = f"""\
Please act as a professional and very skilled employee recruiter preparing for an applicant interview for an open role as CPG Account Strategist.

Here is the applicant's resume: 

---

{resume}


---

Here is the job description we are hiring for:

---
    """

        user_prompt = HumanMessagePromptTemplate.from_template(template)
        chat_prompt = ChatPromptTemplate.from_messages([user_prompt])
        formatted_prompt = chat_prompt.format_prompt(user_input=inputs).to_messages()
        llm = chat
        result = llm(formatted_prompt)
        return result.content
