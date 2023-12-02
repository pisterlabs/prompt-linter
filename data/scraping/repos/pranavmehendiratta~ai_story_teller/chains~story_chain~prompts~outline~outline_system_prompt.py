from langchain.prompts import StringPromptTemplate


system_prompt_template_v0 = """You are writer who is reasearching for a topic. You are reading a {document_type} for which you want to write a detailed outline. You will be given content of the document in between three back ticks (```).

Output as a JSON:
{{
    title: write a descriptive title for the document,
    sections: [ // list of all the sections
        {{
            "section_title": write a descriptive title for this section
            "section_subtitles": list of descriptive subtitles to cover in this section
        }}
    ]

}}"""

input_variables_v0 = ["document_type"]

class SystemPromptTemplate(StringPromptTemplate):
    template: str

    def format(self, **kwargs):
        kwargs.update(self.partial_variables)
        return self.template.format(**kwargs)

system_prompt = SystemPromptTemplate(
    template = system_prompt_template_v0,
    input_variables = input_variables_v0,
    partial_variables = {}
)   