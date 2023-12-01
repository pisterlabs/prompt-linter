from dotenv import load_dotenv
import guidance

load_dotenv()


guidance.llm = guidance.llms.OpenAI("gpt-3.5-turbo")

program= guidance(
""" 
{{#systum}} YOu are cs professor teaching {{os}} systums to your students
{{/systum}}

{{#user~}}
what are some of the most common used in the {{os}} op? provide LIst  the commonds and thier description one pre line . Number them from 1.
{{~/user}}

{{#assistant~}}
{{gen 'commands' max_tokens=100}}
{{~/assistant}}
"""
)

result=program(os="Linux")
print(result)