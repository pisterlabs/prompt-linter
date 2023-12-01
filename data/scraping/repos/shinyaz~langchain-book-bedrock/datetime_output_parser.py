from langchain.prompts import PromptTemplate
from langchain.chat_models import BedrockChat
from langchain.output_parsers import DatetimeOutputParser
from langchain.schema import HumanMessage

output_parser = DatetimeOutputParser()

chat = BedrockChat(
    model_id="anthropic.claude-v2"
)

prompt = PromptTemplate.from_template("{product}のリリース日を教えて")

result = chat([
    HumanMessage(content=prompt.format(product="iPhone8")),
    HumanMessage(content=output_parser.get_format_instructions())
])

output = output_parser.parse(result.content)

print(output)
