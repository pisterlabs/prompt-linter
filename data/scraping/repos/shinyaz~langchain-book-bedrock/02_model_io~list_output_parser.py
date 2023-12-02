from langchain.chat_models import BedrockChat
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.schema import HumanMessage

output_parser = CommaSeparatedListOutputParser()

chat = BedrockChat(
    model_id="anthropic.claude-v2"
)

result = chat([
    HumanMessage(content="Appleが開発した代表的な製品を3つ教えてください"),
    HumanMessage(content=output_parser.get_format_instructions()),
])

output = output_parser.parse(result.content)

for item in output:
    print("代表的な製品 => " + item)
