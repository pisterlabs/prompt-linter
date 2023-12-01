from langchain.document_loaders import PyPDFLoader
from main import prompt, parser, chat_model

loader = PyPDFLoader("elon.pdf")
document = loader.load()

document_query = "Create a profile based on this description: " + document[0].page_content

_input = prompt.format_prompt(question=document_query)
output = chat_model(_input.to_messages())
parsed = parser.parse(output.content)

print(parsed)