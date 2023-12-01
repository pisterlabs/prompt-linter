from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI

from langchain.llms import OpenAI


def write_to_file(file_path, text):
    with open(file_path, 'w') as file:
        file.write(text)

def read_file(file_path):
    try:
        with open(file_path, 'r') as file:
            contents = file.read()
            return contents
    except FileNotFoundError:
        print("File not found.")
    except IOError:
        print("Error reading the file.")

llm = OpenAI(model_name="gpt-3.5-turbo-16k-0613")

template = """
I want you format the following text making it very pretty in markdown: {text_blob}?
"""

prompt = PromptTemplate(
 input_variables=["text_blob"],
 template=template,
)


chain = ChatOpenAI(llm = llm, 
                  prompt = prompt)

file_path = "/tank0/data/Summaries/docs.flowiseai.com__deployment_digital-ocean.txt"  # Replace with the actual file path

content_to_format = read_file(file_path)


num_tokens = llm.get_num_tokens(content_to_format)


# Run the chain only specifying the input variable.
formatted_output = chain.run(content_to_format)

file_path = '/tank0/data/Summaries/docs.flowiseai.com__deployment_digital-ocean.md'

write_to_file(file_path, formatted_output)
