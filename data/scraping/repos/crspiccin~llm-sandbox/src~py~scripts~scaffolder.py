from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    SystemMessage
)
import json
import os
from dotenv import load_dotenv
from pathlib import Path
load_dotenv(dotenv_path=Path("../.env"))


def load_config():
    with open('config.json', 'r') as file:
        config_file = json.load(file)
        print(json.dumps(config_file, indent=4))
        return config_file



# def run_chat_model():
#     print('starting the model ')
#     chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)

#     system_template = "Act as a seasoned senior software developer specialist in {language}"
#     human_template = "Create a {entity} entity following hexagonal architecture with services layer covered by unit test including operations like create, update, read and delete following SQL ANSI on a repository layer, and expose it on adapter layer by a {api_type} api."

#     chat_prompt = ChatPromptTemplate.from_messages([SystemMessagePromptTemplate.from_template(system_template), HumanMessagePromptTemplate.from_template(human_template)])

#     messages = chat_prompt.format_prompt(language="python", entity="Transaction", api_type="REST").to_messages()

#     messages.append(SystemMessage(content="On each file put BREAK at the beginning followed by filename"))

#     content = chat(messages).content
#     return content

def run_llm_chain_model(config_file):
    print('Starting the llm model ...')
    # llm = OpenAI(temperature=config_file['temperature'])
    chat = ChatOpenAI(temperature=config_file['temperature'])

    prompt = PromptTemplate(
        template = config_file['prompt_template'],
        input_variables=["language", "entity", "api_type"]
    )

    chain = LLMChain(llm=chat, prompt=prompt)

    result = chain.run({"language":config_file['language'], "entity":config_file['entity'], "api_type":config_file['api_type']})
    print(result)
    return result

output_dir =  "../../../output/"

def output_files(llm_output, output_dir):
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    files = llm_output.split("BREAK")
    for file in files:
        tokens = file.split("\n")
        for token in tokens:
            if token != '':
                print(token)
                # print(file.replace(token, ""))
                with open(os.path.join(output_dir, token) , 'w+') as fp:
                    fp.write(file.replace(token, ""))
                    pass
                break


config_file = load_config()
output_files(run_llm_chain_model(config_file), config_file['output_dir'])


