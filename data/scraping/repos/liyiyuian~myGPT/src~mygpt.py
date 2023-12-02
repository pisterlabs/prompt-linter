from openai import OpenAI
import os
import cowsay
from assistant import *
from chat import *
from args import *
from utils import *
# from image import *


# greeting words in the begining of the program
def greeting():
    print(cowsay.get_output_string('fox',
                                   "Welcome to mygpt! what's in your mind?"))


# get_assistant use openai client to load the assistants created and exist
# on the server
def get_assistant():
    client = OpenAI()
    my_assistants = client.beta.assistants.list(
            order="desc",
            limit="20",
            )
    if len(my_assistants.data) > 0:
        options = ""
        for i, a in enumerate(my_assistants.data):
            tools = "+".join([tool.type for tool in a.tools])
            options += f"[{i}] name: {a.name}, model: {a.model}, "
            options += f"tools: {tools}, description: {a.description}\n"
        print("\n> There are a few existing assistants.")
        selection = int(input("> Which assistant do you want to retrieve?" +
                              " select 99 if you want to create a new one.\n" +
                              options +
                              "\n> your option: "))
        return my_assistants.data[selection] if selection != 99 else False

    else:
        print("there is no existing assistant. let's create one!")
        return False


# file_not_found return true if on of the file does not exist, otherwise, true
def file_not_found(filepaths):
    for file in filepaths.strip().split():
        if not os.path.exists(file):
            print(f"{file} not found!")
            return True
    return False


# clean_filepath clean the format. naive way, can be improved.
def clean_filepath(filepaths):
    out = []
    for file in filepaths.strip().split():
        out.append(file)
    return out


# clean_args return the arguements with default values or cleaned format
def clean_args(name, instructions, filepaths, ofilename, model, tools, des):
    if not name.strip():
        name = "default_name"
    if not instructions.strip():
        instructions = """You are a helpful assistant. try to answer the
        question to you best knowledge. try to answer it step-by-step and
        provide a summary in the end."""

    if not ofilename.strip():
        ofilename = "default_name_output.md"

    if not filepaths.strip():
        filepaths = []
    elif file_not_found(filepaths):
        filepaths = []
    else:
        filepaths = clean_filepath(filepaths)

    model_options = ["gpt-3.5-turbo-1106", "gpt-4-1106-preview"]
    if not model.strip():
        model = model_options[0]
    elif model.strip() not in model_options:
        print(f"> The model you provided is not in {model_options}")
        print(f"> Change to default {model_options[0]}")
        model = model_options[0]

    tool_options = ["retrieval", "code_interpreter"]
    if not tools.strip():
        tools = [{"type": "retrieval"}]
    elif (opt := set(tools.strip().split()) & set(tool_options)):
        tools = []
        for tool in opt:
            tools.append({"type": tool})

    print("\n###################################################")
    print("> Creating an Assistant with the Following Options")
    print("###################################################")
    print(f"> Name: {name.strip()}")
    print(f"> Instructions: {instructions.strip()}")
    print(f"> Upload files: {filepaths}")
    print(f"> Output filename: {ofilename.strip()}")
    print(f"> Model: {model.strip()}")
    print(f"> Tools: {tools}")
    print(f"> Description: {des.strip()}")
    print("###################################################")

    return name.strip(), instructions.strip(), filepaths, ofilename.strip(), model.strip(), tools, des.strip()


def run_assistant():
    if (assistant := get_assistant()):
        message = input("> what do you want to ask this assistant to do: ")
        ans_instruct = input("> Answer instruction: ")
        ofilename = input("> Output markdown filename: ")
        Assistant(assistant=assistant,
                  message=message,
                  ans_instructions=ans_instruct,
                  ofilename=ofilename)
    else:
        assistant = Assistant()
        print("> Okay! Let's create a new assistant! press enter to skip the question.")
        name = input("> Name your assistant (no space): ")
        instructions = input("> Customize instructions: ")
        filepaths = input("> Upload files: ")
        ofilename = input("> Output markdown filename: ")
        model = input("""
> Model (default: gpt-3.5-turbo-1106, try gpt-4-1106-preview): """)
        tools = input("> Tools (retrieval and/or code_interpreter): ")
        des = input("> Description: ")

        name, instructions, filepaths, ofilename, model, tools, des = clean_args(name, instructions, filepaths, ofilename, model, tools, des)

        assistant.init_assistant(name=name,
                                 instructions=instructions,
                                 description=des.strip(),
                                 filepaths=filepaths,
                                 ofilename=ofilename,
                                 model=model,
                                 tools=tools)
        print(f"> Assistant {name} created.")
        print("###################################################")
        question = input("> What do you want to ask this assistant to do: ")
        assistant.add_message(question)
        assistant.runjob = assistant.run(assistant.assistant.id)
        results = assistant.check_status()

        messages = []
        for i, data in reversed(list(enumerate(results.data))):
            messages.append(f"{i}: {data.content[0].text.value.strip()}")
        time.sleep(10)
        assistant.output_md(messages)

        delete_files = input("> Do you want to remove upload files (y/n)? ")
        delete_files = True if delete_files == 'y' else False
        if delete_files:
            print("> Deleting files...")
            assistant.del_file()
        del_assistant = input("> Do you want to remove this assistant (y/n)?")
        del_assistant = True if del_assistant == 'y' else False
        if del_assistant:
            assistant.del_assistant()
    os.system(f'mdcat {ofilename}')
    print(f"> Output is written to {ofilename}")


def main():
    args = parse_args()
    greeting()
    if args.mode[0] == 'chat':
        user_query()
    elif args.mode[0] == 'assistant':
        run_assistant()
        # todo: check mkcat installation

    elif args.mode[0] == 'image':
        pass
    else:
        print("> Entering Other Modes...")
        if args.delete_all:
            delete_all()


if __name__ == '__main__':
    main()
