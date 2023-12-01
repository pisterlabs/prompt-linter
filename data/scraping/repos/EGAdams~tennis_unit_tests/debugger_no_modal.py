import sys
import os
import openai
import dotenv

generatedDir    = "generated"
TURBO_MODEL     = "gpt-3.5-turbo-16k"
EXPENSIVE_MODEL = "gpt-4"

def read_file(filename):
    with open(filename, "r") as file:
        return file.read()

def walk_directory( directory ):
    image_extensions = [ ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".svg", ".ico", ".tif", ".tiff" ]
    code_contents = {}
    for root, dirs, files in os.walk(directory):
        for file in files:
            if not any(file.endswith(ext) for ext in image_extensions):
                try:
                    relative_filepath = os.path.relpath(
                        os.path.join(root, file), directory
                    )
                    print( "Reading file: ", relative_filepath )
                    code_contents[relative_filepath] = read_file(
                        os.path.join(root, file)
                    )
                except Exception as e:
                    code_contents[
                        relative_filepath
                    ] = f"Error reading file {file}: {str(e)}"
    print( "Code contents: ", code_contents )
    return code_contents

def main(prompt, model=TURBO_MODEL ):
    code_contents = walk_directory( "/home/adamsl/linuxBash/SMOL_AI/tennis_unit_tests/Mode1Score" )
    # Now, `code_contents` is a dictionary that contains the content of all your non-image files
    # You can send this to OpenAI's text-davinci-003 for help

    context = "\n".join(
        f"{path}:\n{contents}" for path, contents in code_contents.items()
    )

    # system = "You are an AI debugger who is trying to debug a program for me based on the contents of the C++ header, source and Makefile on my file system. I am providing you with the following files and their contents, finally followed by the error message or issue that I am facing"

    # rephrased by g4...
    system = "You are an expert C++ Developer specializing in writting brilliant unit tests with the C++ Google framework.  Your task is to help me troubleshoot a problem with my C++ project build. This task will involve analyzing the C++ header files, source files, and Makefile present on my system. I will provide you with these files and their contents, as well as the error message or issue I'm encountering."

    prompt = (
        "My files are as follows: \n```"
        + context
        + "```\n\n"
        + "My issue is as follows: "
        + prompt
    )
    prompt += (
        "\n\nGive me ideas for what could be wrong and what fixes to do in which files."
    )

    # write system message to file
    with open( os.path.join(".", "system_prompt.md" ), "w" ) as file:
        file.write( system )

    # now append prompt to system message
    with open( os.path.join( ".", "system_prompt.md" ), "a" ) as file:
        file.write( prompt )

    res = generate_response(system, prompt, model)

    # print res in teal
    print("\033[96m" + res + "\033[0m")


def generate_response(system_prompt, user_prompt, model=TURBO_MODEL, *args):
    import openai

    # Set up your OpenAI API credentials
    openai.api_key = os.environ["OPENAI_API_KEY"]

def generate_response(system_prompt, user_prompt, model=TURBO_MODEL , *args):
    # import openai
    openai.api_key = dotenv.dotenv_values()["OPENAI_API_KEY"] # Set up your OpenAI API credentials
    # openai.api_key = ""
    messages = []
    messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    # loop thru each arg and add it to messages alternating role between "assistant" and "user"
    role = "assistant"
    for value in args:
        messages.append({"role": role, "content": value})
        role = "user" if role == "assistant" else "assistant"

    params = { "model": model, "messages": messages, "max_tokens": 1500, "temperature": 0 }
    response = openai.ChatCompletion.create(**params) # Send the API request
    reply = response.choices[0]["message"]["content"]
    print( "Reply: ", reply )
    return reply

if __name__ == "__main__":
    full_path_to_propmt = "/home/adamsl/linuxBash/SMOL_AI/tennis_unit_tests/make_error_prompt.md"
    prompt = read_file( full_path_to_propmt )
    model = sys.argv[2] if len(sys.argv) > 2 else TURBO_MODEL 
    main( prompt, model )
