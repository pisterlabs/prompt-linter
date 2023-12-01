import sys
import os
import openai
import dotenv

generatedDir    = "generated"
TURBO_MODEL     = "gpt-3.5-turbo"
EXPENSIVE_MODEL = "gpt-4"

def read_file(filename):
    with open(filename, "r") as file:
        return file.read()

def walk_directory( directory ):
    image_extensions = [ ".o .png", ".jpg", ".jpeg", ".gif", ".bmp", ".svg", ".ico", ".tif", ".tiff" ]
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
    code_contents = walk_directory( "/home/adamsl/linuxBash/SMOL_AI/tennis_unit_tests/test_templates" )
    # Now, `code_contents` is a dictionary that contains the content of all your non-image files
    # You can send this to OpenAI's text-davinci-003 for help

    context = "\n".join(
        f"{path}:\n{contents}" for path, contents in code_contents.items()
    )
     
    full_path_to_object = "/home/adamsl/linuxBash/SMOL_AI/tennis_unit_tests/Mode1Score/Mode1Score.cpp"
    object_to_test = read_file( full_path_to_object )
    system = """Act as a superintelligent C++ developer specializing in writing brilliant unit testing systems with the C++ Google Test framework. """
    prompt = (
        "Please write a unit test for a C++ Object using the following files as a template.\nMy files are as follows: \n```"
        + context
        + "```\n\n"
        + "The object to test is as follows:\n```" + object_to_test + "```\n\n" +
        "Please write the Makefile and C++ unit test file using the information provided." )
    
    # write system and prompt concatenated to a file
    file = open( "/home/adamsl/linuxBash/SMOL_AI/tennis_unit_tests/test_tuesday.md", "w" )
    file.write( system + "\n" + prompt )

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
    # if len(sys.argv) < 2:
    #     print("Please provide a prompt")
    #     sys.exit(1)
    # prompt = sys.argv[1]
    # read make_error_prompt.md into prompt variable
    full_path_to_propmt = "/home/adamsl/linuxBash/SMOL_AI/tennis_unit_tests/tuesday_debug.md"
    prompt = read_file( full_path_to_propmt )
    model = sys.argv[2] if len(sys.argv) > 2 else TURBO_MODEL 
    main( prompt, model )
