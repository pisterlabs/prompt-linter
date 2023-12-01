import openai
import json
import idc
from idc import *
import idaapi
from idautils import *


openai.api_key = "" # modify it! Refer https://mp.weixin.qq.com/s/E4n63jltBPbAo8ZIMH10ig to register openai.

# Get a list of all functions in the binary
functions = Functions()

MAX_LINE_TOKEN = 60

class ExplainHandler(idaapi.action_handler_t):

    def __init__(self):
        idaapi.action_handler_t.__init__(self)
    def activate(self, ctx):
        search_noname_function()
        return True
    def update(self, ctx):
        return idaapi.AST_ENABLE_ALWAYS

class OpenAIVeryGOOD(idaapi.plugin_t):
    explain_action_name = "OpenAIVeryGOOD:rename_function"

    explain_menu_path = "Edit/OpenAIVeryGOOD/Please help me Auto Recover Sub Function"

    explain_action = idaapi.action_desc_t(explain_action_name,
                                    'Please help me Auto Recover Sub Function',
                                    ExplainHandler(),
                                    "Ctrl+Alt+K",
                                    'Use davinci-003 to explain the currently selected function',
                                    199)

    idaapi.register_action(explain_action)

    idaapi.attach_action_to_menu(explain_menu_path, explain_action_name, idaapi.SETMENU_APP)


def query_model(pseudocode):
    query = "Can you explain what the following C function does and suggest a better name for it?\n%s"%(pseudocode)
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=query,
        temperature=0.6,
        max_tokens=2500,
        top_p=1,
        frequency_penalty=1,
        presence_penalty=1,
        timeout=60  # Wait 60 seconds maximum
    )
    jobj = json.loads(str(response))

    func_comment,func_name = jobj['choices'][0]['text'],jobj['choices'][0]['text'].split()[-1]
    return (func_comment,func_name)


def search_noname_function():
    # Iterate over the functions
    for f in functions:
        # Get the name of the function
        name = GetFunctionName(f)

        # Check if the name starts with "sub_"
        if name.startswith("sub_"):
            # Print the function name and its address
            
            # Use the idaapi.decompile function to get the pseudocode
            pseudocode = idaapi.decompile(f)

            # Print the pseudocode
            #print(pseudocode)

            # Count the number of lines in the pseudocode
            lines = str(pseudocode).split("\n")
            
            if(len(lines)<MAX_LINE_TOKEN):
                func_comment,new_func_name = query_model(pseudocode)
                new_func_name = new_func_name.replace("\"","").replace("(","").replace(")","").replace("'","").replace(".","")
                print("Function {} found at 0x{:X}, ready rename function: {}".format(name, f, new_func_name))
                MakeName(eval("0x{:X}".format(f)),new_func_name)

                new_cmt = ""
                for i in range(len(func_comment)//100+1):
                    new_cmt += func_comment[i*100:(i+1)*100]+"\n"
                set_func_cmt(eval("0x{:X}".format(f)),new_cmt,0)
            else:
                print("[-] Max line limited, Pass!")


def PLUGIN_ENTRY():
    OpenAIVeryGOOD()
