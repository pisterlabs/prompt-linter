import openai
import setcreds
import pyinotify

C_OUTFILE = "./ui/src/Main.js"

def generate_output(imports, body):
    output = f"""
// imports required by the code below are here
import * as React from 'react';
import {{ useState, useEffect }} from 'react';
// now import the React DnD library
// import {{ DraggableItem, DragDropContext, Droppable, Draggable }} from 'react-dnd';

{imports}

export const Main = () => {{
{body}
}}; 
"""

    # open the file and write the output
    with open(C_OUTFILE, "w") as f:
        f.write(output)

def call_openai_complete(prompt, stop_text):
    print("begin complete")
    response_text = ""

    hit_end = False
    max_iterations = 3

    # loop until we find the stop text 
    while not hit_end and (max_iterations > 0):
        print("completing...")
        completion = openai.Completion.create(
            engine="code-davinci-002",
            max_tokens=2048, 
            # logprobs=1,
            temperature=0.1,
            prompt=prompt + response_text,
            stop=stop_text
        )

        # stop if the completion is complete
        response_text += completion.choices[0].text

        print(f"text: {completion.choices[0].text}")

        # print (response_text)

        print (f"Finish reason: {completion.choices[0].finish_reason}")
        print (f"Logprobs: {completion.choices[0].logprobs}")

        # hit_end = completion.choices[0].finish_reason != 'length'

        max_iterations -= 1
    
    # # remove everything after the stop text
    # response_text = response_text.split(stop)[0]

    print(f"end complete: {response_text}")
    return response_text

# def generate_imports(spec_body, function_body):
#     spec_body = "\n".join(["// " + line for line in spec_body.split("\n")]) 

#     prompt = f"""
# // The Main() function is a react component. It does the following:
# {spec_body}
# //
# // The function is as follows:
# export const Main = () => {{
# {function_body}
# }}; 

# // Missing imports required for this function are as follows:
# import React from 'react';
# """

#     stop = "\n\n"

#     return call_openai_complete(prompt, stop)

def generate_body(spec_body):
    spec_body = "\n".join(["// " + line for line in spec_body.split("\n")]) 

    prompt = f"""
// The Main() function is a react component. It does the following:
{spec_body}

export const Main = () => {{
"""

    stop = "\n}"

    return call_openai_complete(prompt, stop)

def do_generate(spec):
    body = generate_body(spec)

    # imports = generate_imports(spec, body)
    imports = ""

    generate_output(imports, body)

class ModHandler(pyinotify.ProcessEvent):
    # evt has useful properties, including pathname
    prev_spec = None

    def process_IN_MODIFY(self, evt):
        with open("spec.txt", "r") as f:
            spec = f.read()

            if self.prev_spec != spec: 
                self.prev_spec = spec

                do_generate(spec)

def main2():
    # use pyinotify to wait for the file "spec.txt" to change.
    
    handler = ModHandler()
    wm = pyinotify.WatchManager()
    notifier = pyinotify.Notifier(wm, handler)
    wm.add_watch("spec.txt", pyinotify.IN_MODIFY)

    with open("spec.txt", "r") as f:
        spec = f.read()
        do_generate(spec)

    notifier.loop()





# # this is the main function
# def main():
#     clauses = []

#     # get a clause from the user
#     prompt = input("Enter a clause: ")



#     while prompt != "exit":
#         # add the clause to the list
#         clauses.append(prompt)

#         imports = """
# import React from 'react';
# """
#         body = generate_body(clauses)

#         generate_output(imports, body)

#         prompt = input("Enter a clause: ")

# call the main function
if __name__ == "__main__":
    main2()
