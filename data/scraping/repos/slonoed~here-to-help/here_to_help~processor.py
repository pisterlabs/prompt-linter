import guidance
import readline

default_model = 'gpt-4'

interactive_suffix = """
{{~#geneach 'chat' stop=False}}
{{#user~}}
{{set 'this.input' (await 'input') hidden=False}}
{{~/user}}
{{#assistant~}}
{{gen 'this.output' temperature=0 max_tokens=300}}
{{~/assistant}}
{{~/geneach}}
"""

def get_model(name):
    match name:
        case "gpt-3.5-turbo":
            return guidance.llms.OpenAI("gpt-3.5-turbo")
        case "gpt-4":
            return guidance.llms.OpenAI("gpt-4")
    exit(f"Model not found: {name}")

def run(prompt, name_values, force_interactive=False):
    content = prompt['content']
    if not prompt.get('_interactive', False) and force_interactive:
        content = content + "\n" + interactive_suffix

    llm = get_model(prompt.get('model', default_model))
    program = guidance(content, llm=llm)
    out = ""
    def r(s):
        nonlocal out
        out = out + "\n\n" + s

    name_values['out'] = r

    program_args = {key: value for key, value in name_values.items()}
    program = program(**program_args)

    # Run infinite loop when interactive mode is enabled
    if prompt.get('_interactive', False) or force_interactive:
        if out != "":
            print(out)
        elif program.variables().get('output') != None:
            print(program.variables()['output'])

        while True:
            i = input(f"> ")
            text = program.__str__()
            program = guidance(text, **program.variables())
            program = program(input=i)
            print(program['chat'][-2].get('output', '<NONE>'))

    if out != "":
        return out
    elif program.variables().get('output') != None:
        return (program.variables()['output'])
    else:
        return (program)