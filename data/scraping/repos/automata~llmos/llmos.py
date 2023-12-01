import re

from openai import OpenAI

from prompts import OS_SYSTEM

class Component:
    """
    Super to define some interfaces for components.
    """
    def step(self, cmd):
        return self.wrap("Not implemented yet.")

    def wrap(self, response):
        return f"{self.label} {response}"

class OS(Component):
    def __init__(self):
        self.label = "[LLMOS]"
        self.client = OpenAI()
        self.model = "gpt-4-1106-preview"
        self.system_prompt = OS_SYSTEM
        self.messages = [
            {
                "role": "system",
                "content": self.system_prompt 
            }
        ]

    def step(self, next_cmd):
        self.messages.append({
            "role": "user",
            "content": next_cmd
        })
        return self.query()

    def query(self):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            temperature=1,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return str(response.choices[0].message.content)

class CPU(Component):
    def __init__(self):
        self.label = "[CPU]"

    def step(self, cmd):
        return self.wrap(eval(cmd))

    def wrap(self, response):
        return f"{self.label} {response}"

class Mem(Component):
    def __init__(self):
        self.label = "[MEM]"
        self.data = {}

class FS(Component):
    def __init__(self):
        self.label = "[FS]"

class IO(Component):
    def __init__(self):
        self.label = "[IO]"

def parse(content):
    search_res = re.search("\[.*\]", content)
    start, end = search_res.span()
    component_name = content[start:end][1:-1]
    cmd = content[end+1:]
    return component_name, cmd

def run():
    os = OS()
    cpu = CPU()
    mem = Mem()
    fs = FS()
    io = IO()

    # Kick off
    response = "Start"
    while True:
        # Tick the OS (yeah, it's probably a CPU than OS?)
        response = os.step(response)
        component_name, cmd = parse(response)
        print("<", component_name, cmd)
        if component_name == "CPU":
            response = cpu.step(cmd)
        elif component_name == "MEM":
            response = mem.step(cmd)
        elif component_name == "FS":
            response = fs.step(cmd)
        elif component_name == "IO":
            response = io.step()
        else:
            user_input = input("> ")
            if user_input:
                response = user_input

if __name__ == "__main__":
    run()