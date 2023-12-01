import openai


prompt_template = """You are a code interpreter specializing in shell commands. Your role is to analyze the given task and provide the appropriate shell command to complete it. Remember, your response should exclusively consist of the shell command required to perform the task. Avoid including any extraneous information or explanations.

Example:

Task: Create a new directory named 'projects' in the current location.

Shell Command: `mkdir projects`


New Task: {task}

Shell Command:"""


class ShellAgent:
    def __init__(self, model_name="gpt-3.5-turbo"):
        self.client = openai.ChatCompletion
        self.shell = self.init_shell_engine()
        self.model_name = model_name
        self.template = prompt_template

    def init_shell_engine(self):
        from langchain.tools import ShellTool
        shell_tool = ShellTool()

        return shell_tool

    def run(self, input):
        
        prompts = self.template.format(task=input)
        completion = self.client.create(model=self.model_name, messages=[{"role": "user", "content": prompts}])
        content = completion.choices[0].message.content

        try:
            shell_command = content.split("Shell Command:")[-1].strip()
            res = self.shell.run(shell_command)
            return res
        except Exception as e:
            return f"{input} raised error: {e}. Please try again with a valid shell command"