from langchain.tools import Tool, BaseTool
# from langchain.utilities import BashProcess

from prompt_toolkit.shortcuts import clear

ASSISTANT = "Assistant"

def exit_shell(tool_input: str | dict,
    verbose: bool | None = None,
    start_color: str | None = "green",
    color: str | None = "green",
    callbacks = None,
    **kwargs):
    #spinner.stop()
    if tool_input and tool_input != "None": print(f"[{ASSISTANT}] {tool_input}")
    exit(0)

class ClearScreenTool(BaseTool):
    name = "Clear"
    description = "useful when you need to clear the screen or start a fresh conversation. Don't forget to say something nice."
    
    def _run(self, input: str, **kwargs):
        #spinner.stop()
        clear()
        if input and input != "None": print(f"{ASSISTANT}: {input}")
    
    async def _arun(self):
        #spinner.stop()
        clear()

def shell_func(tool_input: str | dict,
    verbose: bool | None = None,
    start_color: str | None = "green",
    color: str | None = "green",
    callbacks = None,
    **kwargs):
    return "shell: error: not implemented"

def get_tool():
    # persistent_process = BashProcess(persistent=True)
    # shell = ShellTool(process=persistent_process)
    clear_tool = ClearScreenTool()
    return [
            Tool(
                name="Shell",
                func=shell_func,
                description="useful when you need to use the system to achieve something; input must be valid bash code; implemented using subprocess so no tty support. Use `gnome-terminal -- $SHELL -c '$YOUR_COMMANDS_HERE'` if you want to launch commands in a new window.",
            ),
            Tool(
                name="Exit",
                func=exit_shell,
                description="useful when you need to exit the shell or stop the conversation, dont forget to tell the user that you can't wait for your next conversation first.",
            ),
            Tool(
                name="Clear",
                func=clear_tool.run,
                description="useful when you need to clear the screen or start a fresh conversation. Don't forget to say something nice.",
            ),
        ]
