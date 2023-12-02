from langchain.tools import ShellTool

from python.gptsql.gpttool import GptTool


class DBToolCaller:
    def __init__(self):
        self.shell_tool = ShellTool()
        self.dbtool = GptTool("gptsql")
        print(self.shell_tool.run({"commands": ["echo 'Hello World!'", "printenv"]}))

    def put(self, link_key="default_link_key", key="default_key", value="default_value"):
        self.dbtool.put(link_key, key, value)
        this_command = f'echo DBTool(\'gptsql\').put(\'{link_key}\', \'{key}\', \'{value}\')'
        print(self.shell_tool.run({"commands": [this_command]}))


def main():
    dbtoolcaller = DBToolCaller()
    dbtoolcaller.put("link_key", "key", "value")


if __name__ == "__main__":
    main()

