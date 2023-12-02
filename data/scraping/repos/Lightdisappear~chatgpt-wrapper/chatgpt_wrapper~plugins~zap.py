from langchain.llms import OpenAI
from langchain.agents import initialize_agent
from langchain.agents.agent_toolkits import ZapierToolkit
from langchain.utilities.zapier import ZapierNLAWrapper

from chatgpt_wrapper.plugin import Plugin
import chatgpt_wrapper.debug as debug
if False:
    debug.console(None)

class Zap(Plugin):

    def setup(self):
        self.log.info(f"Setting up zap plugin, running with backend: {self.backend.name}")
        self.llm = OpenAI(temperature=0)
        self.zapier = ZapierNLAWrapper()
        self.toolkit = ZapierToolkit.from_zapier_nla_wrapper(self.zapier)
        self.agent = initialize_agent(self.toolkit.get_tools(), self.llm, agent="zero-shot-react-description", verbose=True)

    # def get_shell_completions(self, _base_shell_completions):
    #     commands = {}
    #     commands[self.shell.command_with_leader('test')] = self.shell.list_to_completion_hash(['one', 'two', 'three'])
    #     return commands

    async def do_zap(self, arg):
        """
        Send natural language commands to Zapier actions

        Requires exporting a Zapier Personal API Key into the following environment variable:
            ZAPIER_NLA_API_KEY

        To learn more: https://nla.zapier.com/get-started/

        Arguments:
            command: The natural language command to send to Zapier.

        Examples:
            {COMMAND} send an email to foo@bar.com with a random top 10 list
        """
        if not arg:
            return False, arg, "Command is required"
        try:
            self.agent.run(arg)
        except ValueError as e:
            return False, arg, e
        return True, arg, "Zap run completed"
