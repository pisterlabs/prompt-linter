from anthropic_agent import AgentManager

COMMANDS = [
    {'aliases': ['create_agent', 'ca'], 'function': AgentManager.create_agent, 'help': 'Create a new agent with an optional initial prompt: ca <agent_name> <optional_initial_prompt>'},
    {'aliases': ['switch_agent', 'sa'], 'function': AgentManager.switch_agent, 'help': 'Switch to an existing agent: sa <agent_name>'},
    {'aliases': ['talk', 't'], 'function': AgentManager.agents_conversation, 'help': 'Make two agents talk to each other for a given number of turns with an initial prompt: talk <agent1_name> <agent2_name> <initial_prompt>'},
    {'aliases': ['list_agents', 'la'], 'function': AgentManager.list_agents, 'help': 'List all existing agents: la'},
    {'aliases': ['remove_agent', 'ra'], 'function': AgentManager.remove_agent, 'help': 'Remove an existing agent: ra <agent_name>'},
]

def execute_command(input_string, agent_manager):
    command_parts = input_string.split(maxsplit=1)
    command = command_parts[0].lower()
    args = command_parts[1] if len(command_parts) > 1 else None

    cmd = None
    for command_data in COMMANDS:
        if command in command_data['aliases']:
            cmd = command_data
            break

    if cmd is None:
        return f"I'm sorry, I don't recognize the command '{command}'."

    return cmd['function'](agent_manager, args)
