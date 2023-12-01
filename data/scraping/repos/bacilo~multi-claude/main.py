# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import anthropic_agent
from commands import COMMANDS, execute_command

def process_input(input_string, agent_manager):
    if input_string.startswith(tuple(alias for cmd in COMMANDS for alias in cmd['aliases'])):
        result = execute_command(input_string, agent_manager)
    else:
        agent = agent_manager.get_current_agent()
        response, _ = agent.get_ai_response(input_string)
        agent.update_conversation_history(input_string, response)
        result = response
    return result


def main():
    print("Type 'quit' to exit the conversation.")
    print("Commands:")
    for cmd in COMMANDS:
        print(cmd['help'])

    agent_manager = anthropic_agent.AgentManager()

    while True:
        user_input = input("You: ")

        if user_input.lower() == "quit":
            break

        if not user_input.strip():
            print("Please enter a valid command or prompt.")
            continue

        result = process_input(user_input, agent_manager)

        print(result)

if __name__ == "__main__":
    main()


