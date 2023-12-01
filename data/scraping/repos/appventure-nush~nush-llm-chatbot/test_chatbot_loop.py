from llamaindex import chatbot_agent

while True:
    text_input = input("User: ")
    response = chatbot_agent.agent_chain.run(input=text_input)
    print(f'Agent: {response}')
