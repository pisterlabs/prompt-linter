"""
{prompt_content}
Current conversation:
{history}

Question: {input}

### Response:
""""""
{llama_instruction}
Continue the chat dialogue below. Write {character}'s next reply in a chat between User and {character}. Write a single reply only.

{llama_input}
Description:
{description}

Scenario:
{scenario}

Message Examples:
{mes_example}

Current conversation:
{history}

Question: {input}

{llama_response}
"""