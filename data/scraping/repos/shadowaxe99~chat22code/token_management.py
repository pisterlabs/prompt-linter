```python
import openai

def get_token_count(text):
    response = openai.Completion.create(
        engine="davinci-codex",
        prompt=text,
        max_tokens=0,
        log_level="info"
    )
    return response['usage']['total_tokens']

def get_message_token_count(message):
    return get_token_count(message['content'])

def get_chat_token_count(chat):
    token_count = 0
    for message in chat['messages']:
        token_count += get_message_token_count(message)
    return token_count

def get_chat_token_counts(chat_history):
    token_counts = []
    for chat in chat_history:
        token_count = get_chat_token_count(chat)
        token_counts.append(token_count)
    return token_counts

def display_token_counts(chat_history):
    token_counts = get_chat_token_counts(chat_history)
    for i, token_count in enumerate(token_counts):
        print(f"Chat {i+1} token count: {token_count}")

# Example usage
chat_history = [
    {
        'messages': [
            {'role': 'system', 'content': 'You are now connected to the chat.'},
            {'role': 'user', 'content': 'Hello, how are you?'},
            {'role': 'assistant', 'content': 'I am doing well, thank you.'}
        ]
    },
    {
        'messages': [
            {'role': 'system', 'content': 'You are now connected to the chat.'},
            {'role': 'user', 'content': 'What is the weather like today?'},
            {'role': 'assistant', 'content': 'The weather is sunny and warm.'}
        ]
    }
]

display_token_counts(chat_history)
```

This code defines functions to calculate the token count for a given text, message, or chat. The `get_token_count` function uses OpenAI's `Completion.create` method to get the token count for a given text prompt. The `get_message_token_count` function calculates the token count for a single message in a chat. The `get_chat_token_count` function calculates the total token count for a chat by summing the token counts of all messages. The `get_chat_token_counts` function calculates the token counts for multiple chats in a chat history. Finally, the `display_token_counts` function prints the token counts for each chat in the chat history.

You can use this code as a starting point to implement the token management feature in your MacOS application.