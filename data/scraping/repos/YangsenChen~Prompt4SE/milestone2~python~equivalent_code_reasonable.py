import json

# read the JSON data from file
with open('reasonable_dataset/reasonable.json', 'r') as file:
    data = json.load(file)

value = data['function']
import openai
inn = 1

# Replace "your_api_key_here" with your actual API key
openai.api_key = "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"

def send_message_to_chatgpt(message):
    # conversation_history.append({'role': 'system', 'content': 'start the chat'})
    conversation_history.append({'role': 'user', 'content': message})

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
                   {'role': 'system', 'content': 'You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture.'}
               ] + conversation_history
    )

    assistant_message = response.choices[0].message.content
    conversation_history.append({'role': 'assistant', 'content': assistant_message})
    return assistant_message

for method in value:
    # prompts = []
    #
    # prompts.append(
    # '''Now with the following code
    # ```python
    # %s
    # ```
    # Can you generate another code that is different from this one, but produces the same output given the same input?'''
    # % (method))
    # conversation_history = []
    #
    # chat = {}
    # chatstr = ""
    # for i, q in enumerate(prompts):
    #     chat.update([("Q" + str(i), q)])
    #     chatstr += "\n\n## user: \n\n" + q
    #     # pyperclip.copy(chatstr)
    #
    #     print("\n\n## user: \n\n", q)
    #
    #     a = send_message_to_chatgpt(q)
    #
    #     chat.update([("A" + str(i), a)])
    #     chatstr += "\n\n## chatgpt: \n\n" + a
    #     print("\n\n## chatgpt: \n\n", a)
    #
    # with open('chat_history/'+str(inn)+'_task3.md', 'w') as file:
    #     file.write(chatstr)

    prompts = []
    prompts.append(
    '''
Create a model that can generate semantically equivalent Python code for any given input. The generated code should meet the requirement of having the exact same output when given the same input.

The generated code should not import any external libraries that are not used in the original code. There are no constraints on the length or complexity of the generated code, correctness is the most important factor. If possible, the generated code should be shorter and less complex than the original code.

Here's the code I want you to generate about:
    ```python
    %s
    ```
    '''
    % (method))
    conversation_history = []

    chat = {}
    chatstr = ""
    for i, q in enumerate(prompts):
        chat.update([("Q" + str(i), q)])
        chatstr += "\n\n## user: \n\n" + q
        # pyperclip.copy(chatstr)

        print("\n\n## user: \n\n", q)

        a = send_message_to_chatgpt(q)

        chat.update([("A" + str(i), a)])
        chatstr += "\n\n## chatgpt: \n\n" + a
        print("\n\n## chatgpt: \n\n", a)

    with open('chat_history/'+str(inn)+'_task3_complicated.md', 'w') as file:
        file.write(chatstr)
    inn += 1


