import json
import os

temp = ["00","02","03","04","05","06","08","09","12","14","16","20","21","22","23","25","26","27","28","29","30","33","34","38","39","40"]

import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

for iii in temp:
    folderPath = 'generated_code/code_' + iii +'/mutants'
    fileList = os.listdir(folderPath)
    count = 0
    for f in fileList:
        count += 1
        if f == ".DS_Store":
            count -= 1
            continue
        # read the JSON data from file
        with open(os.path.join(folderPath,f), 'r',encoding='gbk') as file:
            data = file.read()

            prompts = []

            prompts.append(
            '''Is the following Python code buggy? Try to reason the code first. The bugs can be in different forms, for example, replaced arithmatic, replaced return value, replaced conditional boundary,etc.
            ```python
            %s
            ```
            '''
            % (data))

            prompts.append(
            '''
            Given the fact that the following Python code is buggy, Can you spot the statements involved in the bug? The bugs can be in different forms, for example, replaced arithmatic, replaced return value, replaced conditional boundary,etc. I care about the correctness.
            ```python
            %s
            ```
            '''
            % (data))


            import openai

            # Replace "your_api_key_here" with your actual API key
            openai.api_key = "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"

            # Initialize an empty list to store conversation history
            conversation_history = []

            @retry(wait=wait_random_exponential(min=1,max=60),stop=stop_after_attempt(6))
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

            # Example usage of the send_message_to_chatgpt function
            # import pyperclip
            chat = {}
            chatstr = ""
            for i,q in enumerate(prompts):

                chat.update([("Q" + str(i), q)])
                chatstr += "\n\n## user: \n\n" + q
                # pyperclip.copy(chatstr)

                print("\n\n## user: \n\n", q)

                a = send_message_to_chatgpt(q)

                chat.update([("A" + str(i), a)])
                chatstr += "\n\n## chatgpt: \n\n"+ a
                print("\n\n## chatgpt: \n\n", a)

            with open('buggy_results/code'+iii+'/'+str(count)+'.md', 'w') as file:
                file.write(chatstr)

            #save chatstr to markdown file
            # file.write(chatstr)

        # with open('chat_history/10.md', 'w') as file:
        #     file.write(chatstr)