# Example of an OpenAI Completion request, using the stream=True option
# https://beta.openai.com/docs/api-reference/completions/create

# record the time before the request is sent
#my test
import time
import openai

openai.api_key=""

start_time = time.time()

# send a Completion request to count to 100
 # Example of an OpenAI ChatCompletion request with stream=True
# https://platform.openai.com/docs/guides/chat

# record the time before the request is sent
start_time = time.time()

# send a ChatCompletion request to count to 100
response = openai.ChatCompletion.create(
    model='gpt-3.5-turbo',
    messages=[
        {'role': 'user', 'content': 'Count to 100, with a comma between each number and no newlines. E.g., 1, 2, 3, ...'}
    ],
    temperature=0,
    stream=True  # again, we set stream=True
)

# create variables to collect the stream of chunks
collected_chunks = []
collected_messages = []
# iterate through the stream of events
for chunk in response:
    chunk_time = time.time() - start_time  # calculate the time delay of the chunk
    collected_chunks.append(chunk)  # save the event response
    # chunk_message = chunk['choices'][0]['delta']  # extract the message
    # collected_messages.append(chunk_message)  # save the message
    try:
        chunk_message = chunk['choices'][0]['delta']['content']
        print(chunk_message,end="")  # print the delay and text
    except:
        pass

# print the time delay and text received
print(f"Full response received {chunk_time:.2f} seconds after request")