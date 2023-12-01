import openai
import time

with open('C:/Users/AZEEM/Desktop/API.txt') as file:
    api_key = file.read().strip()

openai.api_key = api_key
# WRITE YOUR CODE HERE

start_time = time.time()

prompts = [
    "Write a summary of a book about artificial intelligence.",
    "Explain the importance of machine learning in today's world.",
    "Describe the role of neural networks in natural language processing."
]

responses = []
total_tokens = 0
prompt_tokens = 0
completion_tokens = 0
error_count = 0

for prompt in prompts:
    try:
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            max_tokens=50,
            n=1,
            stop=None,
            temperature=0.7
        )
        responses.append(response.choices[0].text.strip())
        usage = response['usage']
        total_tokens += usage['total_tokens']
        prompt_tokens += usage['prompt_tokens']
        completion_tokens += usage['completion_tokens']
    except Exception as e:
        error_count += 1
        print(f"Error encountered for prompt '{prompt}': {e}")

end_time = time.time()
response_time = end_time - start_time

num_requests = len(prompts)
throughput = num_requests / response_time

error_rate = error_count / num_requests

# Iterate through the prompts and their corresponding responses
for i, (prompt, response) in enumerate(zip(prompts, responses)):
    print(f"Prompt {i + 1}: {prompt}")
    print(f"Response {i + 1}: {response}\n")

print(f"Response time: {response_time} seconds")
print(f"Throughput: {throughput} requests per second")
print(f"API Usage:")
print(f"  Total tokens: {total_tokens}")
print(f"  Prompt tokens: {prompt_tokens}")
print(f"  Completion tokens: {completion_tokens}")
print(f"Error rate: {error_rate}")