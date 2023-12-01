from sympy.ntheory import isprime
import math
from openai import OpenAI

# Initialize the OpenAI client with your secret key
client = OpenAI(api_key="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")

# Function to check if a number is a Carmichael number
def is_carmichael(n):
    if isprime(n) or n < 3:
        return False
    s = set()
    x = n - 1
    while x % 2 == 0:
        s.add(2)
        x //= 2
    for p in s:
        if pow(p, n - 1, n) != 1:
            return False
    return all(pow(a, n - 1, n) == 1 for a in range(2, n) if math.gcd(a, n) == 1)

# Function to check if a list of numbers are all Carmichael numbers
def are_carmichael_numbers(numbers):
    return all(is_carmichael(n) for n in numbers)



# Define the prompt message to generate code for Carmichael numbers
prompt_message = "Write a Python function that generates a list of Carmichael numbers less than 10000. your response should be only the code without starting by ```python and ending by ```"

number_of_calls = 100  # For example, let's call the API 5 times

# Loop to simulate calling the API the specified number of times
for i in range(number_of_calls):
    # Make the API call to generate the code
    try:
        response  = client.chat.completions.create(
            model="gpt-4-1106-preview",
            seed=30354229,
            messages=[
                {"role": "user", "content":prompt_message},
            ],
            temperature=0.1
        )
        generated_code = response.choices[0].message.content

        # Execute the generated code
        exec(generated_code)
        carmichael_numbers_generated_by_gpt = locals().get('carmichael_numbers', None)

        if carmichael_numbers_generated_by_gpt:
            # Call the function with the required 'limit' argument to get the list of Carmichael numbers
            # For example, let's use 10000 as the limit
            limit = 10000
            carmichael_numbers = carmichael_numbers_generated_by_gpt(limit)

            # Verify the generated Carmichael numbers
            if are_carmichael_numbers(carmichael_numbers):
                print(f"The {i} generated code correctly identified Carmichael numbers.")
            else:
                print(f"The generated code failed to correctly identify Carmichael numbers. {generated_code}")
        else:
            print(f"The generated code did not produce a variable named 'carmichael_numbers'. {generated_code}")
    except Exception as e:
        print(f"An error occurred: {e} code {generated_code}")
