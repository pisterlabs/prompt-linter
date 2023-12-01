from openai import OpenAI

# Initialize the OpenAI client with your secret key
client = OpenAI(api_key="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")

# Define a function to check if the generated code for the sieve of Eratosthenes is correct
def is_sieve_correct(code, test_range=30):
    # Known primes less than test_range for validation
    known_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    
    # Define a dictionary to act as a namespace for exec
    namespace = {}
    
    # Try to execute the code and catch any errors
    try:
        exec(code, namespace)
    except Exception as e:
        print(f"Error in executing the generated code: {e}")
        return False
    
    # Retrieve the sieve function from the namespace, if it exists
    sieve_function = namespace.get('sieve_of_eratosthenes', None)
    
    if not callable(sieve_function):
        print("No callable 'sieve_of_eratosthenes' function found in the namespace.")
        return False
    
    # Try to get the list of primes using the sieve function
    try:
        primes = sieve_function(test_range)
    except Exception as e:
        print(f"Error in calling the sieve_of_eratosthenes function: {e}")
        return False
    
    # Check if the generated list of primes is correct
    return primes == known_primes

# Now you can test the generated code using this adjusted function.
# Use the generated code you have provided as the input to this function.


# Define the number of times to call the API
number_of_calls = 1000  # For example, let's call the API 5 times
prompt_message = f"""
    Write a Python function that generates a list of prime numbers less than 30.
    Write it in a way that is easy to understand and maintain.
    the function should be named sieve_of_eratosthenes and should take one argument named limit.
    Write it in an original way, do not copy and paste from the internet.
    Your response should be only the code without starting by ```python and ending by ```
    """
# Loop to simulate calling the API the specified number of times
for i in range(number_of_calls):
    
    report_response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        seed=30354229,
        messages=[
            {"role": "user", "content": prompt_message},
        ],
        temperature=0.1
    )

        # Extract the medical report content
    generated_code = report_response.choices[0].message.content
    
    # Test the generated code
    if is_sieve_correct(generated_code):
        print(f"Generated code is correct. Code from iteration {i+1}")
    else:
        print(f"Generated code from iteration {i+1} is not correct:\n{generated_code}")
