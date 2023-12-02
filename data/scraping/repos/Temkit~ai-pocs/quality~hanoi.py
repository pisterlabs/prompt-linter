from openai import OpenAI

# Initialize the OpenAI client with your secret key
client = OpenAI(api_key="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")


def is_towers_of_hanoi_correct(code, number_of_disks=3):
    # Output list to capture the print statements
    output = []

    # Custom print function to capture the print output within the exec environment
    def custom_print(*args, **kwargs):
        output.append(' '.join(map(str, args)))

    # Setup the towers for testing
    A = list(range(number_of_disks, 0, -1))  # Source tower with 'number_of_disks' disks
    B = []  # Target tower
    C = []  # Auxiliary tower
    
    # Define a dictionary to act as a namespace for exec
    namespace = {
        'print': custom_print,
        'A': A,
        'B': B,
        'C': C
    }

    # Execute the code and test the generated function
    try:
        exec(code, namespace)
    except Exception as e:
        print(f"Error in executing the generated code: {e}")
        return False

    # Retrieve the hanoi function from the namespace, if it exists
    hanoi_function = namespace.get('hanoi', None)

    if not callable(hanoi_function):
        print("No callable 'hanoi' function found in the namespace.")
        return False

    # Execute the towers of hanoi function and check the result
    try:
        hanoi_function(number_of_disks, 'A', 'C', 'B')  # Use the string names to match the simulated code's expectations
    except Exception as e:
        print(f"Error in calling the hanoi function: {e}")
        return False

    # Check if the output has the expected number of moves
    expected_moves = 2 ** number_of_disks - 1
    actual_moves = len(output)
    # Check if the target tower 'B' has all the disks in the correct order
    correct_solution = namespace['B'] == list(range(number_of_disks, 0, -1))

    return actual_moves == expected_moves and correct_solution

# Define the number of times to call the API
number_of_calls = 1  # Let's call the API once

# Loop to simulate calling the API the specified number of times
for i in range(number_of_calls):
    # Placeholder for an API call to generate the code for Towers of Hanoi
    # Replace this with an actual API call as per your OpenAI client setup
    report_response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        seed=30354229,
        messages=[
            {"role": "user", "content": """Write a Python function named `hanoi` that solves the Towers of Hanoi puzzle. The function should take the number of disks and the names of the rods as inputs and print each move as a separate line in the format "Move disk {n} from rod {source} to rod {target}". The function should recursively move the disks from the source rod to the target rod using the auxiliary rod following the rules of the Towers of Hanoi puzzle.
. your response should be only the code without starting by ```python and ending by ``` """},
        ],
        temperature=0.1
    )

    # Extract the generated code
    generated_code = report_response.choices[0].message.content
    
    # Test the generated code
    if is_towers_of_hanoi_correct(generated_code):
        print(f"Generated code is correct. Code from iteration {i+1}")
    else:
        print(f"Generated code from iteration {i+1} is not correct:\n{generated_code}")
