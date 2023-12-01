"""
Building on the example from n_shot_learning.py, we add a more complex example that uses multiple functions
to solve a problem.
"""
import inspect
import os
import openai
import logging

import http.client as http_client
import function_calling


def get_sources(module):
    functions_source = []
    for func in inspect.getmembers(module, inspect.isfunction):
        source = inspect.getsource(func[1])
        functions_source.append(source)

    return functions_source


functions = get_sources(function_calling)

if os.getenv("DEBUG"):
    http_client.HTTPConnection.debuglevel = 1
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)
    requests_log = logging.getLogger("requests.packages.urllib3")
    requests_log.setLevel(logging.DEBUG)
    requests_log.propagate = True

openai.debug = True
openai.api_key = os.getenv("OPENAI_API_KEY")

functions_as_string = ""
for func in functions:
    functions_as_string += func + "\n"

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "system",
               "content": "You are a source code generator. You will take python source code of functions and use only these functions to create a new function that fulfills the task given by the user"},
              {"role": "user", "content": """
              Given the following python functions:

              def get_greeting(name: str) -> str:
                return f"Hello {name}"

              def get_customer_name() -> str:
                return "John"

             Task: Create a function that returns the greeting for a customer
              """},
              {"role": "system", "content": """
              def get_greeting_for_customer() -> str:
                return get_greeting(get_customer_name())
              """
               },
              {"role": "user", "content": """
             Given the following python functions:
            
             def get_customer_name() -> str:
               return "John"
               
             def get_phone_number_for_customer(name: str) -> str:
                return "555-1234"
            
            Task: Create a function that returns the phone number of a customer
             """},
            {"role": "system", "content": """
             def get_phone_number_of_customer() -> str:
               return get_phone_number_for_customer(get_customer_name())
             """
               },
              {"role": "user", "content": f"""
              Given the following python functions:
              
                {functions_as_string}
                
              Task: Create a function named get_balance_for_customer that returns the balance of an account
              The parameter of the function is the customers name.
                """}
              ],
)

function_code = response["choices"][0]["message"]["content"]

print(function_code)

# Add the function to the module function_calling
exec(function_code, function_calling.__dict__)

# Get the function from using a string as the name
function = getattr(function_calling, "get_balance_for_customer")

# Call the function and print the result
balance = function("John Doe")
assert balance == 100

balance = function("Jack the ripper")
assert balance == 0
