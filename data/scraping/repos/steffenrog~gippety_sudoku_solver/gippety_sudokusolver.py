import json
import openai
import inspect
from typing import get_type_hints

# Keep your API key secure and confidential
openai.api_key = 'sk-Your_key'

def to_json(func) -> dict:  
    json_representation = dict()
    json_representation['name'] = func.__name__ 
    json_representation['description'] = func.__doc__.strip()
    
    parameters = inspect.signature(func).parameters
    func_type_hints = get_type_hints(func)
    
    json_parameters = dict()
    json_parameters['type'] = 'object'
    json_parameters['properties'] = {}
    
    for name, param in parameters.items():
        if name == 'return':
            continue
        
        param_info = {}
        param_info['description'] = str(param.default) if param.default != param.empty else 'No description'
        param_annotation = func_type_hints.get(name)
        
        if param_annotation:
            param_info['type'] = 'array' if param_annotation is list else 'string' if param_annotation is str else param_annotation.__name__
                    
        if name == 'self':
            continue
        
        json_parameters['properties'][name] = param_info
    
    json_representation['parameters'] = json_parameters
    
    return json_representation


def format_call(function_call: dict) -> str:
    json_data: dict = json.loads(function_call.__str__())
    
    function_name: str = json_data["name"]
    args_json: str = json_data["arguments"]
    
    args_dict: dict = json.loads(args_json)
    args: str = ', '.join([f"{k}='{v}'" for k, v in args_dict.items()])

    return f"{function_name}({args})"


prompt = 'Solve the following Sudoku puzzle: [[2, 5, 0, 0, 3, 0, 9, 0, 1], [0, 1, 0, 0, 0, 4, 0, 0, 0], [4, 0, 7, 0, 0, 0, 2, 0, 8], [0, 0, 5, 2, 0, 0, 0, 0, 0], [0, 0, 0, 0, 9, 8, 1, 0, 0], [0, 4, 0, 0, 0, 3, 0, 0, 0], [0, 0, 0, 3, 6, 0, 0, 7, 2], [0, 7, 0, 0, 0, 0, 0, 0, 3], [9, 0, 3, 0, 0, 0, 6, 0, 4]]'
conversation = [
    {
        'role': 'assistant',
        'content': f'You are ChatGPT, an Artificial Intelligence developed by OpenAI, respond in a concise way'
    },
    {
        'role': 'user',
        'content': prompt
    }
]



def solve_sudoku(grid: str) -> str:
    """Solve the provided Sudoku puzzle"""

    #print(f"Received input: {grid}") 
    
    grid = json.loads(grid)
    
    def is_valid(grid, row, col, num):
        for x in range(9):
            if grid[row][x] == num or grid[x][col] == num:
                return False
        start_row, start_col = 3 * (row // 3), 3 * (col // 3)
        for i in range(3):
            for j in range(3):
                if grid[i + start_row][j + start_col] == num:
                    return False
        return True

    def solver(grid):
        for i in range(9):
            for j in range(9):
                if grid[i][j] == 0:
                    for num in range(1, 10):
                        if is_valid(grid, i, j, num):
                            grid[i][j] = num
                            if solver(grid):
                                return True
                            grid[i][j] = 0
                    return False
        return True
    
    solved_grid = grid if solver(grid) else []
    #print(f"Returning output: {solved_grid}")  
    return json.dumps(solved_grid)


sudoku_solver_function_json = to_json(solve_sudoku)
#print (sudoku_solver_function_json)



completion = openai.ChatCompletion.create(
    model='gpt-3.5-turbo-0613', 
    messages=conversation, 
    functions=[sudoku_solver_function_json]
)


# Use the Chat API to interact with the model
completion = openai.ChatCompletion.create(model='gpt-3.5-turbo-0613', 
                             messages=conversation, functions=[sudoku_solver_function_json])

# Add the function call and its result to the conversation
conversation += [json.loads(completion.choices[0].message.__str__())]
conversation += [{
    'role': 'function',
    'name': completion.choices[0].message.function_call.name,
    'content': eval(format_call(completion.choices[0].message.function_call)),
}]

# Continue the conversation with the updated function call
completion = openai.ChatCompletion.create(model='gpt-3.5-turbo-0613', 
                             messages=conversation, functions=[sudoku_solver_function_json])

# Print the generated response
print(completion.choices[0].message.content)


"""
The Sudoku puzzle has been solved. Here is the solution:

[[2, 5, 8, 7, 3, 6, 9, 4, 1], 
 [6, 1, 9, 8, 2, 4, 3, 5, 7], 
 [4, 3, 7, 9, 1, 5, 2, 6, 8], 
 [3, 9, 5, 2, 7, 1, 4, 8, 6], 
 [7, 6, 2, 4, 9, 8, 1, 3, 5], 
 [8, 4, 1, 6, 5, 3, 7, 2, 9], 
 [1, 8, 4, 3, 6, 9, 5, 7, 2], 
 [5, 7, 6, 1, 4, 2, 8, 9, 3], 
 [9, 2, 3, 5, 8, 7, 6, 1, 4]]

"""

""" 
The solved Sudoku puzzle is as follows:

```
2 5 8 | 7 3 6 | 9 4 1
6 1 9 | 8 2 4 | 3 5 7
4 3 7 | 9 1 5 | 2 6 8
---------------------
3 9 5 | 2 7 1 | 4 8 6
7 6 2 | 4 9 8 | 1 3 5
8 4 1 | 6 5 3 | 7 2 9
---------------------
1 8 4 | 3 6 9 | 5 7 2
5 7 6 | 1 4 2 | 8 9 3
9 2 3 | 5 8 7 | 6 1 4
```

I hope this solution is helpful! Let me know if there's anything else I can assist you with.

 """
#print(f"API Response: {completion}")
#print(f"Function Call: {completion.choices[0].message.function_call}")
#print(solve_sudoku("{\n  \"grid\": \"[[2, 5, 0, 0, 3, 0, 9, 0, 1], [0, 1, 0, 0, 0, 4, 0, 0, 0], [4, 0, 7, 0, 0, 0, 2, 0, 8], [0, 0, 5, 2, 0, 0, 0, 0, 0], [0, 0, 0, 0, 9, 8, 1, 0, 0], [0, 4, 0, 0, 0, 3, 0, 0, 0], [0, 0, 0, 3, 6, 0, 0, 7, 2], [0, 7, 0, 0, 0, 0, 0, 0, 3], [9, 0, 3, 0, 0, 0, 6, 0, 4]]\"\n}"))

"""
# Parse the API response to get the function call details
function_call = completion.choices[0].message.function_call
function_name = function_call['name']
arguments = json.loads(function_call['arguments'])

# Call the function with the provided arguments and get the response
if function_name == 'solve_sudoku':
    response = solve_sudoku(arguments)
else:
    response = "Unknown function"

# Send the response back to the user
print(response)
"""