import openai

def generate_function(object_repr, action):
    openai.api_key = "your_API_key"
    prompt = f"Create a Python function named 'gen_fun' that takes a Python object represented as {object_repr} and performs the action: {action}. The function should return the result."
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=200,
        n=1,
        stop=None,
        temperature=0.5,
    )
    function_code = response.choices[0].text.strip()
    return function_code

# Example usage:
object_repr = "a list of integers"
action = "calculate the sum of all the elements in the list"
generated_function_code = generate_function(object_repr, action)
print(generated_function_code)
print()
try:
    exec(generated_function_code)
except:
    gfc = generate_function(object_repr, action)
    print(gfc)
    exec(gfc)

result = gen_fun([1, 2, 3, 4, 5])
print("Result:", result)
print()
object_repr = "The location of an jpeg"
action = "add from PIL import Image. using PIL resize the image to (512,512) and save as {image name}_resize.jpg, return True on sucess."
generated_function_code = generate_function(object_repr, action)
print(generated_function_code)
print()
try:
    exec(generated_function_code)
except:
    gfc = generate_function(object_repr, action)
    print(gfc)
    exec(gfc)
result = gen_fun('garden.jpg')  # Call the defined function by its name 'gen_fun'
print("Result:", result)



