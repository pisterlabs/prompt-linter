import os
import openai
import tiktoken

# Set the API key for OpenAI API
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Set the encoding for GPT-4 with tiktoken
tiktoken.enc = tiktoken.encoding_for_model("gpt-4")

def token_count(string: str) -> int:
    """
    Calculate the number of tokens in a given string.
    
    Parameters:
    string (str): The text for which to calculate the token count.
    
    Returns:
    int: The total number of tokens.
    """
    num_tokens = len(tiktoken.enc.encode(string))
    return num_tokens

def completion(messages, max_token_count, functions=None, temperature=0.7):
    """
    Perform a chat-based completion using OpenAI's API.
    
    Parameters:
    messages (list): A list of message objects for the conversation.
    max_token_count (int): Maximum number of tokens for the completion.
    functions (list): Optional list of function definitions.
    temperature (float): Controls randomness of output. Default is 0.7.
    
    Returns:
    str: The generated text from the model.
    """
    # Check if functions are specified
    if functions is None:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=messages,
            temperature=temperature
        )
        # Write the model's response to a file
        return response['choices'][0]['message']['content']

    else:
        function_call = {}
        if len(functions) == 1:
            function_call = functions[0]
            response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=messages,
            temperature=temperature,
            functions=functions,
            stream=True,
            function_call=function_call)
        else:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-16k",
                messages=messages,
                temperature=temperature,
                functions=functions,
                stream=True
            )
        
        full_resp = ''
        try:
            for resp in response:
                delta = resp['choices'][0].get('delta')
                if delta is not None:
                    function_call = delta.get('function_call')
                    content = delta.get('content')

                    if function_call is not None:
                        arguments = function_call.get('arguments', '')
                        full_resp += arguments  
                        print(full_resp)
                    elif content is not None:
                        print(full_resp)
                        full_resp += content

            return full_resp

        except Exception as e:
            print(f"An error occurred in Completion: {e}")
            return ""

if __name__ == "__main__":
    # Example usage of the completion function
    example_msg = [{"role": "user", "content": "Tell me a joke."}]
    completion_output = completion(messages=example_msg, max_token_count=150)
    print(completion_output)
