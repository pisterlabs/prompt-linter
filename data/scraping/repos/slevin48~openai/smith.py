import streamlit as st
import openai, json

st.set_page_config(page_title="Smith",page_icon="🤖")
openai.api_key = st.secrets['OPEN_AI_KEY']
model = "gpt-3.5-turbo"
# debug = False

# Initialization
convo = []
# if 'convo' not in st.session_state:
#     st.session_state.convo = []

# Example dummy function hard coded to return the same weather
# In production, this could be your backend API or an external API
def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    weather_info = {
        "location": location,
        "temperature": "72",
        "unit": unit,
        "forecast": ["sunny", "windy"],
    }
    return json.dumps(weather_info)

def chat(messages,function=False):
    # Generate a response from the ChatGPT model
    if function:
        response = openai.chat.completions.create(
            model=model,
            messages=messages,
            functions=functions,
            function_call="auto",  # auto is default, but we'll be explicit
        )
    else:
        response = openai.chat.completions.create(
        model=model,
        messages=messages,
        )
    response_choices = response.choices
    response_message = response_choices[-1].message
    return response_message


st.title('🤖 Agent Smith')


# Create a text input widget in the Streamlit app
prompt = st.text_input('Ask a question to Smith','What is the weather in Boston')
debug = st.checkbox('Debug')

if st.button("Submit", type="primary"):
    # Step 1: send the conversation and available functions to GPT
    messages = [{"role": "user", "content": prompt}]
    functions = [
        {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        }
    ]

    # Append the text input to the conversation
    convo.append({'role': 'user', 'content': prompt })
    # st.session_state.convo.append({'role': 'user', 'content': prompt })

    # Query the chatbot with the complete conversation
    # response = dumb_chat()
    # response = chat(st.session_state.convo)
    response_message = chat(convo,function=True)

    if debug:
        st.write('🤖 Agent select function to call',response_message)

    # Step 2: check if GPT wanted to call a function
    if response_message.function_call:
        # Step 3: call the function
        # Note: the JSON response may not always be valid; be sure to handle errors
        available_functions = {
            "get_current_weather": get_current_weather,
        }  # only one function in this example, but you can have multiple
        function_name = response_message.function_call.name
        function_to_call = available_functions[function_name]
        function_args = json.loads(response_message.function_call.arguments)
        function_response = function_to_call(
            location=function_args.get("location"),
            unit=function_args.get("unit"),
        )
        # st.write(function_response)
        # Step 4: send the info on the function call and function response to GPT
        # convo.append(response_message)  # extend conversation with assistant's reply
        function_message = {
                "role": "function",
                "name": function_name,
                "content": function_response,
            }
        convo.append(function_message)  # extend conversation with function response
        
        if debug:
            st.write('🧪 Function result',function_message)

        second_response = chat(convo)  # get a new response from GPT where it can see the function response
        
        if debug:
            st.write('🤖 Agent response integrating Function result',second_response)

        st.write('🤖',second_response.content)


        
    # Add response to the conversation
    # st.session_state.convo.append({'role': 'assistant', 'content': response })
    # convo.append({'role': 'assistant', 'content': response_message })

    # st.write('🐱',convo)

    # for line in st.session_state.convo:
    #     if line['role'] == 'user':
    #         st.write('🐱',line['content'])
    #     else:
    #         st.write('🤖',line['content'])