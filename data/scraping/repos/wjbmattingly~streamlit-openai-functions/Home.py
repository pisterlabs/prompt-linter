import streamlit as st
import openai
import json

open_ai_key = st.text_input("OpenAI API Key")
openai.api_key = open_ai_key
st.title("OpenAI Function Prototype Creation App")
display_template = st.expander("Show OpenAI Example")
display_template.markdown(

    """
    ```python
    messages = [{"role": "user", "content": "What's the weather like in Boston?"}]
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
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=messages,
        functions=functions,
        function_call="auto",  # auto is default, but we'll be explicit
    )
    response_message = response["choices"][0]["message"]

    ```
    """
)


st.write("---")

name = st.text_input("Function Name")
description = st.text_input("Function Description")
num_properties = st.number_input("Number of Properties", 1, 3)


role = st.selectbox("Role", ["user"])
content = st.text_area("Message")

st.write("---")
tabs = st.tabs([f"Property {i}" for i in range(num_properties)])

properties = {}
required = []

for i, tab in enumerate(tabs):
    property_name = tab.text_input(f"Property {i} Name", key=f"property_name_{i}")
    property_type = tab.selectbox(f"Property {i} Type", ["string", "boolean", "number", "date"], key=f"property_type_{i}")
    property_description = tab.text_area(f"Property {i} Description", key=f"property_description_{i}")
    property_required = tab.checkbox("Is it required?", key=f"property_required_{i}")
    if property_required:
        required.append(property_name)

    properties[property_name] = {"type": property_type, "description": property_description}

st.write("---")
messages = [{"role": role, "content": content}]

functions = [{

    "name": name,
    "description": description,
    "parameters": {

        "type": "object",
        "properties": properties,
        "required": required
    }
}]

result = f"""
```python
    completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0613",
    messages = {messages},
    functions = {functions}
)
"""
result_expander = st.expander("Result Code")
result_expander.markdown(result)

if st.button("Test"):
    completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0613",
    messages = messages,
    functions = functions
)
    st.header("Output")
    result  = json.loads(completion.choices[0].message["function_call"]["arguments"])
    st.write(result)