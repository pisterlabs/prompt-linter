# Copyright (c) Microsoft. All rights reserved.

import asyncio

import semantic_kernel as sk
import semantic_kernel.connectors.ai.open_ai as sk_oai
from semantic_kernel.connectors.ai.open_ai.services.open_ai_chat_completion import OpenAIChatCompletion

kernel = sk.Kernel()

# deployment_name, api_key, endpoint = sk.azure_openai_settings_from_dot_env()
api_key, org_id = sk.openai_settings_from_dot_env()
kernel.add_chat_service("chat-gpt", OpenAIChatCompletion("gpt-4", api_key))

# enabling or disabling function calling is done by setting the function_call parameter for the completion.
# when the function_call parameter is set to "auto" the model will decide which function to use, if any.
# if you only want to use a specific function, set the name of that function in this parameter,
# the format for that is 'SkillName-FunctionName', (i.e. 'math-Add').
# if the model or api version do not support this you will get an error.
prompt_config = sk.PromptTemplateConfig.from_completion_parameters(
    max_tokens=2000,
    temperature=0.7,
    top_p=0.8,
    function_call="auto",
    chat_system_prompt="You are a AI assistant.",
)
prompt_template = sk.ChatPromptTemplate(
    "{{$user_input}}", kernel.prompt_template_engine, prompt_config
)

function_config = sk.SemanticFunctionConfig(prompt_config, prompt_template)
chat_function = kernel.register_semantic_function("ChatBot", "Chat", function_config)
# define the functions available
functions = [
    {
        "name": "search_hotels",
        "description": "Retrieves hotels from the search index based on the parameters provided",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The location of the hotel (i.e. Seattle, WA)",
                },
                "max_price": {
                    "type": "number",
                    "description": "The maximum price for the hotel",
                },
                "features": {
                    "type": "string",
                    "description": "A comma separated list of features (i.e. beachfront, free wifi, etc.)",
                },
            },
            "required": ["location"],
        },
    }
]


async def main() -> None:
    context = kernel.create_new_context()
    context.variables[
        "user_input"
    ] = "I want to find a hotel in Seattle with free wifi and a pool and my max price is $200"

    context = await chat_function.invoke_async(context=context, functions=functions)
    if function_call := context.objects.pop('function_call', None):
        print(f"Function to be called: {function_call.name}")
        print(f"Function parameters: \n{function_call.arguments}")
        return
    print("No function was called")
    print(f"Output was: {str(context)}")


if __name__ == "__main__":
    asyncio.run(main())