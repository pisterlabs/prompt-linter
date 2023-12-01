# Author: Rajib
# Description: This is an example of how to create a router chain using semantic kernel
# It uses the Router Chain class in router_chain.py
import semantic_kernel as sk

from semantic_kernel.connectors.ai.open_ai.services.open_ai_chat_completion import OpenAIChatCompletion

from chains.router_chain import RouterChain

if __name__ == "__main__":
    templates = {}
    # One of the prompt templates to be fed to the router chain
    tourist_guide_template = "You are an expert guide." \
                             "You have knowledge of all the historic places in the world." \
                             "Please provide answer to the provided question below." \
                             "question:{{$question}}"

    # The second prompt template to be fed to the router chain
    teacher_template = "You are a teacher of history. You teach students the history of the world. " \
                       "Please provide answer to the provided question below." \
                       "question:{{$question}}"

    # templates["tourist_guide_template"] = tourist_guide_template
    # templates["teacher_template"] = teacher_template

    # Creating a list of the prompt templates to send to the router chain
    # Prompt name and description are very important. Needs to clearly mention what the prompt should do
    prompt_templates = [
        {"prompt_name": "tourist_guide_template",
         "prompt_desc": "Good for answering questions about historic placess in the world",
         "prompt_template": tourist_guide_template,
         },
        {"prompt_name": "teacher_template",
         "prompt_desc": "Good for answering student questions on the history of the world",
         "prompt_template": teacher_template
         }

    ]

    # Initializing the kernel
    kernel = sk.Kernel()
    api_key, org_id = sk.openai_settings_from_dot_env()
    kernel.add_chat_service("gpt-4", OpenAIChatCompletion("gpt-4", api_key))

    # The question to be asked to the router chain. I used this for testing
    # input = "Where is TajMahal?"
    input = "When did India became independent?"

    rtc = RouterChain()
    # After the chain runs, it sends back the appropiate goal(prompt template)
    # along with the question that needs to be part of the goal.
    goal, input = rtc.run(prompt_templates,
                          input,
                          kernel)

    # Initializing the context
    sk_context = kernel.create_new_context()
    sk_context["question"] = input

    # Destination chain
    qa_chat_bot = kernel.create_semantic_function(
        prompt_template=goal,
        description="Provides answer to an input question",
        max_tokens=1000
    )
    # Getting the final answer
    answer = qa_chat_bot.invoke(context=sk_context)

    print(answer)
