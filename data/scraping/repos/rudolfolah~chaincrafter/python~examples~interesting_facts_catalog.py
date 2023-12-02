import os

from chaincrafter import Catalog, Chain, Prompt, response_format_list, response_style, response_length, extract_items_from_list
from chaincrafter.models import OpenAiChat

gpt4_chat = OpenAiChat(temperature=0.9, model_name="gpt-4", top_p=0.3, max_tokens=500)
catalog = Catalog()
path = os.path.dirname(__file__)
catalog.load(os.path.join(path, "catalog.yml"))

example_prompt = catalog.get_prompt(
    "facts_about_topic",
    [
        # Commonly used instructions or prompts can be added as modifiers and re-used
        response_style("professional economist"),
        response_length("long", "answer"),
        response_format_list("Interesting fact"),
    ],
)
followup_question_prompt = catalog.get_prompt(
    "followup_question",
    [
        response_style("a pirate who has just returned from the Galapagos Islands"),
        response_length("short", "4-5 sentences"),
    ],
    # Parses and extracts data from the previous response to populate the input variable that is used by the prompt
    facts_list=lambda facts_list: extract_items_from_list(facts_list)[0],
)
chain = Chain(
    catalog["helpful_assistant"],
    # The prompt message to send and the output key to store the response in
    (example_prompt, "facts_list"),
    (followup_question_prompt, "output"),
)
messages = chain.run(gpt4_chat, {"topic": input("Topic? ")})
for message in messages:
    print(f"{message['role']}: {message['content']}")
    print()
