import openai
import yaml
import os
import sys

# Load the YAML file
with open("dialog.yaml", "r") as f:
    dialog = yaml.safe_load(f)

# Load the YAML prompt_decision template file
with open("prompt_decision_template.yaml", "r") as f:
    prompt_decision_template = f.read()

agent = "Sortium"

# Initialize the dialog
current_id = "start"

while True:
    # Find the current dialog object
    current_dialog = next(
        (obj for obj in dialog if obj["id"] == current_id), None)
    if not current_dialog:
        print("Oops, something went wrong. Please try again.")
        break

    # Print the current text and options
    print(f"{agent}: {current_dialog['text']}")
    for option in current_dialog["options"]:
        print(f"- {option['option']}")

    # map options to options.option
    options = list(map(lambda x: x["option"], current_dialog["options"]))

    # Prompt the user for input
    user_input = input("User: ")

    # Replace the placeholders in the prompt_decision template with actual text
    decision_prompt = current_dialog["text"]
    option_list = yaml.dump(options, sort_keys=False)
    user_response = user_input

    # Create the prompt_decision
    prompt_decision = prompt_decision_template.format(
        decision_prompt=decision_prompt,
        option_list=option_list,
        user_response=user_response
    )

    # print("=========== OpenAI prompt ===========")
    # print(prompt_decision)
    # print("=====================================")

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt_decision,
        suffix="",
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    # print("OpenAI response:", response)

    choice = response.choices[0].text.strip()

    # print("OpenAI choice:", choice)

    # Try to match the user's response with one of the options
    for i, option in enumerate(current_dialog["options"]):
        # print("option:", option["option"])
        if option["option"] == choice:
            choice = i
            break
    else:
        # If no match is found, repeat the prompt
        print(f"{agent}: I'm sorry, I didn't understand your response.")
        continue

    # Get the next dialog ID based on the user's choice
    next_id = current_dialog["options"][choice]["next_id"]

    if next_id == "exit":
        # If the user chooses to exit, end the dialog
        print(f"{agent}: Thank you for using the dialog system.")
        break
    else:
        # Otherwise, continue to the next dialog
        current_id = next_id
