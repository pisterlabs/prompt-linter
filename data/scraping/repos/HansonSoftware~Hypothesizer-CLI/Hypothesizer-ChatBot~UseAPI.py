import openai
import json
from decouple import config

API_KEY = config("OPENAI_KEY")
openai.api_key = API_KEY

# Terminal Color Defenitions:
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
NORMAL = "\033[0m"


# Chat: v1/chat/completions
# Valid Models: gpt-3.5-turbo, gpt-3.5-turbo-16k
def chatCompletion(model, temp, messages):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temp,
        # Adjust max_tokens to increase the max response length.
        # 1 token ~ 3/4th of a word
        max_tokens=5000,
    )
    return response


def printChatHistory(messages):
    print("--------------------------------")
    for message in messages:
        if message["role"] == "system" or message["role"] == "assistant":
            print(RED + "%s: " % message["role"].title() + NORMAL)
        if message["role"] == "user":
            print(BLUE + "%s: " % message["role"].title() + NORMAL)
        print("Message: %s" % message["content"])

        print("--------------------------------")

    return


def addMessage(role, content, messages):
    message = {"role": role, "content": content}
    messages.append(message)
    return


# This is where the magic happens, The conversation starts here.
def interactiveChat(model, system, prompt, temp):
    messages = []
    messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    while True:
        printChatHistory(messages)
        response = json.loads(str(chatCompletion(model, temp, messages)))
        content = response["choices"][0]["message"]["content"]
        print(content)
        addMessage("assistant", content, messages)
        userInput = input(
            "\nEnter [q, quit or exit] to quit.\nEnter another "
            + RED
            + "prompt "
            + NORMAL
            + ": "
        )
        # if q, quit, exit
        if userInput == "q" or input == "quit" or input == "exit":
            print("\nThanks for chatting!")
            print(RED + "Exiting..." + NORMAL)
            break
        else:
            addMessage("user", userInput, messages)
    return


# Command Line Interface Helper Functions:
def welcome():
    print(
        "Welcome to the "
        + BLUE
        + "Hypothesizer"
        + NORMAL
        + " OpenAI Command Line Interface!"
    )
    print(RED + "Be sure to place your prompt into Prompt.txt!" + NORMAL)
    print(RED + "Be sure to place your system instruction into System.txt!" + NORMAL)
    print(BLUE + "Model: " + NORMAL + "[turbo, turbo-16k, gpt-4] Enter one of these values.")
    print(YELLOW + "Temperature: " + NORMAL + "Enter a value between 0.0 and 2.0")


def getInputs():
    # Chat Model Validation:
    validModels = ["turbo", "turbo-16k", "gpt-4"]
    while True:
        model = input("Enter the " + BLUE + "Model " + NORMAL + "you want to use: ")
        if model.lower() in validModels:
            if model.lower() == "turbo":
                model = "gpt-3.5-turbo"
            elif model.lower() == "turbo-16k":
                model = "gpt-3.5-turbo-16k"
            elif model.lower() == "gpt-4":
                model = "gpt-4"
            break
        else:
            print('Invalid Chat Model. Please enter "turbo" OR "turbo-16k" OR "gpt-4". Try again.')
    # Temperature Validation:
    while True:
        try:
            temp = float(
                input("Enter a " + YELLOW + "Temperature " + NORMAL + "[0.0, 2.0]: ")
            )
            if 0.0 <= temp <= 2.0:
                break
            else:
                print(
                    "Invalid temperature. Please enter a float value between [0.0 and 2.0]. Try again."
                )
        except ValueError:
            print(
                "Invalid temperature. Please enter a valid float value between [0.0 and 2.0]. Try again."
            )

    return model, temp


def printChoices(model, temp):
    print("\nYou Chose:")
    print(BLUE + "Model:" + NORMAL, model)
    print(YELLOW + "Temperature:" + NORMAL, temp)


def main():
    welcome()
    choice = input("\nProceed? (Y/n): ")
    # Proceed:
    if choice.lower() in ["y", "yes"]:
        model, temp = getInputs()
        file = open("Prompt.txt", "r")
        prompt = file.read()
        file.close()
        printChoices(model, temp)
        print("\nPrompt:\n%s" % prompt)
        file = open("System.txt", "r")
        instruction = file.read()
        file.close()
        print("System Instruction:\n%s" % instruction)

        choice = input("\nProceed? (Y/n): ")

        # Proceed:
        if choice.lower() in ["y", "yes"]:
            print(GREEN + "Proceeding..." + NORMAL)
            print("\nStarting conversation with Chat GPT:")
            # Start the chat
            interactiveChat(model, instruction, prompt, temp)
        # Exit:
        else:
            print(RED + "Exiting..." + NORMAL)
    # Exit:
    else:
        print(RED + "Exiting..." + NORMAL)


if __name__ == "__main__":
    main()
