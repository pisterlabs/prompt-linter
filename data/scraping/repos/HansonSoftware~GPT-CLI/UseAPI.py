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
        model = model,
        messages = messages,
        temperature = temp,
        # Adjust max_tokens to increase the max response length.
        # 1 token ~ 3/4th of a word
        max_tokens = 100
    )
    return response;

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
    flag = 0
    while True:
        printChatHistory(messages)
        response = json.loads(str(chatCompletion(model, temp, messages)))
        content = response['choices'][0]['message']['content']
        print(content)
        addMessage("assistant", content, messages)
        userInput = input("\nEnter [q, quit or exit] to quit.\nEnter another "+ RED + "prompt " + NORMAL + ": ")
        # if q, quit, exit
        if userInput == "q" or input == "quit" or input == "exit":
            print("\nThanks for chatting!")
            print(RED + "Exiting..." + NORMAL)
            break
        else:
            addMessage("user", userInput, messages)
    return

# Complete: /v1/completions
# Valid Models: davinci, curie, babbage, ada
def textCompletion(model, prompt, temp):
    response = openai.Completion.create(
        model = model,
        prompt = prompt,
        temperature = temp,
        # Adjust max_tokens to increase the max response length.
        # 1 token ~ 3/4th of a word
        max_tokens = 10
    )
    response = json.loads(str(response))
    content = response['choices'][0]['text']
    return content

# Edit: /v1/edits
# Valid Models: davinci-edit, davinci-code
def textEdit(model, prompt, inputValue, temp):
    response = openai.Edit.create(
        model = model,
        instruction = prompt,
        temperature = temp,
        input = inputValue,
        top_p = 1
    )
    response = json.loads(str(response))
    content = response['choices'][0]['text']
    print(content)
    return content


# Command Line Interface Helper Functions:
def welcome():
    print("Welcome to the " + RED + "DevX" + NORMAL + " OpenAI Command Line Interface!")
    print("You will be prompted to enter some values to tweak the API call made to OpenAI")
    print(RED + "Be sure to place your prompt into Prompt.txt!" + NORMAL)
    print(RED + "If you plan to edit, be sure to place your text to modify into Input.txt!" + NORMAL)
    print(RED + "If you plan to chat, be sure to place your system instruction into Input.txt!" + NORMAL)
    print(GREEN + "Use Case: " + NORMAL + "[edit, complete, chat] Enter one of these values.")
    print("    This controls which function called.\n    edit: ChatGPT will edit your prompt\n    complete: ChatGPT will respond to your prompt\n    chat: Chat with the gpt-3.5 models!\n")
    print(BLUE + "Model: " + NORMAL + "complete: [davinci, curie, babbage, ada], chat: [turbo, turbo-16k] Enter one of these values.")
    print('    This controls which model to use, davinci is the "smartest".\n    If your use case is "edit", the model will be preset.')
    print(YELLOW + "Temperature: " + NORMAL + "Enter a value between 0.0 and 1.0")
    print('    Temperature controls the "creativity" of the response.\n    1.0 is most "creative"\n')

def getInputs():
    # Use Case Validation:
    while True:
        useCase = input("Enter your desired " + GREEN + "Use Case: " + NORMAL)
        if useCase.lower() == "edit" or useCase.lower() == "complete" or useCase.lower() == "chat":
            break
        else:
            print("Invalid use case. Please enter either 'edit' or 'complete'. Try again.")    
    # Model Validation:
    if useCase.lower() == "chat":
        # Chat Model Validation:
        validModels = ["turbo", "turbo-16k"]
        while True:
            model = input("Enter the "+ BLUE + "Model " + NORMAL + "you want to use: ")
            if model.lower() in validModels:
                if model.lower() == "turbo":
                    model = "gpt-3.5-turbo"
                elif model.lower() == "turbo-16k":
                    model = "gpt-3.5-turbo-16k"
                break
            else:
                print('Invalid Chat Model. Please enter "turbo" OR "turbo-16k". Try again.')
    if useCase.lower() == "complete":  
        # Complete Model Validation:
        validModels = ["davinci", "curie", "babbage", "ada"]
        while True:
            model = input("Enter the "+ BLUE + "Model " + NORMAL + "you want to use: ")
            if model.lower() in validModels:
                if(model == "davinci"):
                    model = "text-davinci-003"
                elif(model == "curie"):
                    model = "text-curie-001"
                elif(model == "babbage"):
                    model = "text-babbage-001"
                elif(model == "ada"):
                    model = "text-ada-001"
                break
            else:
                print("Invalid model. Please enter either 'davinci', 'curie', 'babbage', or 'ada'. Try again.")
    # Edit Model needs to be set to "text-davinci-edit-001"
    if useCase.lower() == "edit": 
        model = "text-davinci-edit-001"
    # Temperature Validation:
    while True:
        try:
            temp = float(input("Enter a " + YELLOW + "Temperature " + NORMAL + "[0.0, 1.0]: "))
            if (0.0 <= temp <= 1.0):
                break
            else:
                print("Invalid temperature. Please enter a float value between [0.0 and 1.0]. Try again.")
        except ValueError:
            print("Invalid temperature. Please enter a valid float value between [0.0 and 1.0]. Try again.")

    return useCase, model, temp

def printChoices(useCase, model, temp):
    print("\nYou Chose:")
    print(GREEN + "Use Case:" + NORMAL, useCase)
    print(BLUE + "Model:" + NORMAL, model)
    print(YELLOW + "Temperature:" + NORMAL, temp)

def main():
    welcome()
    useCase, model, temp = getInputs()
    file = open("Prompt.txt", "r")
    prompt = file.read()
    file.close()
    printChoices(useCase, model, temp)
    print("\nPrompt:\n%s" % prompt)

    if(useCase.lower() == "edit" or useCase.lower() == "chat"):
        file = open("Input.txt" , "r")
        instruction = file.read()
        file.close()
        if useCase.lower() == "edit":
            print("Text to edit:\n%s" % instruction)
        if useCase.lower() == "chat":
            print("System Instruction:\n%s" % instruction)
    
    choice = input("\nProceed? (Y/n): ") 
    # User Proceeds:
    if choice.lower() in ["y", "yes"]:
        print(GREEN + "Proceeding..." + NORMAL)
        print("\nChat GPT Output:") 
        if (useCase.lower() == "chat"):
            interactiveChat(model, instruction, prompt, temp)
        elif(useCase.lower() == "complete"):
            textCompletion(model, prompt, temp)
        elif(useCase.lower() == "edit"):
            textEdit(model, prompt, instruction, temp)
    # User Exits:       
    else:
        print(RED + "Exiting..." + NORMAL)

if __name__ == "__main__":
    main()
