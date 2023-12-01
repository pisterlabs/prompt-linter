import openai
import csv
import os

openai.api_key = os.environ.get("OPENAI_API_KEY")
chat_history_dir = os.environ.get("CHAT_HISTORY_DIR") if os.environ.get("CHAT_HISTORY_DIR") else "./chat_history"
kwargs={
    "model":"gpt-3.5-turbo-16k",
    "temperature":1.0,
}


def get_response(chat):
    try:
        response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=chat, temperature=1.0 # diy it
    )
    except:
        save_chat(chat)
    return response["choices"][0]["message"]["content"]


def save_chat(chat:list):
    chat.append(
        {
            "role": "user",
            "content": "summarize the entire conversation in under 4 words",
        }
    )
    with open(
        f"{chat_history_dir}/{get_response(chat[1:])}csv", mode="w",encoding='utf8',newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=["role", "content"])
        writer.writeheader()
        for row in chat[1:]:
            writer.writerow(row)


def reinput_line(history):
    target=int(input("input the line number you want to reinput:"))
    history[target]["role"]=input("input the target,user or content or system:")
    print("now history:",history)
    history[target]["content"]=input("input the content:")
    print("now history:",history)
    return history


def default_command(history):
    print("Invalid command")
    return inputProcess(input("\n:"),history)


def save_template(history):
    file_name = "./templates/" + input("Enter file name to load the chat history:")
    with open(file_name, "w",encoding='utf8') as f:
        for message in history:
            f.write(message["role"] + ": " + message["content"] + "\n")
    print("Chat history saved successfully\n")
    return history


def System():
    pass


def load_template(history):
    file_name = input("Enter file name to load the chat history: ")
    try:
        with open("./templates/"+file_name, "r",encoding="utf8") as f:
            lines = f.readlines()
            for line in lines:
                if line.strip() != "":
                    if line[0] == "u":
                        history.append({"role": "user", "content": line[2:].strip()})
                    elif line[0] == "a":
                        history.append({"role": "assistant", "content": line[2:].strip()})
                    else:
                        history.append({"role": "system", "content": line[2:].strip()})
            print("Chat history loaded successfully\n")
    except FileNotFoundError:
        print("File not found\n")
    return inputProcess(input("\n:"),history)


def inputProcess(user_input, history:list):
    global command_dict
    if user_input[0] == "/" or user_input[0] == ":" or user_input[0]=='\\':
        command = user_input[1:]
        if command in command_dict:
            result = command_dict[command](history)
            if result:
                return result
        else:
            result = command_dict["default"](history)  # 执行默认命令
            if result:
                return result
    else:
        history.append({"role": "user", "content": user_input})
        return history


def INPUT(history:list[dict]):
    print("input:")
    i=0
    while True:
        line = input()
        if line:
            if line!="END":
                i+=1
                inputProcess(line, history)
            else:
                content=""
                for j in history[-i:]:
                    content+=j["content"]+"\n"
                history=history[:-i]+[{"role": "user", "content": content}]
                print("LONG INPUT END")
                break
        else:
            continue
    return history

def HELP(history:list):
    print(command_dict)
    default_command(history)

def turing(history:list):
    print(history)
    while True:
        i=input("input command:")
        if i=="q":
            break
        exec(i)
        print(history)
    # history.append({"role": "user", "content": ""})
    return history
command_dict = {
    "input":INPUT,
    "i":INPUT,
    "save": save_template,
    "load": load_template,
    "print": lambda history: (print(history), inputProcess(input("\n:"),history)),
    "quit": lambda history: (save_chat(history), exit()),
    "exit": lambda history: (save_chat(history), exit()),
    "q": lambda history: (save_chat(history), exit()),
    "q!":exit,
    "reinput": reinput_line,
    "default": default_command,  # 添加默认命令
    "turing" :lambda history: turing(history),
    # "new":lambda history: (save_chat(history),history=[{"role": "system", "content": f"You are a helpful assistant."}]),
    "help": HELP,
}


def main():
    print(f"Welcome to the ChatGPT command line tool!\n")
    history = [{"role": "system", "content": f"You are a helpful assistant."}]
    i = 1
    
    while True:
        user_input = input(str(i) + " > user: ").strip()
        print()
        if user_input:
            history = inputProcess(user_input, history)
        else:
            print("Input your content")
            continue
        rsp_content = get_response(history)
        print(f"> ChatGPT: {rsp_content}\n")
        history.append({"role": "assistant", "content": rsp_content})
        i += 1

if __name__ == "__main__":
    main()