import openai
import os
import argparse

# Set up OpenAI API credentials
openai.api_key = os.environ["OPENAI_API_KEY"]

def query(args):
    response = openai.ChatCompletion.create(
                    model=args.model,
                    messages=[
                        {"role": "user", "content": args.q },
                    ],
                    )

    return response["choices"][0]["message"]["content"]

def chat(messages, args):
    response = openai.ChatCompletion.create(
                    model=args.model,
                    messages=messages,
                    )

    return response["choices"][0]["message"]["content"]

if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--q', type=str, default=None, help='query to ask the AI')
    # add flag for chat mode
    argparse.add_argument('--c', action='store_true', help='chat mode')
    argparse.add_argument('--model', type=str, default='gpt-3.5-turbo', help='openai model to use, default "gpt-3.5-turbo" or use "gpt-4-1106-preview"')
    args = argparse.parse_args()
    
    
    if args.c:
        messages = []
        while True:
            inp = input("You: ")
            if inp == "exit":
                break
            messages.append({"role": "user", "content": inp})
            response = chat(messages,args)
            print("AI: " + response)
            messages.append({"role": "assistant", "content": response})
    elif args.q:
        response = query(args)
        print(response)
    
