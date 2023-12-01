import openai
import argparse

import argparse

parser = argparse.ArgumentParser(description='Talk to a custom GPT3 Model')
parser.add_argument('model', type=str, default='curie', help='Model Name')
parser.add_argument('--prompt', type=str , default='a human interviews an AI about its opinions on a topic', help='context for the conversation')
parser.add_argument('--backforths', type=int , default='3', help='how long the bot will remember previoys responses')
parser.add_argument('--max_tokens', type=int, default=75, help='How long responces can be')
parser.add_argument('--temp', type=float, default=0.9, help='How whacky a response can be')
parser.add_argument('--presence_penalty', type=float, default=0.1 , help='Penalty for repeated topics')
args = parser.parse_args()

log = "Context: " + args.prompt + "\n\n###\n\n"
counter = 0

def resp(human, log):
    log+="Human: " + human + "\nAI:"
    response = openai.Completion.create(
        model=args.model,
        prompt=log,
        temperature=args.temp,
        max_tokens=args.max_tokens,
        presence_penalty = args.presence_penalty,
        stop=["\n", "AI:"]
    )
    log+= str(response.choices[0]['text']) + "\n"
    print("AI:" + str(response.choices[0]['text']))
    return log

print("Start the conversation by saying something below")
while __name__ == "__main__":
    human = input("You: ")
    log = resp(human, log)
    counter+=1
    if(counter % args.backforths == 0): 
        log = "Context: " + args.prompt + "\n\n###\n\n"
        print("log cleared")


