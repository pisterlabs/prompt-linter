
import sys
import openai
import random
import json
import tqdm
import yaml
import random


# reads above file from config.yaml file
config = yaml.load(open('config.yaml'), Loader=yaml.FullLoader)
num_tweets = config['twitter']['num_tweets']
engine = config['openai']['openai_engine']
api_key = config['openai']['openai_key']
persona = config['twitter']['twitter_handle']

openai.api_key = api_key



def generate_command_prompt_header(twitter_data, persona, n_turns):
    """
    Generate a command to be sent to the chatbot API
    args:
        persona: name of the persona
        n_turns: number of turns of the conversation
    """
    command = f"use the tweets below to build a virtual persona named {persona}:\n" 
    command += f"1) produce a summary of the character personality in 1-2 sentences,\n"
    command += "so that it can be used to give a context to a conversation with an agent.\n"
    command += "start your text with: {persona} is a persona with the following interests and attribute\n\n"

    for d in random.choices(twitter_data, k=n_turns):
        command += json.dumps(d) + "\n"
    
    return command

def generate_command_virtual_conversation_data(twitter_data, persona, n_turns=10):
    """
    Generate a command to be sent to the chatbot API
    args:
        twitter_data: list of tweets
        persona: name of the persona
        n_turns: number of turns of the conversation
    """
    command = f"Use the tweets below to build a virtual persona named {persona}:\n"
    command += f"1) simulate {n_turns} turns of a hypothetic conversation between rdp and a random user that never spoke together\n"
    command += f"2) the random user starts the conversation with a random realistic sentence\n"
    command += f"3) generate a list of python data dictionaries [{{\"prompt\": ... , \"completion\": ...}},...] for these {n_turns} turns\n" 
    command += "4) return only a json dump of the above list.\n\n" 
    
    for d in random.choices(twitter_data, k=n_turns):
        command += json.dumps(d) + "\n"
    
    return command


def call_chat_gpt(command, max_tokens=2040, temperature=0.5, n=1, stop=None):
    """
    Call the OpenAI chatbot API
    args:
        command: the command to be sent to the chatbot
        max_tokens: maximum number of tokens to be generated
        temperature: temperature of the chatbot
        n: number of responses to be generated
    """
    response = openai.Completion.create(
        engine=engine,
        prompt=command,
        max_tokens=max_tokens,
        temperature=temperature,
        n = 1,
        stop=stop,
    )

    return response


def generate_virtual_conversation_data(twitter_data, persona, n_turns=10):
    """
    Generate a virtual conversation between a persona and a random user
    args:
        twitter_data: list of tweets
        persona: name of the persona
        n_turns: number of turns of the conversation
    """
    command = generate_command_virtual_conversation_data(twitter_data, persona, n_turns)    
    virtual_conversation = None

    try:
        response = call_chat_gpt(command)
        virtual_conversation = response["choices"][0]["text"]
        idx = virtual_conversation.index("[")
        virtual_conversation = json.loads(virtual_conversation[idx:])
    except:
        pass

    return virtual_conversation


def generate_persona(twitter_data, persona, n_turns=30):
    """
    Generate a persona from a list of tweets
    args:
        twitter_data: list of tweets    
        persona: name of the persona
        n_turns: number of tweets to use to generate the persona
    """
    command = generate_command_prompt_header(twitter_data, persona, n_turns)
    response = call_chat_gpt(command, max_tokens=200, temperature=0.7, n=1)
    
    return response["choices"][0]["text"]


def generate_training_data(twitter_data, persona, n_turns, memory_length=10):
    """
    Generate training data for the chatbot
    args:
        twitter: list of tweets
        persona: name of the persona
        n_turns: number of turns of the conversation
    """
    virtual_conversation_data = generate_virtual_conversation_data(twitter_data, persona, n_turns)
    training_data = None
    persona_data = None
    if not virtual_conversation_data is None:
        
        try:
            persona_data = generate_persona(twitter_data, persona, 30)
        except:
            return None

        training_data = []
        for i in range(len(virtual_conversation_data)):
            
            try:

                print(i)
                s = persona_data+"\n"
                s+= "memory: \n\n"

                #print("###############################################")
                #print("memory", memory_length)
                #print(virtual_conversation_data)
                #import sys
                #sys.exit()
                #print("###############################################")

                for vc in virtual_conversation_data[max(0,i-memory_length):i]:
                
                    s += "user: "+vc["prompt"]+"\n"
                    s += "agent: "+vc["completion"]+"\n"    
                    s += "\n"

                s += "user: "+virtual_conversation_data[i]["prompt"]+"\n"

                d = {"prompt": s, "completion": virtual_conversation_data[i]["completion"]}
                training_data.append(d)

            except:
                pass

        return persona_data, training_data if persona_data is not None and training_data is not None \
              else None



if __name__ == '__main__':

    # read the twitter handle from the command line
    if len(sys.argv) != 3:
        print("Usage: python random_tweets.py <character> <twitter_handle>")
        sys.exit(1)
    character = " ".join(sys.argv[1].split("_"))
    twitter_handle = sys.argv[2]


    # load twitter data
    file = f"data/twitter_data_{twitter_handle}.jsonl"
    # load data
    with open(file) as f:
        twitter_data = [json.loads(line[:-1]) for line in f.readlines()]
    
    persona_list = []

    n_turns = 15
    persona = twitter_handle
    train_data_list = []
        
    percent = 0.
    persona_list = []
    with open(f"data/train_twitter_data_{twitter_handle}.jsonl", "w") as file_training:
        
        while len(train_data_list) < num_tweets:
            
            print("len train_data: ", len(train_data_list))
            data = generate_training_data(twitter_data, persona, n_turns)
            if data is not None:
                persona_data, training_data = data
                #print("persona_data", persona_data)
            
                # append personas
                if persona_data is not None:
                    persona_list.append(persona_data)
                    #file_personas.write(persona_data+"\n") 
                    print("persona_list",len(persona_list))

                # write line by line
                if training_data is not None:
                    for d in training_data:
                        file_training.write(json.dumps(d) + "\n")

                    train_data_list.extend(training_data)
            
                if len(train_data_list) / num_tweets > percent + 0.1:
                    percent = len(train_data_list) / num_tweets + 0.1
                    print("completion percent:",percent*100,"%")

    # finally write persona into file
    with open(f"data/persona_list_{twitter_handle}.jsonl", "w") as file_personas:
        for p in persona_list:
            file_personas.write(p+"\n")

  

