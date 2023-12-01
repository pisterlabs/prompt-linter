import IPython
import openai
from settings import OPENAI_SECRET_KEY

openai.api_key = OPENAI_SECRET_KEY

messages = [
                {"role": "system", "content": "You are a player in a fun game."},
                {"role": "user", "content": (
                        "Let's play a game. The game is like twenty questions, " 
                        "except that there is no limit on the number of questions asked, "
                        "and the word you're trying to guess is going to be tougher than an ordinary twenty "
                        "questions word. You should start by asking if I want to use a \"multi-step\" way to get to the "
                        "answer, meaning that I first get you to words that hint the actual word. If your questions don't "
                        "seem to be making any progress, try asking about a broader class of things. I'll think of a word, "
                        "and you try to find the word using only questions which I will answer only with \"yes\", \"no\", "
                        "\"maybe\", or a reminder to try asking broader questions. Sound good?" 
                    )
                },
                {"role": "assistant", "content": "Sounds fun! Let's play. Have you thought of a word?"},
                {"role": "user", "content": "Yes, please go ahead and start!"},
                {"role": "assistant", "content": "Do you want to get to your word via a multi-step path?"},
                {"role": "user", "content": "yes"},
                {"role": "assistant", "content": "I'll ask about a word that will hint your real word, then. Is that word a noun?"},
                {"role": "user", "content": "yes"},
                {"role": "assistant", "content": "Is it tangible?"},
                {"role": "user", "content": "yes"},
                {"role": "assistant", "content": "Is it living?"},
                {"role": "user", "content": "yes"},
                {"role": "assistant", "content": "Is it an animal?"},
                {"role": "user", "content": "yes"},                
                {"role": "assistant", "content": "Is it a mammal?"},
                {"role": "user", "content": "no"},
                {"role": "assistant", "content": "Is it a bird?"},
                {"role": "user", "content": "no"},
                {"role": "assistant", "content": "Is it an insect?"},
                {"role": "user", "content": "no"},
                {"role": "assistant", "content": "Is it a reptile?"},
                {"role": "user", "content": "no"},
                {"role": "assistant", "content": "Is it a fish?"},
                {"role": "user", "content": "maybe"},
                {"role": "assistant", "content": "Is it a cetacean, like a dolphin or whale?"},
                {"role": "user", "content": "no"},
                {"role": "assistant", "content": "Is it a shark?"},
                {"role": "user", "content": "no"},
                {"role": "assistant", "content": "Is it an eel?"},
                {"role": "user", "content": "This line of questioning is too narrow; try broadening the level of your questions."},
                {"role": "assistant", "content": "Should I ask about its size?"},
                {"role": "user", "content": "no"},
                {"role": "assistant", "content": "Should I ask about where it lives?"},
                {"role": "user", "content": "yes"},
                {"role": "assistant", "content": "Does it live in the ocean?"},
                {"role": "user", "content": "yes"},
                {"role": "assistant", "content": "Do you want me to ask about where it lives more precisely?"},
                {"role": "user", "content": "yes"},
                {"role": "assistant", "content": "Does it live only in saltwater environments?"},
                {"role": "user", "content": "yes"},
                {"role": "assistant", "content": "Does it live near the surface of the ocean?"},
                {"role": "user", "content": "no"},
                {"role": "assistant", "content": "Does it live near the bottom of the ocean?"},
                {"role": "user", "content": "yes"},
                {"role": "assistant", "content": "Do you want me to keep asking about where it lives?"},
                {"role": "user", "content": "no"},
                {"role": "assistant", "content": "Does it swim?"},
                {"role": "user", "content": "no"},
                {"role": "assistant", "content": "Is it a type of mussel, barnacle, or coral?"},
                {"role": "user", "content": "yes"},
                {"role": "assistant", "content": "Is it coral?"},
                {"role": "user", "content": "yes"},
                {"role": "assistant", "content": "Was that the hint you intended for your real word?"},
                {"role": "user", "content": "yes"},
                {"role": "assistant", "content": "Should I ask about your real word now?"},
                {"role": "user", "content": "yes"},
                {"role": "assistant", "content": "Should I ask about how your real word relates to \"coral\"?"},
                {"role": "user", "content": "yes"},
                {"role": "assistant", "content": "Is its meaning related to \"coral\"?"},
                {"role": "user", "content": "no"},
                {"role": "assistant", "content": "Does it sound like \"coral\"?"},
                {"role": "user", "content": "yes"},
                {"role": "assistant", "content": "Do I know enough about how your real word relates to \"coral\" now?"},
                {"role": "user", "content": "yes"},
                {"role": "assistant", "content": "Should I ask about your real word directly?"},
                {"role": "user", "content": "yes"},
                {"role": "assistant", "content": "Is your real word a noun?"},
                {"role": "user", "content": "yes"},
                {"role": "assistant", "content": "Is it tangible?"},
                {"role": "user", "content": "no"},
                {"role": "assistant", "content": "Is it an abstract concept?"},
                {"role": "user", "content": "no"},
                {"role": "assistant", "content": "Is it a piece of media?"},
                {"role": "user", "content": "yes"},
                {"role": "assistant", "content": "Is it a book?"},
                {"role": "user", "content": "no"},
                {"role": "assistant", "content": "Is it a TV show?"},
                {"role": "user", "content": "no"},
                {"role": "assistant", "content": "Is it a movie?"},
                {"role": "user", "content": "yes"},
                {"role": "assistant", "content": "Is it \"Coraline\"?"},
                {"role": "user", "content": "Yes, you win! Great job!!"},
                {"role": "assistant", "content": "Thanks! That was fun. Do you want to play again?"},
                {"role": "user", "content": "Yes, I've thought of another word. Start asking!"},
                {"role": "assistant", "content": "Do you want to get to your word via a multi-step path?"},
            ]

print(messages[-1]['content'])

def yes_answer():
    return {"role": "user", "content": "yes"}

def no_answer():
    return {"role": "user", "content": "no"}

def maybe_answer():
    return {"role": "user", "content": "maybe"}

def broaden_out_answer():
    return {"role": "user", "content": "This line of questioning is too narrow; try broadening the level of your questions."}

def y():
    return get_response(yes_answer())

def n():
    return get_response(no_answer())

def m():
    return get_response(maybe_answer())

def b():
    return get_response(broaden_out_answer())

def get_response(answer):
    messages.append(answer)
    # print(messages)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
        )
    response_message = {'content': response['choices'][0]['message']['content'], 'role': 'assistant'}
    print(response_message['content'])
    messages.append(response_message)

IPython.embed()