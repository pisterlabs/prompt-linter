
from langchain.llms import OpenAI

from langchain.agents import initialize_agent , Tool

from langchain.chains.conversation.memory import ConversationBufferMemory

from dotenv import load_dotenv 
import os
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
print('ok')


def fibonacci_of(n):
     if n in {0, 1}:  # Base case
        return n
     return fibonacci_of(n - 1) + fibonacci_of(n - 2)  # Recursive case

def sort_string(string):
  return ''.join(sorted(string))

def toggle_case(word):
    toggled_word = ""
    for char in word:
        if char.islower():
            toggled_word += char.upper()
        elif char.isupper():
            toggled_word += char.lower()
        else:
            toggled_word += char
    return toggled_word


def encrypt_word(word):
    encrypted_word = ''
    for letter in word:
        if letter.isalpha():
            if letter.isupper():
                encrypted_word += chr((ord(letter) - 65 + 13) % 26 + 65)
            else:
                encrypted_word += chr((ord(letter) - 97 + 13) % 26 + 97)
        else:
            encrypted_word += letter
    return encrypted_word



def decrypt_word(word):
    decrypted_word = ''
    for letter in word:
        if letter.isalpha():
            if letter.isupper():
                decrypted_word += chr((ord(letter) - 65 - 13) % 26 + 65)
            else:
                decrypted_word += chr((ord(letter) - 97 - 13) % 26 + 97)
        else:
            decrypted_word += letter
    return decrypted_word

tools = [
    Tool(
        name = "Fibonacci",
        func = lambda n: str(fibonacci_of(int(n))),
        description = "use when you want calculate the nth fibonacci number",
    ),
    Tool(
        name = "Sort String",
        func = lambda string: sort_string(string),
        description = "use when you want sort a string alphabetically",
    ),
    Tool(
        name = "Toogle_case",
        func = lambda word: toggle_case(word),
        description = "use when you want covert the letter to uppercase or lowercase",
    ),
    Tool(
        name = "Encrypt_word",
        func = lambda word: encrypt_word(word),
        description = "use when you want to encrypt a word",
    ),
    Tool(
        name = "decrypt_word",
        func = lambda word: decrypt_word(word),
        description = "use when you want to decrypt a word",
    ),
]

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

llm = OpenAI(temperature = 0 , verbose=True)

agent_chain = initialize_agent(tools, llm, agent = "conversational-react-description", memory=memory,verbose=True)

# agent_chain.run("what is the 10th number of fibonaci series")
# agent_chain.run("what is decrypt of 'i love you")
# agent_chain.run("Sort hello")
agent_chain.run("convert the word I love YOU to uppercase or lowercase")