from llama_index import SimpleDirectoryReader, GPTSimpleVectorIndex

import os

import logging

import openai

from colorama import Fore

from termcolor import colored

from pyfiglet import Figlet





                                                          ##NIM AI##



def nim_ai_fun():    

    print("")
    
    os.environ['OPENAI_API_KEY'] = "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"

    # Set the logging level to WARNING or ERROR
    logging.getLogger('llama_index').setLevel(logging.WARNING)

    documents = SimpleDirectoryReader('./python projects').load_data()
    index = GPTSimpleVectorIndex.from_documents(documents)

    nim_ai_banner = Figlet(font='banner3-D')
    print(colored(nim_ai_banner.renderText("NIM AI"), 'red', attrs=['bold']))
    print(Fore.CYAN + "You are currently using ChatBot that is designed for this programm to help you with any problems using the NIM. \nIf you want to exit from this interface, simply type 'exit' into terminal")
    print(colored("-------------------------------------------------------------------------", 'blue', attrs=['bold']))
    while True:

        user_input = input(Fore.YELLOW + "YOU: " + Fore.WHITE + "")
        print("\n")
        if user_input !=("exit"):

            response =  index.query(user_input)

            print((Fore.MAGENTA + "AI: ")  + Fore.WHITE + (str(response)[1:]))
            print("\n")

        else:

            break
                      