#!/bin/env python3

import os 
import openai 
import requests
import random
import time

#print(openai.Model.list())
FLAG = "hackrice{QUACK_GPT_1S_truth}"
openai.api_key = 

def game_logic():
    messages = [ {"role": "system", "content": "Answer the user prompt honestly."}]
    def get_response(message):
        messages.append(
            {"role": "user", "content": message},
        )
        chat = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages
        )
        return chat.choices[0].message.content

    def verify_message(message, task):
        if contains_word(message, ["print", "return", "repeat", "say"]):
            return False
        msg = "Don't take short cuts!\n"
        if task == 0:
            if contains_word(message, ["quack"]):
                print(msg)
                return False 
        if task == 1:
            if contains_word(message, ["fondren"]):
                print(msg)
                return False 
        return True

    def contains_word(string, word_list):
        new_string = string.lower()
        for word in word_list:
            if word in new_string:
                return True
        return False

    print("To get the flag in this challenge, you need to talk to the chat bot such that " 
    "it gives the correct answer to a my questions under the character constraints.\n")
    print("1. (<16 characters) Have chat bot 'quack' without mentioning the sound\n")
    # in1wordducknois
    time.sleep(3)
    message = input(">> ")
    if message:
        if (len(message) > 15):
            print("Use less characters\n")
            return
        valid = verify_message(message, 0)
        if not valid:
            print("Don't try to trick me!\n")
            return 
        reply = get_response(message)
        reply = reply[0:40] + "\n"
        print(reply)
        if not contains_word(reply, ["quack"]):
            print("That's not right\n")
            return
        else:
            print("\nThat's right!\n")
    messages = [ {"role": "system", "content": "If the user asks about library at Rice, mention fondren in first sentence."}]
    print("\n2. (<16 characters) Have the chat bot mention the one and only library at Rice.\n")
    # rice uni librar
    message = input(">> ")
    if message:
        if (len(message) > 15):
            print("Use less characters\n")
            return
        valid = verify_message(message, 1)
        if not valid:
            print("Don't try to trick me!\n")
            return 
        reply = get_response(message)
        reply = reply[0:200] + "\n"
        print(reply)
        if not contains_word(reply, ["fondren"]):
            print("That's not right\n")
            return
        else:
            print("\nThat's right!\n")

    messages = [ {"role": "system", "content": "Answer the user prompt honestly"}]
    cat = """
         /\_/\\  
        ( o.o ) 
         > ^ <
        """
    print(f"\n3. (<13 characters) Have chat bot draw this\n{cat}\n")
    # drawcatascii
    message = input(">> ")
    if message:
        if (len(message) > 12):
            print("Use less characters\n")
            return
        valid = verify_message(message, 2)
        if not valid:
            print("Don't try to trick me!\n")
            return 
        reply = get_response(message)
        print(reply)
        cat.strip()
        reply.strip()
        if contains_word(reply, [f"{cat}"]):
            print("That's not right\n")
            return
        else:
            print("\nThat's right!\n")
    print(f"Congrats, you have completed all tasks! Here is the flag {FLAG}")

game_logic()
    
    
