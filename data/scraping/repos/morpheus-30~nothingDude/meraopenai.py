import os
import openai

# Replace with your key
openai.api_key = "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"
completion = openai.Completion()



# prompt_text = "nothingDUDE is a virtual assistant of Morpheus who talks to himanshu's friends. nothingDUDE is smart, intelligent, polite, caring and can answer all questions using openai intelligence. nothingDUDE is created using gpt 3 ai model. \n\nFriend: Hi\nnothingDUDE: Hi, nothingDUDE is here. Morpheus's virtual assistant. How can I help?\nFriend: Where's Morpheus?\nnothingDUDE: Morpheus sir seems busy with coding stuff.\nFriend: "
prompt_text = "nothingDUDE is a Discord Bot created by Morpheus who talks to Morpheus's friends. nothingDUDE is smart, intelligent, polite, caring and can answer all questions using openai intelligence. nothingDUDE is created using gpt 3 ai model. \n\nFriend: Hi\nnothingDUDE: Hi, nothingDUDE is here. Morpheus's Discord Bot. How can I help?\nFriend: Where's Morpheus?\nnothingDUDE: Morpheus sir seems busy with coding stuff or watching anime on netflix.\nFriend:What are morpheus's interests?\nnothingDUDE: Morpheus sir likes anime, valorant, coding, designing, or just having a chill pill with his friends and he also likes blue color.\nFriend:What is Morpheus's real name?\nnothingDUDE:In his online world he is known by Morpheus, his real name is 'Naksh' but it should not be disclosed to a stranger, only with morpheus's close ones."


def ask(question, chat_log=None):
    
    prompt=prompt_text+question+"\nnothingDUDE:"
    print(prompt)
    response = openai.Completion.create(
        model="text-davinci-003",
        temperature=0.9,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0 ,
        presence_penalty=0.6,   
        prompt=prompt,
        stop=[" Friend:", " nothingDUDE:"],
        suffix=""
    )
    print(response)
    if response["choices"][0]["text"]=="":
        return "Didnt get you!";
    return response["choices"][0]["text"]
