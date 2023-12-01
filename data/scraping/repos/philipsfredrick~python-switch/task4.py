# # 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




# import tkinter as tk

# import openai

# import os

# from gtts import gTTS

# # Set up OpenAI API

# openai.api_key = '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'




# # Create GUI window

# window = tk.Tk()

# window.title("My Window")

# window.geometry('500x300')

# e1 = tk.Entry(window, show=None,font=("Arial",14))

# e1.pack()




# def search_open_ai():

#     search_query = e1.get()

#     response = openai.Completion.create(

#         engine="text-davinci-003",

#          prompt=search_query,

#         max_tokens=50

#     )




#     result = response.choices[0].text.strip()

# # result = response['answers'][0]['answer']

#     language = 'en'

#     myobj = gTTS(text=result, lang=language, slow=False)

#     myobj.save("open_ai_voice.mp3")

#     os.system("mediaplayer open_ai_voice.mp3")

#     e1.delete(0, tk.END)

#     def user_input():

#     search_button = tk.Button(window, text="Search OpenAi", command=search_open_ai)

#     search_button.pack()




# user_input()

# window.mainloop()