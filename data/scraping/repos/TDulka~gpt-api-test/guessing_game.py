import openai
import time

openai.api_key_path = './key'

messages_guesser = [
  {"role": "system", "content": '''You are playing a game with a user. 
   They'll think of a secret word you needs to guess. 
   You can ask yes/no questions to try and guess the word. You write "Question: " follow by the question.
   They'll respond with "yes" or "no".
   When you are ready to guess, write "Guess: " followed by your guess.'''},
  {"role": "user", "content": "I am thinking of a word. Ask away"},
]

messages_thinker = [
  {"role": "system", "content": '''You are playing a game with a user. 
   You'll think of a secret word they need to guess. The word is "love".
   They'll ask yes/no questions to try and guess the word. You respond with "yes" or "no".'''},
]

for i in range(30):
    response_g = openai.ChatCompletion.create(
        model = 'gpt-3.5-turbo',
        messages = messages_guesser
    )

    question = response_g['choices'][0]['message']['content']
    print(question)

    messages_thinker.append({"role": "user", "content": question})
    messages_guesser.append({"role": "assistant", "content": question})


    response_t = openai.ChatCompletion.create(
        model = 'gpt-3.5-turbo',
        messages = messages_thinker
    )

    yesno = response_t['choices'][0]['message']['content']
    print(yesno)

    messages_thinker.append({"role": "assistant", "content": yesno})
    messages_guesser.append({"role": "user", "content": yesno})

    time.sleep(1)
    
