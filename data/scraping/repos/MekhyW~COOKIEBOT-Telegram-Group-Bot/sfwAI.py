import openai
import json
import datetime
openai.api_key = ''

data = json.load(open('AI_SFW.json'))
questions_list = [q_a['prompt'] for q_a in data['questions_answers']]
answers_list = [q_a['completion'] for q_a in data['questions_answers']]
prompt_beginning = f"Você é um assistente engraçado, bobo e furry que adora zoar com os outros. Seu nome é CookieBot, e seu criador/pai se chama Mekhy. Responda as perguntas abaixo com respostas curtas! Dia de hoje: {datetime.datetime.now().strftime('%d/%m/%Y')}"
messages=[{"role": "system", "content": prompt_beginning}]
for i in range(len(questions_list)):
    messages.append({"role": "user", "content": questions_list[i]})
    messages.append({"role": "system", "content": answers_list[i], "name": "CookieBot"})

query = input("Pergunta: ")
messages.append({"role": "user", "content": query})
completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages, temperature=0.9)
answer = completion.choices[0].message.content

print(answer)