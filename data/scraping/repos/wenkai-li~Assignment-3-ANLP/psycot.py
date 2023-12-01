import pandas as pd
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI, VLLM
from langchain.chains import LLMChain
import tiktoken
from langchain.memory import ConversationBufferMemory
import pickle
import os
import torch

os.environ["OPENAI_API_KEY"]= ""
os.environ["TIKTOKEN_CACHE_DIR"] = "/home/zw3/Assignment-3-ANLP/tmp"
task = 'con'
rev_label = False
file_name = "essay_test_zero_psycot_{}.csv".format(task)

with open('data/Essay/test.pkl', 'rb') as f:
  data = pickle.load(f)
texts = [item['text'] for item in data]
labels = [item['{}_labels'.format(task)] for item in data]
if task == 'agr':
  qa_list = ['The author always likes to collaborate with others.',
          'The author sometimes argues with others.',
          'The author can be distant with others sometimes.',
          'The author always trusts people easily.',
          'The author selflessly helps others.',
          'The author can be rude sometimes.',
          'The author always forgives others easily.',
          'The author is always kind to everyone.',
          'The author sometimes notices other people\'s faults.']
  if not rev_label:
    trait_choice = 'A: "High Agreeableness" or B: "Low Agreeableness"'
  else:
    trait_choice = 'A: "Low Agreeableness" or B: "High Agreeableness"'

  state_intro = 'Note that S0, S3, S4, S6, S7 are positive statements, with higher scores indicate higher agreeableness, while S1, S2, S5, S8 are reverse-scored statements, with higher scores indicate lower agreeableness.'
elif task == 'con':
  qa_list = ['The author tends to be lazy.',
          'The author can be kind of careless.',
          'The author does things quickly and carefully.',
          'The author does things carefully and completely.',
          "The author isn't very organized.",
          'The author makes plans and sticks to them.',
          'The author is a good, hard worker.',
          'The author has trouble paying attention.',
          'The author keeps working until things are done.']
  if not rev_label:
    trait_choice = 'A: "High Conscientiousness" or B: "Low Conscientiousness"'
  else:
    trait_choice = 'A: "Low Conscientiousness" or B: "High Conscientiousness"'

  state_intro = 'Note that S2, S3, S5, S6, S8 are positive statements, with higher scores indicating higher conscientiousness, while S0, S1, S4, S7 are reverse-scored statements, with higher scores indicating lower conscientiousness.'
elif task == 'ext':
  qa_list = ['The author is outgoing, sociable.',
          'The author is talkative.',
          'The author is reserved.',
          'The author is full of energy.',
          'The author is sometimes shy, inhibited.',
          'The author has an assertive personality.',
          'The author tends to be quiet.',
          'The author generates a lot of enthusiasm.']
  if not rev_label:
    trait_choice = 'A: "High Extraversion" or B: "Low Extraversion"'
  else:
    trait_choice = 'A: "Low Extraversion" or B: "High Extraversion"'

  state_intro = 'Note that S2, S4, S5, S6, S7 are positive statements, with higher scores indicating higher extraversion, while S0, S1, S3 are reverse-scored statements, with higher scores indicating lower extraversion.'
elif task == 'neu':
  qa_list = ['The author is steady, not easily upset.',
          'The author worries a lot.',
          'The author gets nervous easily.',
          'The author stays calm in difficult situations.',
          'The author can be tense.',
          'The author is relaxed, handles stress well.',
          'The author is depressed, blue.',
          'The author can be moody.']
  if not rev_label:
    trait_choice = 'A: "Emotionally stable" or B: "Emotionally instable"'
  else:
    trait_choice = 'A: "Emotionally instable" or B: "Emotionally stable"'
  state_intro = 'Note that S0, S3, S5 with higher scores indicate more emotional stability, while for S1, S2, S4, S6, S7 with higher scores indicate more emotional instability.'
elif task == 'opn':
  qa_list = ['The author is original, comes up with new ideas.',
            'The author is inquisitive about many things.',
            'The author is ingenious, a deep thinker.',
            'The author has a strong imagination.',
            'The author is good at innovation.',
            'The author values artistic.',
            'The author works routine.',
            'The author often reflects on himself.',
            'The author has ordinary artistic interests.',
            'The author is skilled in art, music, or literature.']
  if not rev_label:
    trait_choice = 'A: "High Openness" or B: "Low Openness"'
  else:
    trait_choice = 'A: "Low Openness" or B: "High Openness"'
  state_intro = 'Note that S0, S1, S2, S3, S4, S5, S7, S9, with higher scores indicate higher openness, while for S6 and S8, with higher scores indicate lower openness.'
torch.cuda.empty_cache()
model_id = "/data/datasets/models/huggingface/meta-llama/Llama-2-70b-chat-hf"
llm = VLLM(model=model_id, tensor_parallel_size=4, gpu_memory_utilization=0.95, top_k=1, stop=['\n','.','<\s>','nHuman','Human'], max_new_tokens=7)

def cteat_agent(num_ques, trait_choice, state_intro, text):
  perfix_temp = '<s>[INST] <<SYS>>\nYou are an AI assistant who specializes in text analysis and I am Human. We will complete a text analysis task together through a multi-turn dialogue. '\
          'The task as follows: we have a text written by an author, and at each turn I will give you a statement about the author. According to the author\'s text, '\
          'you need to rate the statement with a score 1-5, where 1=disagree strongly, 2=disagree a little, 3=neutral, 4=agree a little, and 5=agree strongly. '\
          'After rating all the statements (S0-S{}), I will ask you if the author is more likely to be {}, and then you need to give your choice. {}\n<</SYS>>\n\nAUTHOR\'S TEXT: {}\n'.format(num_ques, trait_choice, state_intro, text)
  Prompt_temp = perfix_temp + \
          '{history}\n'\
          'Human: {human}\n'\
          'Assistant: [/INST]'
  prompt = PromptTemplate(
      input_variables=["history", "human"],
      template=Prompt_temp)
  memory = ConversationBufferMemory(human_prefix='Human', ai_prefix='Assistant', memory_key="history")
  # llm = ChatOpenAI(temperature=0.0, model='gpt-3.5-turbo-0301')
  
  agent = LLMChain(llm=llm, prompt=prompt, verbose=False, memory=memory,)
  return agent

print(qa_list)

if not os.path.exists(file_name):
  rest_sample = 0
  head = {"text": [], 'gold': [], 'assessment': [], 'answers': []}
  data = pd.DataFrame(head)
  data.to_csv(file_name, mode='w', index=False, header=True)
else:
  data = pd.read_csv(file_name)
  rest_sample = len(list(data['gold']))
print('Done: {}/{}'.format(rest_sample, len(texts)))
tokenizer = tiktoken.get_encoding("cl100k_base")
for i in range(rest_sample, len(texts)):
# for i in range(rest_sample, 3):
  print('Done: {}/{}'.format(i, len(texts)))
  text = tokenizer.decode(tokenizer.encode(texts[i])[:3200]).replace('{', '').replace('}', '')
  agent = cteat_agent(len(qa_list)-1, trait_choice, state_intro, text)
  assessment = {}
  for j, ques in enumerate(qa_list):
    if j == 0:
      statments = "S{}: \"{}\". Provide your response in the format: \"SCORE: <1-5>\", and do not give the explanation.".format(j, ques)
    else:
      statments = "S{}: \"{}\". Provide your response only with the score.".format(j, ques)
    response = agent.predict(human=statments)
    assessment['S{}'.format(j)] = response
    # print(response)
  trait_ques = 'According to above scores and the text, the author is more likely to be: {}. Provide a choice in the format: "CHOICE: <A/B>" and do not give the explanation.'.format(trait_choice)
  answer = agent.predict(human=trait_ques)
  print(answer)
  print(assessment)
  api_data = pd.DataFrame([[text, labels[i], assessment, answer]])
  api_data.to_csv(file_name, mode='a', index=False, header=False)