import openai
import argparse
import time
parser = argparse.ArgumentParser(
    description=
    '一些必须的文件')
parser.add_argument('--key',
                    type=str,
                    help='openai的key',
                    default = "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")
parser.add_argument('--savf',
                    type=str,
                    help='存储谈话的文件.',
                    default='res/conversation.txt')
args = parser.parse_args()

def is_japanese(string):
        for ch in string:
            if ord(ch) > 0x3040 and ord(ch) < 0x30FF:
                return True
        return False  

def gpt_chat(text):
  call_name = '机器人'
  openai.api_key = args.key
  identity = ''
  start_sequence = '\n'+str(call_name)+':'
  restart_sequence = "\nYou: "
  if 1 == 1:
     prompt0 = text #当期prompt
  if text == 'quit':
     return prompt0
  prompt = identity + prompt0 + start_sequence

  response = openai.Completion.create(
    model="text-davinci-003",
    prompt=prompt,
    temperature=0.5,
    max_tokens=1000,
    top_p=1.0,
    frequency_penalty=0.5,
    presence_penalty=0.0,
    stop=["\nYou:"]
  )
  return response['choices'][0]['text'].strip()


while True:
  your_text = input("你：")
  t1 = time.time()
  text = gpt_chat(your_text)
  t2 = time.time()
  print("api回复你总共用了", (t2 - t1), "s，真是太棒啦！")
  print('回答：'+text)
  text = text.replace('\n','。').replace(' ',',')
  text = f"[JA]{text}[JA]" if is_japanese(text) else f"[ZH]{text}[ZH]"
  with open(args.savf, "w", encoding="utf-8") as f1:
    f1.write(text)