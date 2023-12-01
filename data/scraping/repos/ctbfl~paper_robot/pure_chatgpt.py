import openai
import os
import secret

# 设置 API 密钥
openai.api_base = secret.openai_api_base
openai.api_key = secret.openai_api_key
os.environ["OPENAI_API_KEY"] = secret.openai_api_key
os.environ["OPENAI_API_BASE"] = secret.openai_api_base

def completion(messages):
  # 调用文本生成 API
  model = secret.gpt_model_name
  response = openai.ChatCompletion.create(
    model=model,
    messages=messages,
  )

  result = response["choices"][0]["message"]["content"]
  #print(result)
  return result

if __name__ == '__main__':
    history = history = [{'role':'system','content':"你是一个paper robot, 性格开朗活泼, 你接下来会收到用户的交流，请你活泼开朗的回复他"}]
    text = "下午好呀"
    print("你：",text)
    user_content = {'role':'user','content':text}
    history.append(user_content)
    print(history)
    chatgpt_response = completion(text)
    history.append({'role':'assistant','content':chatgpt_response})
    print("paper_robot:",chatgpt_response)