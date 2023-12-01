import openai
import json
#from jieba_cn import generateQuestions #中文
from jiagu_cn import generateQuestions #中文
#from nltk_en import generateQuestions #英文
import os
openai.api_key = ""  # 请将 YOUR_API_KEY 替换为您的 API 密钥
proxy_address = 'http://127.0.0.1:7890'
os.environ['HTTP_PROXY'] = proxy_address
os.environ['HTTPS_PROXY'] = proxy_address
def genAnswer(input_text,question):
    prompt="请根据以下内容回答问题:'"+input_text+"',问题:"+question+",如果根据文本无法回答问题，请根据你的知识直接给出回答"
    prompt_en="please answer queition based on these text:'"+input_text+"',quesition:"+question+",if you can't answer based on given text,please give answer by what you know!"
    completions = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=3096,
        n=1,
        stop=None,
        temperature=0.7
    )
    answer = completions.choices[0].text.strip()
    print('question:',question)
    print('answer:',answer)
    return answer

def genQA(input_text,savepath=None):
    result=[]
    #生成问题=>question
    questions=generateQuestions(input_text)
    #让gpt根据文本内容回答=>answer
    for question in questions:
        answer=genAnswer(input_text,question)
        qa={"question":question,"answer":answer}
        result.append(qa)
    if savepath:
        with open(savepath,'w+',encoding="utf-8") as f:
            json.dump(result,f,ensure_ascii=False,indent=4)
    return result

if __name__=="__main__":
    text_cn = "乔布斯是美国苹果公司的创始人之一,1955年出生于加利福尼亚。"
    text_en="Steve Jobs and Steve Wozniak founded Apple Inc. in 1976. The company is headquartered in Cupertino, California."
    genQA(text_cn,'./qa_cn_jiagu.json')

    