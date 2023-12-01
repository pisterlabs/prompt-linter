import openai
import tiktoken
import json
import os
import requests

os.environ["all_proxy"] = 'http://127.0.0.1:10792' #对应代理地址或科学上

openai.api_key = "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"

llm_system_prompt = "You are an assistant that provides news and headlines to user requests. Always try to get the lastest breaking stories using the available function calls."
llm_max_tokens = 15500
llm_model = "gpt-3.5-turbo-16k"
encoding_model_messages = "gpt-3.5-turbo-0613"
encoding_model_strings = "cl100k_base"
function_call_limit = 3

# 计算token数量
def num_tokens_from_messages(messages):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(encoding_model_messages)
    except KeyError:
        encoding = tiktoken.get_encoding(encoding_model_strings)

    num_tokens = 0
    for message in messages:
        num_tokens += 4
        for key, value in message.items():
            num_tokens += len(encoding.encode(str(value)))
            if key == "name":
                num_tokens += -1
    num_tokens += 2
    return num_tokens

# 获取消息
def get_top_headlines(query: str = None, country: str = None, category: str = None):
    """Retrieve top headlines from newsapi.org (API key required)"""

    base_url = "https://newsapi.org/v2/top-headlines"
    headers = {
        "x-api-key": "f1bb8b3914444bb5b31abcf86a627913"
    }
    params = { "category": "general" }
    if query is not None:
        params['q'] = query
    if country is not None:
        params['country'] = country
    if category is not None:
        params['category'] = category

    # Fetch from newsapi.org - reference: https://newsapi.org/docs/endpoints/top-headlines
    response = requests.get(base_url, params=params, headers=headers)
    data = response.json()

    if data['status'] == 'ok':
        print(f"Processing {data['totalResults']} articles from newsapi.org")
        return json.dumps(data['articles'])
    else:
        print("Request failed with message:", data['message'])
        return 'No articles found'


# functions 传给GPT的functions
signature_get_top_headlines = {
    "name": "get_top_headlines",
    "description": "获取按国家和/或类别分类的头条新闻。",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "自由输入关键词或短语进行搜索。",
            },
            "country": {
                "type": "string",
                "description": "要获取头条新闻的国家的2位ISO 3166-1代码。",
            },
            "category": {
                "type": "string",
                "description": "要获取头条新闻的类别",
                "enum": ["business","entertainment","general","health","science","sports","technology"]
            }
        },
        "required": [],
    }
}

# 调用GPT接口
def complete(messages, function_call: str = "auto"):
    """Fetch completion from OpenAI's GPT"""

    messages.append({"role": "system", "content": llm_system_prompt})

    # delete older completions to keep conversation under token limit
    while num_tokens_from_messages(messages) >= llm_max_tokens:
        messages.pop(0)

    print('Working...')

    print('1-messages:',messages)
    
    res = openai.ChatCompletion.create(
        model=llm_model,
        messages=messages,
        functions=[signature_get_top_headlines],
        function_call=function_call
    )
    
    print('res:',res)

    print('2-messages:',messages)
    
    # remove system message and append response from the LLM
    messages.pop(-1)

    print('3-messages:',messages)


    response = res["choices"][0]["message"]
    messages.append(response)

    print('4-messages:',messages)

    # call functions requested by the model
    if response.get("function_call"):
        function_name = response["function_call"]["name"]
        if function_name == "get_top_headlines":
            args = json.loads(response["function_call"]["arguments"])
            headlines = get_top_headlines(
                query=args.get("query"),
                country=args.get("country"),
                category=args.get("category")        
            )
            messages.append({ "role": "function", "name": "get_top_headline", "content": headlines})
        # elif function_name == "get_zsxq_article":
        #     args = json.loads(response["function_call"]["arguments"])
        #     headlines = get_zsxq_article(query=args.get("query"))
        #     messages.append({ "role": "function", "name": "get_top_headline", "content": headlines})
    print('5-messages:',messages)

# 运行程序
def run():
    print("\n你好，我是你的小助手，你有什么问题都可以问我噢～")
    print("你可以这样问我:\n - 告诉我最近有什么技术发现？\n - 最近的体育有什么新闻\n - 知识星球最近有什么精彩内容\n")

    messages = []
    while True:
        prompt = input("\n你想知道些什么? => ")
        messages.append({"role": "user", "content": prompt})
        complete(messages)

        # the LLM can chain function calls, this implements a limit
        call_count = 0
        while messages[-1]['role'] == "function":
            call_count = call_count + 1
            if call_count < function_call_limit:
                complete(messages)
            else:
                complete(messages, function_call="none")

            # print last message
            print("\n\n==Response==\n")
            print(messages[-1]["content"].strip())
            print("\n==End of response==")

if __name__ == '__main__':
    run()


# e6fff9c9c48e4603a8d0128db08e819d

