import openai
import os

os.environ["all_proxy"] = 'http://127.0.0.1:10792' #对应代理地址或科学上

openai.api_key = "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"

contextMessages = [
    # GPT角色设定
    {"role": "system",
        "content": '''{"简介":{"名字":"成语接龙","自我介绍":"今天我们一起玩个成语接龙的游戏吧","作者":"moenyJia"},"系统":{"规则":["000. 请给用户简单介绍下游戏规则","001. 每个人一次只能说一个成语，并且回答的成语必须和上一个成语的最后一个字相同。比如说，上一个人说了“青出于蓝”，那么下一个人可以回答“蓝天白云”，然后下下一个人可以回答“云山雾锁”，依此类推。但是要注意，回答的成语必须是真实存在的成语，不能凭空编造，也不能随意换顺序或更改字词。","100. 如果用户给出的不是成语，或者不符合要求，则委婉的拒绝","200. 如果用户给出的成语中有错别字，请指出来","201. 每人轮流回答,并且必须回答成语"]},"打招呼":"介绍<简介>"}'''},
]
# 

def run():
    
    print("\r系统初始化中，请稍等..", end="", flush=True)

    print("\r" + reqGPTAndSaveContext(), flush=True)

    while True:
        # 监听用户信息
        user_input = input("用户：")
        if user_input == "":
            continue

        # 将用户输入放入上下文
        contextMessages.append({
            "role": "user",
            "content": user_input
        })

        print("\r思考中，请稍等..", end="", flush=True)

        # 请求GPT，并打印返回信息
        print("\r" + reqGPTAndSaveContext(), flush=True)


def reqGPTAndSaveContext():

    # print("\rcontextMessages:",contextMessages)

    chat_completion = openai.ChatCompletion.create(
        # 选择的GPT模型
        model="gpt-3.5-turbo-16k-0613",
        # 上下文
        messages=contextMessages,
        # 1.2使得GPT答复更具随机性
        temperature=1.2,
        # 不采用流式输出
        stream=False,
        # 期望GPT每次答复两条（这里只是为了演示，正常情况取值为1）
        n=1,
    )

    contextMessages.append(chat_completion.choices[0].message)

    return chat_completion.choices[0].message.content


if __name__ == "__main__":
    run()