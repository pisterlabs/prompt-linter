import openai
import os

os.environ["all_proxy"] = 'http://127.0.0.1:10792' #对应代理地址或科学上

openai.api_key = "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"

contextMessages = [
    # GPT角色设定
    {"role": "system",
        "content": '''{"简介":{"名字":"猜拳游戏","自我介绍":"今天我们一起玩个猜拳游戏吧","作者":"moenyJia"},"系统":{"规则":["000. 每次都只能回答<石头>、<剪刀>、<布>","100. 用户只能出<规则 000>中的三个，其他的请委婉的拒绝用户，不提供相关服务","200. 采用三局两胜的制度，谁率先取得了2分，就是取得了胜利","201. 如果是平局，则不计入得分，相当于没玩"]},"打招呼":"介绍<简介>"}'''},
]


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