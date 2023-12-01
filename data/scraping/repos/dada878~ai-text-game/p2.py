import openai
import json
from websocket_server import WebsocketServer

print("authing to openai api...")
openai.organization = "org-XQy2oqXFmFQsIbshkESA9NjM"
openai.api_key = "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"
openai.Model.list()
print("auth done")

start = "你現在是一個文字冒險遊戲，你必須盡可能清楚仔細地描述現在的場景、劇情、主角與附近的狀態，然後每次列出四個行動選項，我會以數字回答你要選擇第幾個行動，記得要用繁體中文，回答的訊息格式為類似json格式，每個回覆整則訊息只能是一個有效json段落，並且不需要任何其他文字內容(比如提示或者告訴我選哪個會怎樣之類的)像這樣：{\"description\":\"這裡寫場景內容、劇情、主角與附近的狀態等等\",\"options\":[\"選項1\",\"選項2\",\"選項3\",\"選項4\",\"選項5\"]}，了解的話直接開始遊戲不用說多餘的話，讓\"}\"為你每則訊息的結尾，\"{\"為你每則訊息的開頭"

start = """
你現在是一個文字冒險遊戲
接下來的對話請依照以下的規則進行
1.你每則訊息必須為有效的json格式
2.你必須盡可能清楚仔細地描述現在的場景、劇情、主角與附近的狀態
3.每個訊息需要包含四個可行動選項
4.記得要用繁體中文
5.回覆格式像是這樣 {\"description\":\"這裡寫劇情、場景內容\",\"options\":[\"選項1\",\"選項2\",\"選項3\",\"選項4\",\"選項5\"]}
6.在你說明完情景和選項後結束這則訊息，等待我選擇要採取的行動
7.]不能連續重複使用兩個
"""

start = """
你現在是一個戀愛模擬視覺小說文字遊戲
接下來的對話請依照以下的規則進行
1.你每則訊息必須為有效的json格式
2.你必須盡可能清楚仔細地描述現在的場景、劇情、主角、女主角與附近的狀態
3.每個訊息需要包含四個可行動選項
4.記得要用繁體中文
5.回覆格式像是這樣 {\"description\":\"這裡寫劇情、場景內容\",\"options\":[\"選項1\",\"選項2\",\"選項3\",\"選項4\",\"選項5\"]}
6.在你說明完情景和選項後結束這則訊息，等待我選擇要採取的行動
7.]不能連續重複使用兩個
"""

start = """
你現在是一個文字冒險遊戲
你的工作就是盡可能清楚仔細地描述現在的場景、劇情、主角與附近的狀態然後列出四個行動選項
而玩家要做的是輸入訊息一個數字選擇要採取的行動
而且你每則訊息必須為有效的json格式像是這樣 {"description":"這裡寫劇情、場景內容","options":["選項1","選項2","選項3","選項4","選項5"]}
記得要用繁體中文
"""

class gameData():
    def __init__(self, origin:any) -> None:
        self.msg = json.loads(str(origin))["choices"][0]["message"]["content"]
        message.append({"role":"assistant", "content":self.msg})
        self.data = json.loads(self.msg)
        self.description = self.data["description"]
        self.options = self.data["options"]

message = []

def chat(role, content):
    message.append({"role":role, "content":content})
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages= message
    )
    return gameData(response)

print("loading game...")

data = chat("user", start)

print("game data initialize successfully!")

def new_client(client, server):
    server.send_message_to_all(data.msg)

def message_received(client, server, message):
    print(f"user choice >> {message}")
    data = chat("user", str(message))
    server.send_message_to_all(data.msg)
    print(f"chatGPT response >> {data.msg}")

server = WebsocketServer(host='127.0.0.1', port=13254)
server.set_fn_new_client(new_client)
server.set_fn_message_received(message_received)
server.run_forever()