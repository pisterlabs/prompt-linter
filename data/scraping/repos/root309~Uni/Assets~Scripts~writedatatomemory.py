from multiprocessing import shared_memory
import time
import openai
import os

# 共有メモリの名前
shm_name = "my_memory"

# 共有メモリを開く
shm = shared_memory.SharedMemory(name=shm_name)
buf = shm.buf

# バイト形式のデータを文字列に変換
str_data = buf.tobytes().decode('utf-8').strip('\x00')

# ChatGPT APIを使って回答を生成
openai.api_key = os.getenv('OPENAI_API_KEY')
response = openai.Completion.create(
    engine="text-davinci-003",
    prompt=str_data,
    max_tokens=60
)

# 生成した回答をバイト形式に変換して共有メモリに書き込み
response_text = response.choices[0].text.strip()
bytes_data = response_text.encode('utf-8')
buf[:len(bytes_data)] = bytes(bytes_data)

# C#から共有メモリにアクセスするための時間に余裕を持たせるための遅延処理
time.sleep(600) # 10分

# 共有メモリをクローズ
shm.close()
