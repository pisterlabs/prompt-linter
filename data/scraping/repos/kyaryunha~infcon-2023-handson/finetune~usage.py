import openai

from api_key import api_key
openai.api_key = api_key

print(openai.File.create(file=open("data.jsonl", "rb"), purpose="fine-tune"))  # 파일 업로드

print(openai.File.list())  # 파일 리스트

print(openai.File.delete("file-abcdef"))  # 파일 삭제

print(openai.File.retrieve("file-abcdef"))  # 특정 파일 정보 보기

a = openai.File.download("file-abcdef")  # 파일 다운로드 하기

with open("download.jsonl", "wb") as f:
    f.write(a)

filelist = openai.File.list()   # 업로드한 전체 파일 삭제하기
for file in filelist["data"]:
    openai.File.delete(file["id"])
