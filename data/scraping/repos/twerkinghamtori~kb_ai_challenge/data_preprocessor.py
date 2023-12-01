# 라이브러리 호출 파트
import pandas as pd
import numpy as np
import openai
from tqdm import tqdm
import json
import csv

#################################입력해야 실행 가능####################################
json_file_path = "<sample.json_path>"
csv_file = "<feature_data_path>"
######################################################################################

# json 파일 로드
with open(json_file_path, "r", encoding="utf-8") as json_file:
    data = json.load(json_file)

# 생성 데이터 전처리
frame = {}
for index in range(len(data)):
    tray=[]
    a=data[f"{index}"].split("2.")
    b=a[-1].split("3.")
    c=b[-1].split("4.")
    d=c[-1].split("5.")
    tray.append(a[0][3:].replace("\n",""))
    tray.append(b[0].replace("\n",""))
    tray.append(c[0].replace("\n",""))
    tray.append(d[0].replace("\n",""))
    tray.append(d[-1].replace("\n",""))
    for sample in tray:
        frame[sample]=index

# 전처리된 데이터 저장
with open(csv_file, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(['대상', '상품'])
    for category, value in frame.items():
        writer.writerow([category, value])
        
