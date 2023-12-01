# -*- coding: utf-8 -*-
import openai
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

PROMPT = """
你是一個精通於繁體中文的研究報告撰寫助手，協助我撰寫水位/濁度AI模型優化與驗證的報告。
關於我的研究計畫，有以下幾個要點：
1. 這是一個防汛相關的地理資訊系統(GIS)，以下簡稱本系統，主要的客戶是臺北水源特定區管理分署，可以從圖台上看到雨量站、水位站及濁度站的觀測數值與AI預報數值。
2. 本系統除了水位濁度觀測與預測以外，還介接了許多外部API，可於本系統中便利的觀測氣象、降雨資訊。本系統也開發許多與防汛有關的工具，例如傳真通報發送工具、通訊錄、查看CCTV影像、彙整報表等等。
3. 本系統需要針對水位/濁度AI模型的整體工作框架進行優化，現有的AI模型是以之前3-5年的水位、濁度觀測數據來進行LSTM模型的訓練。去年計畫以驗證有不錯的成效。
4. 本案今年將導入機器學習生命週期的概念來優化整體AI預報模式。
5. 本案今年將導入MLFlow來做為未來AI模型優化的工具。
"""

filecontent = """
在機器學習生命週期的階段中，本年度聚焦在運行中的模型之成效監控，以實務運用的成效分析，驗證此模型是否達到本案業務目標(是否達到business goal)，並以此監控分析來調整運行中的模型監控項目、模型開發(包含模型調校fine tunine)甚至未來的資料收集與資料預處理流程。

1. 請為我詳細介紹何謂機器機器學習生命週期（Machine Learning Lifecycle）？
包含那些階段？各階段如何設計？是否有標準化的生命週期架構？

2. 以本案水位/濁度預報的LSTM模型而言，實務上可以如何規劃這樣的機器學習生命週期？
這部份請幫我詳細介紹各個階段的具體內容與方法，主標題須包含中文及英文，共提供約1000至1500字篇幅之論述與報告書內容。
"""

def shorten_content(filecontent: str, model_name, filename: str):
    messages = [
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": f"{filecontent}"},
    ]

    print("processing....")
    resp = openai.ChatCompletion.create(
        model = model_name,
        messages = messages,
        temperature = 0.8
    )
    output = resp["choices"][0]["message"]["content"]
    print(output)
    usage = resp["usage"]
    print(usage)
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"{output} \n\n [system PROMPT]\n{PROMPT} \n\n [user prompt]\n{filecontent}\n\n [Token and usage]\n{usage}")


if __name__ == "__main__":
    shorten_content(filecontent, "gpt-4", "output-1.txt")
