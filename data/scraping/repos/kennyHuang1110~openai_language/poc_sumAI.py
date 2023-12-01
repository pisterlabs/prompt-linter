import os
import openai
from dotenv import load_dotenv

openai.api_key = os.getenv("OPENAI_API_KEY")

# 讀取 CSV 檔案並將文章存儲為列表
def read_articles_from_csv(file_path):
    articles = []
    with open(file_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            # 假設 CSV 檔案中每一行是一篇文章，如果不是，請適當調整程式碼以讀取正確的欄位
            article = row[0]  # 假設文章在第一個欄位，如果不是，請根據實際情況修改索引
            articles.append(article)
    return articles

# CSV 檔案路徑
csv_file_path = r"C:\Users\USER\Desktop\選舉\fine tuning\p.txt"    

# 讀取 CSV 檔案中的文章
articles = read_articles_from_csv(csv_file_path)


# 讀取 CSV 檔案中的文章
articles = read_articles_from_csv(csv_file_path)

# 將文章合併成一個字串，每篇文章以換行符號連接
articles_combined = "\n".join(articles)

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "您將獲得這些文章，您的任務是對這些文章進行如下總結:討論的總體摘要，讓候選人知道社會大眾對於該議題的想法"},
        {"role": "user", "content": articles_combined},
    ]
)

summary = response['choices'][0]['message']['content'].strip()

print("總結摘要：\n" + summary + "\n")
with open("summary.txt", "w", encoding="utf-8") as f:
    f.write(summary)
