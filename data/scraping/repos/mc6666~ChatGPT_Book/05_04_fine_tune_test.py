# 載入套件
import openai
    
if len(sys.argv) < 3:
    print("執行方式：python 05_04_fine_tune_test.py <prompt> <model>")
    exit()

# 呼叫 API    
prompt = sys.argv[1] + '->'
response = openai.Completion.create(
    model=sys.argv[2],
    prompt=prompt,
    temperature=0,
    max_tokens=1,
)

# 顯示回答
print(response.choices[0].text.strip())