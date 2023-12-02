from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
from langchain.text_splitter import CharacterTextSplitter

model_name = 'bert-large-uncased-whole-word-masking-finetuned-squad'
# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
model_name = 'models/bert-large-uncased-float16.bin'
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
# 轉換為半精度
# model.half()

# 儲存模型
# model.save_pretrained('models/bert-large-uncased-float16.bin')

filepath = 'data/vue3.txt'
with open(filepath, 'r', encoding='utf-8') as f:
    text = f.read()
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=200)
docs = text_splitter.split_text(text)
print(f'{docs=}')

question = "how to create vue3 app?"

# 使用 BERT 模型獲取嵌入向量
inputs = tokenizer(question, truncation=True, padding=True, return_tensors='pt')
outputs = model(**inputs)

# 從模型輸出中獲取答案
start_scores = outputs.start_logits
end_scores = outputs.end_logits

# 解碼答案
all_tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
answer = tokenizer.convert_tokens_to_string(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores) + 1])
print(f"Answer: '{answer}'")
