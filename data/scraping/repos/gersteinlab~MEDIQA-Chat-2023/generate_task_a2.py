import csv
import sys
from transformers import AutoTokenizer, AutoModel, BartForConditionalGeneration, PreTrainedTokenizerFast
import torch
import openai
import re
openai.api_key = 'YOUR_API_KEY'

dialog_dict = {}
with open(sys.argv[1]) as input_file:
    reader = csv.DictReader(input_file)
    for row in reader:
        dialog_dict[row['ID']] = row['dialogue']

model_path = 'checkpoint-7776'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = BartForConditionalGeneration.from_pretrained(model_path).to('cuda')


class RobertaClass(torch.nn.Module):
    def __init__(self):
        super(RobertaClass, self).__init__()
        self.l1 = AutoModel.from_pretrained('./RoBERTa-base-PM-M3-Voc-distill-align-hf/')
        # self.d1 = torch.nn.Dropout(dropout_rate)
        # self.l1 = torch.nn.Linear(768, 64)
        # self.bn1 = torch.nn.LayerNorm(64)
        # self.d2 = torch.nn.Dropout(dropout_rate)
        # self.l2 = torch.nn.Linear(64, 2)
        self.pre_classifier = torch.nn.Linear(768, 32) # Original 768
        self.norm = torch.nn.LayerNorm(32)
        self.dropout1 = torch.nn.Dropout(0.7) # Original 0.3
        self.dropout2 = torch.nn.Dropout(0.5)
        self.classifier = torch.nn.Linear(32, 20) # Original 768, 20

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.dropout1(pooler)
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler) # RELU -> Tanh
        pooler = self.norm(pooler)
        pooler = self.dropout2(pooler)
        output = self.classifier(pooler)
        return output


classification_model = RobertaClass()
classification_model.load_state_dict(torch.load('final-roberta-classifier'))
classification_model = classification_model.to('cuda')

CATEGORIES = ['GENHX', 'MEDICATIONS', 'CC', 'PASTMEDICALHX', 'ALLERGY', 'FAM/SOCHX',
 'PASTSURGICAL', 'OTHER_HISTORY', 'ASSESSMENT', 'ROS', 'DISPOSITION', 'EXAM',
 'PLAN', 'DIAGNOSIS', 'EDCOURSE', 'IMMUNIZATIONS', 'LABS', 'IMAGING',
 'PROCEDURES', 'GYNHX']

classification_tokenizer = AutoTokenizer.from_pretrained('./RoBERTa-base-PM-M3-Voc-distill-align-hf/')


def predict_category(dialog):
    response = openai.Completion.create(
        model='davinci:ft-personal-2023-03-15-08-22-14',
        prompt=dialog + '\n\n###\n\n',
        temperature=0
    )
    text = response['choices'][0]['text']
    print(f'Text: {text}')
    try:
        idx = int(re.search(r'\d+', text).group())
        return CATEGORIES[idx]
    except:
        print('Failed openai inference..')
        return CATEGORIES[0]


predicted_categories = {}
for ID, dialog in dialog_dict.items():
    category = predict_category(dialog)
    predicted_categories[ID] = category
    print(f"{ID} => {category}")


def predict_summary(dialog_str):
    global model, tokenizer

    inputs_dict = tokenizer(
        dialog_str, max_length=1024, padding="max_length", truncation=True, return_tensors="pt"
    )
    input_ids = inputs_dict.input_ids.to("cuda")
    attention_mask = inputs_dict.attention_mask.to("cuda")

    max_length = 512
    output_ids = model.generate(input_ids, attention_mask=attention_mask, max_length=max_length, num_beams=6)

    prediction = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return prediction


predicted_summaries = {}
for ID, dialog in dialog_dict.items():
    summary = predict_summary(dialog)
    predicted_summaries[ID] = summary
    print(f"{ID} => {summary}")

with open('taskA_gersteinlab_run2.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerow(['TestID', 'SystemOutput1', 'SystemOutput2'])
    for ID, dialog in dialog_dict.items():
        writer.writerow([ID, predicted_categories[ID], predicted_summaries[ID]])

print('Done.')
