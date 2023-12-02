import os
import openai

## GPT 3.5 & 4 models


# text_improvement_system_prompt = "you're an assistant with expertise in understanding, differentiating on legal text & content. \
#         you'll be given a legal draft text which has some irregularities like maybe a word is incomplete & remaining part is in \
#         next line like so: 'informati \n on', fix spellings or similar. Fix that, remove unnecessary new lines to make it compact,\
#         considering to reduce the number of tokens & strictly only output the same legal text back but after improving it as asked."

# with open('../data/legal_drafts_text/GST/Challan/PMT-2.txt') as infile:
#     trial_prompt = infile.read()
#     print(trial_prompt)

# completion1 = openai.ChatCompletion.create(
#   model="gpt-4",
#   messages= [
#     {"role": "system", "content": text_improvement_system_prompt},
#     {"role": "user", "content": trial_prompt}
#   ]
# )
# improved_text = completion1.choices[0].message['content']
# print(improved_text)

# system_prompt = "you are a world class machine learning data engineer with expertise in understanding of laws, legal documents and legal language. \
#                 You can understand types of cases, entities in legal drafts & more. You will get a legal draft text as input & \
#                 you would only reply with an 'outprompt'. note: only reply with the outprompt & nothing except it like 'certainly', 'sure, here is your outline' or anything similar. \
#                 Here's a rough outline of outprompt:\n \
#                 type: draft/document (etc based on the draft text)\n \
#                 category: civil/criminal/business (etc based on the draft text)\n \
#                 subcategory: company registration under ...(etc based on the draft text)\n \
#                 jurisdiction: applicable state or federal law (etc based on the draft text)\n \
#                 parties involved: description of the main parties\n \
#                 context: a short summary or detailed description of the purpose of the draft/document"


# completion = openai.ChatCompletion.create(
#   model="gpt-3.5-turbo",
#   messages= [
#     {"role": "system", "content": system_prompt},
#     {"role": "user", "content": improved_text}
#   ]
# )

# print(completion.choices[0].message['content'])

# -------------------------------------------------------------------------------------------------------------------------

## Legal-BERT

from transformers import pipeline, AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-small-uncased")
model = AutoModel.from_pretrained("nlpaueb/legal-bert-small-uncased")

# inputs = tokenizer("Hello world!", return_tensors="pt")
# outputs = model(**inputs)

generator = pipeline("fill-mask", model="nlpaueb/legal-bert-small-uncased", tokenizer="nlpaueb/legal-bert-small-uncased")

print(generator(f"Legal proceedings in court involves {tokenizer.mask_token}"))
