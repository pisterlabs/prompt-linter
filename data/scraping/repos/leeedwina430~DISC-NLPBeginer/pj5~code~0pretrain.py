#%%
import os
import openai
# from api_key import APIKEY
# openai.api_key = APIKEY

openai.api_key = os.environ["OPENAI_API_KEY"]

openai.Model.list()

#%%
import json

word2id_path = '.\\StepGame\\Code\\babi_format\\'
with open(os.path.join(word2id_path, 'word2id.json'), 'r') as f_word2id:
    word2id = json.load(f_word2id)

#%%
import json
import pandas as pd

classes = ["left", "right", "above", "below", "upper-left", 
           "upper-right", "lower-left", "lower-right", "overlap"]

train_data_path = ".\\StepGame\\Dataset\\TrainVersion\\train.json"

dir = ".\\StepGame\\Dataset\\TrainVersion\\"
data_paths = [
    "train.json",
    "qa1_test.json", "qa2_test.json", "qa3_test.json", 
    "qa4_test.json", "qa5_test.json", "qa6_test.json","qa7_test.json",
    "qa8_test.json","qa9_test.json", "qa10_test.json",

    "qa1_valid.json","qa2_valid.json","qa3_valid.json",
    "qa4_valid.json","qa5_valid.json",
]

#%%
# preprocess
test_text, test_label, val_text, val_label = [], [], [], []
for i, file_name in enumerate(data_paths):
    data_path = os.path.join(dir, file_name)

    with open(data_path, 'r') as f_data:
        data = json.load(f_data)

    texts, labels = [], []
    if i == 0:
        for k in range(0, 50000, 10000):
            for j in range(0, 200):
                ind = k+j
                texts.append('Story: '+' '.join(data[str(ind)]['story'])+ '\nQuestion: ' + data[str(ind)]['question'])
                # labels.append(classes.index(data[str(ind)]['label']))
                labels.append(data[str(ind)]['label'])
    elif i < 11:
        for i in range(0, 10000, 1000):
            for j in range(0, 4):
                ind = i+j
                texts.append('Story: '+' '.join(data[str(ind)]['story'])+ '\nQuestion: ' + data[str(ind)]['question'])
                # labels.append(classes.index(data[str(ind)]['label']))
                labels.append(data[str(ind)]['label'])
                test_text.append('Story: '+' '.join(data[str(ind)]['story'])+ '\nQuestion: ' + data[str(ind)]['question'])
                test_label.append(classes.index(data[str(ind)]['label']))

    else:
        for i in range(0, 1000, 100):
            for j in range(0, 4):
                ind = i+j
                texts.append('Story: '+' '.join(data[str(ind)]['story'])+ '\nQuestion: ' + data[str(ind)]['question'])
                # labels.append(classes.index(data[str(ind)]['label']))
                labels.append(data[str(ind)]['label'])
                val_text.append('Story: '+' '.join(data[str(ind)]['story'])+ '\nQuestion: ' + data[str(ind)]['question'])
                val_label.append(classes.index(data[str(ind)]['label']))

    df = pd.DataFrame(zip(texts, labels), columns = ['prompt','completion']) 
    df.head()
    # df.to_json('string_'+file_name, orient='records', lines=True)


#%%
results = pd.read_csv('result.csv')
results[results['classification/accuracy'].notnull()].tail(1)

#%%
results[results['classification/accuracy'].notnull()]['classification/accuracy'].plot()

#%%
test = pd.read_json('qa1_test.json', lines=True)
test.head()

prediction = []
for i in range(len(test)):
    ft_model = 'ada:ft-personal-2023-05-03-14-17-01'
    res = openai.Completion.create(model=ft_model, prompt=test['prompt'][i], max_tokens=1, temperature=0)
    # print(test['prompt'][i], test['completion'][i])
    prediction.append(int(res['choices'][0]['text'].strip()))
    # labels[int(res['choices'][0]['text'].strip())]

acc = (sum(prediction==test['completion'].values))/len(prediction)
# acc = 0.225

pd.DataFrame(prediction).to_csv('ada_ft-personal-2023-05-03-14-17-01_qa1_test.csv')



#%%
# train data
import pandas as pd

train = pd.read_json('string_train_edit.json', lines=True)

TRAIN_PROMPT = '\n'
for i in range(len(train)):
    TRAIN_PROMPT += '\n'+ f'{i}.' + train['prompt'][i] +' '+ train['completion'][i] + '\n'

#%%
# test data
test = pd.read_json('string_qa1_test.json', lines=True)

TEST_PROMPT = '\n'

for i in range(len(train)):
    TRAIN_PROMPT += '\n'+ f'{i}.' + train['prompt'][i] +' '+ train['completion'][i] + '\n'


for i in range(len(test)):
    # TEST_PROMPT += '\n'+ f'{i+1}.' + test['prompt'][i]
    # print('\n'+ f'{i+len(train)}.' + test['prompt'][i] + '\n')
    # print('\n'+ f'{i+49}.' + test['prompt'][i])
    print(f'{i}.' + test['completion'][i])
    # TEST_PROMPT += '\n'+ f'{i+1}.' + test['prompt'][i]

#%%
# test data
import time
import pandas as pd

TEMPRATURE = 0.0
MAX_TOKENS = 5  # upper-right 是3个token
TOP_P = 1
FREQUENCY_PENALTY = 0.0 
PRESENCE_PENALTY = 0.0
STOP = ["###"]

for k in range(5, 11):
    # k = 4
    test = pd.read_json(f'string_qa{k}_test.json', lines=True)

    INSTRUCTION = 'Answer each of the following questions with exactly a word or a compound modifier, either left, right, upper-left, upper-right, lower-left, lower-right, above, below, or overlap.'
    # TEST_PROMPT = 'Answer each of the following questions with exactly a word or a compound modifier, either left, right, upper-left, upper-right, lower-left, lower-right, above, below, or overlap.'


    reals, predictions = [], []
    for i in range(len(test)):
        # TEST_PROMPT += '\n'+ f'{i+1}.' + test['prompt'][i]
        TEST_PROMPT = INSTRUCTION + '\n' + test['prompt'][i]
        reals.append(test['completion'][i])

        response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
                {"role": "user", "content": TEST_PROMPT},
            ],
        temperature=TEMPRATURE,
        max_tokens=MAX_TOKENS,
        top_p=TOP_P,
        frequency_penalty=FREQUENCY_PENALTY,
        presence_penalty=PRESENCE_PENALTY,
        stop=STOP
        )
        time.sleep(20)

        predictions.append(response['choices'][0]['message']['content'])


    new_predictions = pd.concat([pd.DataFrame(predictions), test[['completion']]], axis=1)
    pd.DataFrame(new_predictions).to_csv(f'gpt35_qa{k}_len1.csv')


#%%
pd.DataFrame(predictions).to_csv(f'gpt35_qa{k}_len1.csv')

#%%

predictions = pd.read_csv('gpt35_qa1_len1.csv', index_col=0)
new_predictions = pd.concat([predictions, test[['completion']]], axis=1)
pd.DataFrame(new_predictions).to_csv('gpt35_qa1_len1.csv')

#%%

TEST_PROMPT = "Answer the following question using either left, right, above, below, upper-left, upper-right, lower-left, lower-right, or overlap.\nStory: X is to the left of K and is on the same horizontal plane.\nQuestion: What is the relation of the agent X to the agent K?"

#%%
# gpt-3.5-turbo

TEMPRATURE = 0.0
MAX_TOKENS = 1000  # upper-right 是3个token
TOP_P = 1
FREQUENCY_PENALTY = 0.0 
PRESENCE_PENALTY = 0.0
STOP = ["###"]


response = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
        {"role": "user", "content": TEST_PROMPT},
    ],
  temperature=TEMPRATURE,
  max_tokens=MAX_TOKENS,
  top_p=TOP_P,
  frequency_penalty=FREQUENCY_PENALTY,
  presence_penalty=PRESENCE_PENALTY,
  stop=STOP
)
print(response['choices'][0]['message']['content'])

    
# response = openai.Completion.create(
#   model="curie",
#   prompt= TEST_PROMPT,
#   temperature=TEMPRATURE,
#   max_tokens=MAX_TOKENS,
#   top_p=TOP_P,
#   frequency_penalty=FREQUENCY_PENALTY,
#   presence_penalty=PRESENCE_PENALTY,
#   stop=STOP
# )

# print(response['choices'][0]['text'])
