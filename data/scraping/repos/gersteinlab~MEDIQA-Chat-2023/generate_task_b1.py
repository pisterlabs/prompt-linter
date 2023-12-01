import pickle
import sys
import csv
import openai
openai.api_key = 'YOUR_API_KEY'
from nltk.tokenize import word_tokenize

dialog_dict = {}
with open(sys.argv[1]) as input_file:
    reader = csv.DictReader(input_file)
    for row in reader:
        dialog_dict[row['encounter_id']] = row['dialogue']


def shorten_dialog(dialog_str, max_token_length):
    utterances = dialog_str.split('\n')
    good_utterances = []
    total_length = 0
    for utterance in utterances:
        utterance_length = len(word_tokenize(utterance))
        if total_length + utterance_length > max_token_length:
            break
        total_length += utterance_length
        good_utterances.append(utterance)
    return '\n'.join(good_utterances)


def predict_note(dialog_str):
    dialog_str = shorten_dialog(dialog_str, 1200)
    try:
        response = openai.Completion.create(
            model='davinci:ft-personal-2023-03-23-05-58-11',
            prompt=dialog_str + '\n\n###\n\n',
            max_tokens=800,
            temperature=0.2,
            stop=["ENDNOTE"]
        )
    except:
        try:
            response = openai.Completion.create(
                model='davinci:ft-personal-2023-03-23-05-58-11',
                prompt=dialog_str + '\n\n###\n\n',
                max_tokens=600,
                temperature=0.2,
                stop=["ENDNOTE"]
            )
        except:
            return ""
    print(response['choices'])
    text = response['choices'][0]['text']
    return text

predicted_notes = {}
for ID, dialog in dialog_dict.items():
    category = predict_note(dialog)
    predicted_notes[ID] = category
    print(f"{ID} => {category}")

try:
    with open('predicted_notes', 'w') as file:
        pickle.dump(predicted_notes, file)
    print('pickled.')
except:
    pass

with open('taskB_gersteinlab_run1.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerow(['TestID', 'SystemOutput'])
    for ID, dialog in dialog_dict.items():
        writer.writerow([ID, predicted_notes[ID]])

print('Done.')
