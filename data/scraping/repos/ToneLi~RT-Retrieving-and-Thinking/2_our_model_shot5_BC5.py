import os
from tqdm import tqdm
from base_access import AccessBase
from logger import get_logger
import json
import argparse
from dataset_name import FULL_DATA
import random
import openai
from torch.utils.data import DataLoader

random.seed(1)
logger = get_logger(__name__)

openai.api_key = "key"

# if torch.cuda.is_available():
#     device = torch.device("cuda")
#     print(torch.cuda.device_count())
#     print('Available:', torch.cuda.get_device_name(0))
# else:
#     print('No GPU available, using the CPU instead.')
#     device = torch.device("cpu")

SYSTEM_PROMPT = "An entity is Chemical, Disease. Sports, sporting events, adjectives, verbs, numbers, adverbs, abstract concepts, sports, are not entities. Dates, years and times are not entities. Possessive words like I, you, him and me are not entities."
# SYSTEM_PROMPT = "An entity is Chemical or Disease. the Chemical entity is a substance (as an element or compound) obtained from a chemical process or used to get a chemical result." \
#                 "the Disease entity are disease, any harmful deviation from the normal structural or functional state of an organism, generally associated with certain signs and symptoms and differing in nature from physical injury. A diseased organism commonly exhibits signs or symptoms indicative of its abnormal state." \
#                 "Please recognize the Disease entity  and Chemical entity in the sentence"#---92.27


USER_PROMPT_1 = "Are you clear about your role?"
ASSISTANT_PROMPT_1 = "Sure, I'm ready to help you with your NER task. Please provide me with the necessary information to get started."


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--source-dir", type=str, help="directory for the input")
    parser.add_argument("--source-name", type=str, help="file name for the input")
    parser.add_argument("--train-name", type=str, default="None", help="file name for the training set")
    parser.add_argument("--data-name", type=str, help="dataset name for the input")
    parser.add_argument("--example-dir", type=str, default="None", help="directory for the example")
    parser.add_argument("--example-name", type=str, default="None", help="file name for the example")
    parser.add_argument("--example-num", type=int, default=16, help="numebr for examples")
    parser.add_argument("--last-results", type=str, default="None", help="unfinished file")
    parser.add_argument("--write-dir", type=str, help="directory for the output")
    parser.add_argument("--write-name", type=str, help="file name for the output")
    
    return parser


def read_mrc_data(dir_):
    # file_name = os.path.join(dir_, f"conll.mrc-ner.{prefix}")
    label_sentence=[]
    with open(dir_,"r",encoding="utf-8") as fr:
        for line in fr.readlines():
            data_=json.loads(line)
            sentence=" ".join(data_["tokens"])
            for tuple_ in data_["entity"]:
                label_sentence.append([tuple_["type"],tuple_["text"],tuple_["offset"],sentence])

    return label_sentence

def mrc_data(dir_):
    # file_name = os.path.join(dir_, f"conll.mrc-ner.{prefix}")
    Dic_={}
    with open(dir_,"r",encoding="utf-8") as fr:
        for line in fr.readlines():
            data_=json.loads(line)
            sentence=" ".join(data_["tokens"])
            label_sentence=[]
            for tuple_ in data_["entity"]:
                label_sentence.append([tuple_["type"],tuple_["text"]])

            Dic_[sentence]=label_sentence

    return Dic_

def read_results(dir_):
    file = open(dir_, "r")
    resulst = file.readlines()
    file.close()
    return resulst

def read_examples(dir_, prefix="dev"):
    print("reading ...")
    file_name = os.path.join(dir_, f"mrc-ner.{prefix}")
    return json.load(open(file_name, encoding="utf-8"))

def read_idx(dir_):
    print("reading ...")
    example_idx = []
    file = open(dir_, "r")
    for line in file:
        example_idx.append(json.loads(line.strip()))
    file.close()
    return example_idx

def mrc2prompt(mrc_data, data_name="CONLL", example_idx=None, train_mrc_data=None, example_num=16, last_results=None,mrc_train_data_source=None):
    print("mrc2prompt ...")

    def get_example(index):
        """
        [tuple_["type"],tuple_["text"],sentence])
        NCBI: DiseaseClass  SpecificDisease  CompositeMention Modifier
        """
        exampel_prompt = ""
        DIc_ = {}
        if len(example_idx)>0:
            for idx_ in example_idx[index][:example_num]:

                type_ = train_mrc_data[idx_][0]
                label_text = train_mrc_data[idx_][1]
                offset=train_mrc_data[idx_][2]
                context = train_mrc_data[idx_][3]
                if context not in DIc_:
                    DIc_[context]=mrc_train_data_source[context]
                # else:
                #     DIc_[context] =DIc_[context] + [[label_text, type_]]
                # print(DIc_)
                # print("--------")
                count_={"Chemical":0,"Disease":0}
            for key, value in DIc_.items():
                sentence=key

                # break
                sentence_split=sentence.split(" ")
                # print("-sentence", sentence_split)
                context=""
                f=-1
                label_texts=[]
                if count_["Chemical"] < 6 and count_["Disease"] < 6:
                    for label in value:
                        f=f+1
                        type_ = label[0]
                        label_text = label[1]
                        label_texts.append(label_text)

                        if type_ == "Chemical":
                            context=context+str(f)+"."+label_text+"|"+"True|"+"as it is a Chemical "
                            count_["Chemical"]= count_["Chemical"]+1
                            # context = context.replace(label_text, "</C>" + label_text + "<C>")
                        if type_ == "Disease":
                            context = context + str(f) + "." + label_text + "|" + "True|" + "as it is a Disease "
                            count_["Disease"] = count_["Disease"] + 1
                label_texts=" ".join(label_texts).split(" ")
                false_excamples=[]
                for j in range(8):
                    false_excample=random.sample(sentence_split,1)[0]
                    if false_excample not in label_texts and false_excample not in false_excamples:
                        false_excamples.append(false_excample)

                fl=0
                for f_excampl in false_excamples:
                    fl =fl+1
                    context = context + str(f+fl) + "." + f_excampl + "|" + "False|" + "as it is not a Disease or Chemical "


                # print(context)
                exampel_prompt += f"The given sentence: {sentence}\n"
                exampel_prompt += f"The labeled sentence: {context}\n"

            """---must be add------- if not add 92.59   add 93.62"""
            exampel_prompt += f"The given sentence:  Naloxone reverses the antihypertensive effect of clonidine In unanesthetized spontaneously hypertensive rats the decrease in blood pressure and heart rate\n"
            exampel_prompt += f"The labeled sentence: 1. Naloxone|True|as it is a Chemical  2.reverses|False| as it is a verb 3. antihypertensive|False| as it is a drug 4. clonidine|True| as it is a Chemical 5. hypertensive|True|as it is a Disease\n"
            exampel_prompt += f"The given sentence:  [mask1] reverses the antihypertensive effect of [mask2] In unanesthetized spontaneously [mask3] rats the decrease in blood pressure and heart rate\n"
            exampel_prompt += f"The labeled sentence: 1.reverses|False| as it is a verb 2. antihypertensive|False| as it is a drug 3.blood pressure|False| as it is a medical word 4.heart rate|False|as it is not a Disease or Chemical\n"

            exampel_prompt += f"The given sentence: test conditions can impact upon other physiological responses to [mask1] such as drug-induced [mask2]\n"
            exampel_prompt += f"The labeled sentence: 1.drug-induced|False| as it is a adj 2.physiological|False| as it is a medicine word 4.conditions|False|as it is not a Disease or Chemical\n"

            exampel_prompt += f"The given sentence: high protein feeding on [mask1] -induced [mask2] in rats. Rats with lithium -induced [mask3] were subjected to high protein (HP\n"
            exampel_prompt += f"The labeled sentence: 1.high|False| as it is a adjective word 2.high protein|False| as it is a medicine word\n"

        else:
            exampel_prompt += f"The given sentence:none\n"
            exampel_prompt += f"The labeled sentence: no excamples\n"
        return exampel_prompt
        
    results = []

    for item_idx in tqdm(range(len(mrc_data))):

        if last_results is not None and last_results[item_idx].strip() != "FRIDAY-ERROR-ErrorType.unknown":
            continue

        item_ = mrc_data[item_idx]
        # print("item_",item_)
        context = item_[0]
        origin_label = item_[1]
        # for origin_label in origin_labels:
        # transfered_label, sub_prompt = FULL_DATA[data_name][origin_label]
        transfered_label = str(list(FULL_DATA[data_name].keys())).replace("[","").replace("]","")
        # prompt_label_name = transfered_label[0].upper() + transfered_label[1:]
        # prompt = f"I want to extract {transfered_label} entities that {sub_prompt}, and if that does not exist output \"none\". Below are some examples.\n"
        # prompt = f"I want to extract {transfered_label} entities that {sub_prompt}. Below are some examples.\n"
        # prompt = f"You are an excellent linguist. Within the OntoNotes5.0 dataset, the task is to label {transfered_label} entities that {sub_prompt}. Below are some examples, and you should make the same prediction as the examples.\n"
        prompt = f"You are an excellent linguist. Within the BC5CDR dataet, the task is to label {transfered_label} entities that in the given sentence. Below are some examples, and you should make the same prediction as the examples.\n"
        # prompt = f"You are an excellent linguist. The task is to label {transfered_label} entities in the given sentence. {prompt_label_name} entities {sub_prompt}. Noted that if the given sentence does not contain any {transfered_label} entities, just output the same sentence, or surround the extracted entities by @@ and ## if there exist {transfered_label} entities. Below are some examples."
        # prompt = f"You are an excellent linguistic. The task is to label {transfered_label} entities that {sub_prompt}. First, articulate the clues and reasoning process for determining {transfered_label} entities in the sentence. Next, based on the clues and your reasoning process, label {transfered_label} entities in the sentence. Below are some examples.\n"

        # prompt += get_knn(test_sentence=context, nums=example_false, label_name=transfered_label, positive_idx=0)
        # prompt += get_knn(test_sentence=context, nums=example_true, label_name=transfered_label, positive_idx=1)
        prompt += get_example(index=item_idx)
        # print(prompt)
        # break
        prompt += f"The given sentence: {context}\nThe labeled sentence:"
        results.append(prompt)
    
    return results

def ner_access(openai_access, ner_pairs, batch=16):
    print("tagging ...")
    results = []
    start_ = 0
    pbar = tqdm(total=len(ner_pairs))
    while start_ < len(ner_pairs):
        end_ = min(start_+batch, len(ner_pairs))
        # print("ner_pairs[start_:end_]",ner_pairs[start_:end_])
        results = results + openai_access.get_multiple_sample(ner_pairs[start_:end_])
        # print("-results---",results)
        pbar.update(end_-start_)
        start_ = end_
    pbar.close()
    return results

def write_file(labels, dir_, last_name):
    print("writing ...")
    file_name = os.path.join(dir_, last_name)
    file = open(file_name, "w")
    for line in labels:
        file.write(line.strip()+'\n')
    file.close()
    # json.dump(labels, open(file_name, "w"), ensure_ascii=False)

def read_ner_test_data(test_dir):
    label_sentence = []
    with open(test_dir, "r", encoding="utf-8") as fr:
        for line in fr.readlines():
            data_ = json.loads(line)
            sentence = " ".join(data_["tokens"])
            L=[]
            for tuple_ in data_["entity"]:
                L.append([tuple_["type"], tuple_["text"]])
            label_sentence.append([sentence,L])

    return label_sentence
class Bertbiofewreader():
    def __init__(self, file=None):
        # super().__init__()
        # self.tokenizer = BertTokenizer.from_pretrained(pretrainedfile)
        if file is not None:
            self.init(file)

    def getinstance(self, data):
        text = ' '.join(data['tokens'])
        labels = ["O"] * len(data['tokens'])
        # print("self.label2id",self.label2id)

        entitys = []

        for entity in data['entity']:
            if entity['type'] in self.target_classes:
                entitys.append(entity)
                for i in range(entity['offset'][0], entity['offset'][1]):
                    if i == entity['offset'][0]:
                        label = 'I-' + entity['type']
                    else:
                        label = 'I-' + entity['type']
                    labels[i] = label
        field = {
            'inputtext': data['tokens'],
            'labels': labels,
        }
        return field

    def init(self, file, labels=None):
        self.dataset = []
        self.labels = []
        with open(file) as f:
            for line in f:
                line = json.loads(line)
                # print(line.keys())
                class_count = {}
                for entity in line['entity']:
                    if entity['type'] not in self.labels:
                        self.labels.append(entity['type'])
                    if entity['type'] not in class_count:
                        class_count[entity['type']] = 1
                    else:
                        class_count[entity['type']] += 1
                line['class_count'] = class_count
                self.dataset.append(line)
        if labels is not None:
            self.labels = labels
        self.target_classes = self.labels

    def buildlabel2id(self):
        self.label2id = {}
        # print(" self.target_classes", self.target_classes)
        for label in self.target_classes:
            if label != 'O':
                # self.label2id['B-' + label] = len(self.label2id)
                self.label2id['I-' + label] = len(self.label2id)
            else:
                self.label2id[label] = len(self.label2id)

    def text_to_instance(self, idx=None, label=True, dataset=None, target_classes=None):
        results = []
        if dataset is not None:
            self.dataset = dataset
        # print("target_classes",target_classes)
        if target_classes is not None:
            # print("------------------fucccccccccccc")
            self.target_classes = target_classes
            self.buildlabel2id()

        if idx is None:
            for data in self.dataset:
                results.append(self.getinstance(data))
        else:
            for index in idx:
                results.append(self.getinstance(self.dataset[index]))
        if idx is None:
            idx = list(range(len(self.dataset)))
        return results, self.label2id

    def _read(self, file):
        with open(file) as f:
            for line in f:
                yield self.text_to_instance(json.loads(line))



def openai_chat_completion_response(final_prompt):
    response = openai.ChatCompletion.create(
        timeout=600,
        max_tokens=390,
        model="gpt-4",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_1},
            {"role": "assistant", "content": ASSISTANT_PROMPT_1},
            {"role": "user", "content": final_prompt}
        ]
    )

    return response['choices'][0]['message']['content'].strip(" \n")


def get_results(testreader, file, batch_size,out_putfile,prompts):
    fw = open(out_putfile, "w", encoding="utf-8")
    with open(file) as f:
        support_number = -1
        for line in f:
            support_number += 1
            dataset = json.loads(line)
            support = dataset['support']
            target_classes_ = dataset['target_label']
            target_classes = []
            for labe in target_classes_:
                if labe != "O":
                    target_classes.append(labe)
            constraint = 'bio_decay'
            target_classes.append('O')
            if 'bio' in constraint:
                id2label = []
                for label in target_classes:
                    if label != 'O':
                        # id2label.append('B-' + label)
                        id2label.append('I-' + label)
                    else:
                        id2label.append(label)
            else:
                id2label = target_classes

            query_set, _ = testreader.text_to_instance(None, target_classes=target_classes)
            query_data_loader = DataLoader(query_set, batch_size, shuffle=False)

            flag = 0
            ps = 0
            rs = 0
            f1s = 0
            i=-1
            for batch in tqdm(query_data_loader):
                flag = flag + 1
                print(flag)
                i=i+1
                b_input_text = [word[0] for word in batch["inputtext"]]
                b_labels = [word[0].replace("I-", "") for word in batch["labels"]]

                if flag >=0:
                    b_input_text = " ".join(b_input_text)
                    GUIDELINES_PROMPT = prompts[i]  # .format(sentence)
                    Dic_ = {}
                    predictions = openai_chat_completion_response(GUIDELINES_PROMPT)
                    # print(predictions)
                    Pre = predictions.replace("'", "\"")
                    Dic_["sentence"] = str(b_input_text)
                    Dic_["pre_labels"] = Pre
                    Dic_["gold_labels"] = " ".join(b_labels)
                    json_str = json.dumps(Dic_)
                    fw.write(json_str)
                    fw.write("\n")
                    fw.flush()
            break





if __name__ == '__main__':
    # test()

    parser = get_parser()
    args = parser.parse_args()

    mrc_train = read_mrc_data(dir_="data/BCD5/dev_CDR.json")
    mrc_train_data_source = mrc_data(dir_="data/BCD5/dev_CDR.json")

    ner_test = read_ner_test_data("data/BCD5/sampled_100_test_BCDR.json")
    # ner_test = read_mrc_data(args.source_dir, prefix=args.source_name)
    # mrc_train = read_mrc_data(dir_=args.source_dir, prefix=args.train_name)
    example_idx = read_idx("data/BCD5/5shot_test.jsonl")

    last_results = None
    # if args.last_results != "None":
    #     last_results = read_results(dir_=args.last_results)
    data_name="BCD5"
    prompts = mrc2prompt(mrc_data=ner_test, data_name=data_name, example_idx=example_idx, train_mrc_data=mrc_train, example_num=args.example_num, last_results=last_results,mrc_train_data_source=mrc_train_data_source)


    inputfile = "1shot_CDR.json"
    out_putfile = "data/BCD5/our_method_shot5_output_just_right.json"
    testreader = Bertbiofewreader('data/BCD5/sampled_100_test_BCDR.json')
    batch_size = 1
    get_results(testreader, inputfile, batch_size, out_putfile,prompts)


"""
BCD  shot1---the final value is 0.9350361398760133
"""
