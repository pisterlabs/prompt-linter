"""
if we know the entites in a sentence, but we do not know the labels, the gpt4 can do it?
"""

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
import re
random.seed(1)
logger = get_logger(__name__)

openai.api_key = "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"

SYSTEM_PROMPT = "you are an excellent linguist. The task is to verify whether the word is an DiseaseClass, SpecificDisease, CompositeMention  or Modifier entity extracted from the given sentence."
USER_PROMPT_1 = "Are you clear about your role?"
ASSISTANT_PROMPT_1 = "Sure, I'm ready to help you with your this task. Please provide me with the necessary information to get started."


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




def read_gpt_out_data(dir_):
    # file_name = os.path.join(dir_, f"conll.mrc-ner.{prefix}")
    all_entities=[]
    sentences=[]
    with open(dir_, "r", encoding="utf-8") as fr:
        for line in fr.readlines():
            entity_sentence=[]
            # flag = flag + 1
            line = json.loads(line.strip())
            sentence = line["sentence"]
            sentences.append(sentence)
            pre_labels = line["pre_labels"]
            # SpecificDisease
            if '</S>' in pre_labels:
                s = r'</S>(.*?)<S>'
                s_entity = re.findall(s, pre_labels)
                # print(s_entity)
                for en in s_entity:
                    end_ = []
                    en_ = en.split(" ")
                    for e in en_:
                        end_.append(e + "|" + "SpecificDisease")
                    end_ = " ".join(end_)
                    # print(end_)
                    entity_sentence.append(end_)

            if '</D>' in pre_labels:
                # DiseaseClass
                d = r'</D>(.*?)<D>'
                d_entity = re.findall(d, pre_labels)

                for dn in d_entity:
                    end_d = []
                    dn_ = dn.split(" ")
                    for d in dn_:
                        end_d.append(d + "|" + "DiseaseClass")
                    end_d = " ".join(end_d)
                    entity_sentence.append(end_d)


            if '</M>' in pre_labels:
                # Modifier
                m = r'</M>(.*?)<M>'
                m_entity = re.findall(m, pre_labels)

                for mn in m_entity:
                    end_m = []
                    mn_ = mn.split(" ")
                    for m in mn_:
                        end_m.append(m + "|" + "Modifier")
                    end_m = " ".join(end_m)
                    entity_sentence.append(end_m)

            if '</C>' in pre_labels:
                # CompositeMention
                c = r'</C>(.*?)<C>'
                c_entity = re.findall(c, pre_labels)

                for cn in c_entity:
                    end_c = []
                    cn_ = cn.split(" ")
                    for c in cn_:
                        end_c.append(c + "|" + "CompositeMention")
                    end_c = " ".join(end_c)
                    # print(end_c)
                    entity_sentence.append(end_c)

            all_entities.append(entity_sentence)
    return all_entities,sentences

def get_gold_labels(dir_):
    # file_name = os.path.join(dir_, f"conll.mrc-ner.{prefix}")
    L=[]

    with open(dir_, "r", encoding="utf-8") as fr:
        for line in fr.readlines():
            entity_sentence=[]
            # flag = flag + 1
            line = json.loads(line.strip())
            sentence = line["sentence"]
            sentences.append(sentence)
            pre_labels = line["gold_labels"]
            L.append(pre_labels)
    return L


 # Disease  Chemical

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


def mrc2prompt(ner_test,gpt_results,transfered_label,sentences):
    print("mrc2prompt ...")

    prompts = []
    for item_idx in tqdm(range(len(ner_test))):
        one_sentence = []

            # prompt = f"{context}\nIs the word \"{entity}\" in the former sentence an {transfered_label} entity? Please answer with yes or no."
            # for entity in entities:
        sentence=ner_test[item_idx][0]
        # print(sentence)
        entties=ner_test[item_idx][1]
        # print(entties)
        enti=[]
        label=[]
        for el in entties:
            type_=el[0]
            entity_text=el[1]
            prompt = f"You are an excellent linguist. The task is to verify whether the word is an {transfered_label} entity extracted from the given sentence..\n\n"
            prompt += f"The given sentence: {sentence}\nIs the label of word \"{entity_text}\" indicating the DiseaseClass, SpecificDisease, CompositeMention or Modifier? Please answer with DiseaseClass, SpecificDisease, CompositeMention or Modifier.\n"
            one_sentence.append(prompt)

        prompts.append(one_sentence)

    return prompts


def ner_access(openai_access, ner_pairs, batch=16):
    print("tagging ...")
    results = []
    start_ = 0
    pbar = tqdm(total=len(ner_pairs))
    while start_ < len(ner_pairs):
        end_ = min(start_ + batch, len(ner_pairs))
        # print("ner_pairs[start_:end_]",ner_pairs[start_:end_])
        results = results + openai_access.get_multiple_sample(ner_pairs[start_:end_])
        # print("-results---",results)
        pbar.update(end_ - start_)
        start_ = end_
    pbar.close()
    return results


def write_file(labels, dir_, last_name):
    print("writing ...")
    file_name = os.path.join(dir_, last_name)
    file = open(file_name, "w")
    for line in labels:
        file.write(line.strip() + '\n')
    file.close()
    # json.dump(labels, open(file_name, "w"), ensure_ascii=False)


def read_ner_test_data(test_dir):
    label_sentence = []
    with open(test_dir, "r", encoding="utf-8") as fr:
        for line in fr.readlines():
            data_ = json.loads(line)
            sentence = " ".join(data_["tokens"])
            L = []
            for tuple_ in data_["entity"]:
                L.append([tuple_["type"], tuple_["text"]])
            label_sentence.append([sentence, L])

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


def get_results( file, out_putfile, prompts,gpt_results,gold_l):
    fw = open(out_putfile, "w", encoding="utf-8")
    flag=-1
    i=-1
    with open(file) as f:
        support_number = -1
        for line in f:
            flag=flag+1
            print(flag)

            i=i+1
            support_number += 1
            dataset = json.loads(line)
            sentence = " ".join(dataset["tokens"])
            pre_labels=sentence

            entity_words = []
            for tuple_ in dataset["entity"]:
                 entity_words.append(tuple_["text"])

            if flag >= 0:
                GUIDELINES_PROMPTs = prompts[i]  # .format(sentence)
               # range(len(ner_test))
               #  print("GUIDELINES_PROMPTs",len(GUIDELINES_PROMPTs))
               #  print("entity_words",len(entity_words))

                words_flag=[]
                for j in range(len(GUIDELINES_PROMPTs)):
                    entity_word=entity_words[j]
                    if entity_word not in words_flag:
                        predictions = openai_chat_completion_response(GUIDELINES_PROMPTs[j])
                        # print("ff",entity_words)
                        print(predictions) #Disease  Chemical
                        if "DiseaseClass" in predictions  or  "diseaseclass"  in predictions:
                            pre_labels = pre_labels.replace(entity_word,"</D>" + entity_word + "<D>")
                        elif "SpecificDisease" in predictions  or  "specificdisease"  in predictions:
                            pre_labels = pre_labels.replace(entity_word, "</S>" + entity_word + "<S>")
                        elif "CompositeMention" in predictions or "compositemention" in predictions:
                            pre_labels = pre_labels.replace(entity_word, "</C>" + entity_word + "<C>")
                        elif "Modifier" in predictions or "modifier" in predictions:
                            pre_labels = pre_labels.replace(entity_word, "</M>" + entity_word + "<M>")
                        else:
                            continue
                    words_flag.append(entity_word)

            Dic_ = {}
            Dic_["sentence"] = sentence
            Dic_["pre_labels"] = pre_labels
            Dic_["gold_labels"] = gold_l[i]
            json_str = json.dumps(Dic_)
            fw.write(json_str)
            fw.write("\n")
            fw.flush()




if __name__ == '__main__':
    # test()

    parser = get_parser()
    args = parser.parse_args()

    inputfile = "data/NCBI/GPTNER_gpt4_output_shot5_NCBI.json"
    gpt_results,sentences = read_gpt_out_data(dir_=inputfile)
    # print(gpt_results)
    ner_test = read_ner_test_data("data/NCBI/test_NCBI.json")

    data_name = "NCBI"
    transfered_label="DiseaseClass,SpecificDisease,CompositeMention,Modifier"
    prompts = mrc2prompt(ner_test=ner_test,gpt_results=gpt_results,transfered_label=transfered_label,sentences=sentences)


    out_putfile = "data/NCBI/NCBI_Idea1_test.json"  #
    testreader = Bertbiofewreader("data/NCBI/test_NCBI.json")
    batch_size = 1
    inputfile="data/NCBI/test_NCBI.json"

    gold_l=get_gold_labels( "data/NCBI/GPTNER_gpt4_output_shot5_NCBI.json")
    # print(len(gold_l))
    # for g in gold_l:
    #     print(g)
    # print(len(prompts))


    get_results(inputfile, out_putfile, prompts,gpt_results,gold_l)

