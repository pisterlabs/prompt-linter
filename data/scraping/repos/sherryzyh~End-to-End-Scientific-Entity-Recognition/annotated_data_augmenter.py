import os
import json
import spacy
import openai
import utils
import spacy
from spacy.lang.en import English
from spacy.tokenizer import Tokenizer
from dotenv import load_dotenv
import time

load_dotenv()

class AnnotatedDataAug:
    def __init__(self, working_dir = "./Annotated_Data"):
        os.chdir(working_dir)
        # if change the dir name of aug papers in Annotated Data we also need to change the parameters here
        self.aug_dir = "./clean_data_aug/"
        # if change the dir name of raw papers in Annotated Data we also need to change the parameters here
        self.raw_dir = "./cleaned_data/"
        self.openai_client = utils.OpenAIClient()
        self.tokenizer = utils.MyTokenizer()
        self.tmp_name = "ABCDEFG"
        self.tmp_len = len(self.tmp_name)
    
    def recover_and_label(self, annotated_tokenized_paper: str):
        """
        we need to recover the tokenized paper to the original sentences,
        and obtain the label for each token of the given sentence. (the token->label mapping is specified within a given sentence)

        Args:
            annotated_tokenized_paper: the path of the paper that is consisted of (token, label) pair for each line
        Returns:
            A list of sentences in the original paper;  
            A dict which map the sentenceIdx to its corresponding token->label map
        """
        # global result
        sentences = []
        sentenceIdx_to_tokenLabelDict = {} # map the index of the sentence to the corresponding token->label dict
        sentenceIdx_to_subNameDict = {} # map the index of the sentence to the corresponnding cnt->(original name, original label) for all the non-O entities
        with open(annotated_tokenized_paper) as f:
            lines = f.readlines()
            # cur maintainer
            cur_dict = {}
            label_set = set()
            cur_sentence = []
            sub_dict = {}
            cnt = 0
            for idx, token_label in enumerate(lines):
                if token_label == "\n":
                    # if split point
                    # add the period "." manually (always make sure there is a period at the end)
                    # if do not want only to paraphase labeled data, remove the condition
                    if len(label_set) > 0:
                        if cur_sentence and cur_sentence[-1] != ".":
                            cur_sentence.append(".")
                        sentences.append(" ".join(cur_sentence))
                        sentenceIdx_to_tokenLabelDict[len(sentences)-1] = cur_dict
                        sentenceIdx_to_subNameDict[len(sentences)-1] = sub_dict
                    cur_dict = {}
                    cur_sentence = []
                    label_set = set()
                    sub_dict = {} 
                    cnt = 0
                    continue
                else:
                    # o.w.
                    token = " "
                    if token_label[0] != " ":
                        token_label = token_label.strip()
                        token_label_lst = token_label.split()
                        if len(token_label_lst) < 2:
                            continue
                        token, label = token_label_lst[0], token_label_lst[1]
                        cur_dict[token] = label
                        if label[0] == "B":
                            label_set.add(label)
                            sub_dict[cnt] = [(token, label)]
                            tmp = f"{self.tmp_name}{cnt}"
                            cur_sentence.append(tmp)
                            cnt += 1
                            continue
                        if label[0] == "I":
                            sub_dict[cnt - 1].append((token, label))
                            continue
                    cur_sentence.append(token)
            if cur_sentence:
                sentences.append(" ".join(cur_sentence))
                sentenceIdx_to_tokenLabelDict[len(sentences)-1] = cur_dict
                sentenceIdx_to_subNameDict[len(sentences)-1] = sub_dict
        return sentences, sentenceIdx_to_tokenLabelDict, sentenceIdx_to_subNameDict
    
    def parapharase_and_relabel(self, sentences, sentenceIdx_to_tokenLabelDict, sentenceIdx_to_subNameDict):
        paraphrased_result_lines = []
        for idx, sentence in enumerate(sentences):
            parapharased_sentence = self.openai_client.getParaphrasedSentence(sentence)
            parapharased_tokens = self.tokenizer.get_tokens(parapharased_sentence)
            original_tokenLabelDict = sentenceIdx_to_tokenLabelDict[idx]
            original_nameDict = sentenceIdx_to_subNameDict[idx]
            para_token_label_lst = []
            for x in parapharased_tokens:
                lower_x = x.text.lower().strip()
                if len(lower_x) > self.tmp_len and lower_x[:self.tmp_len] == self.tmp_name.lower() and lower_x[self.tmp_len:].isdigit():
                    tmp_idx = int(lower_x[self.tmp_len:])
                    if tmp_idx in original_nameDict:
                        tmp_name_label_lst = original_nameDict[tmp_idx]
                        for name, label in tmp_name_label_lst:
                            para_token_label_lst.append(f"{name} {label}\n")
                    else:
                        para_token_label_lst.append(f"{x} O\n")
                else:
                    para_token_label_lst.append(f"{x} O\n")
                    
                # for key, label in original_tokenLabelDict.items():
                #     if key.lower().strip() == lower_x and label != "O":
                #         para_token_label_lst.append(f"{key} {label}\n")
                #         break 
                # else:
                #     para_token_label_lst.append(x.text + " O\n")
            para_token_label_str = "".join(para_token_label_lst)
            paraphrased_result_lines.append(para_token_label_str)
        return paraphrased_result_lines
    
    def data_aug(self, raw_dir = None, aug_dir=None):
        raw_dir = raw_dir if raw_dir else self.raw_dir
        aug_dir = aug_dir if aug_dir else self.aug_dir
        auged_files = set(os.listdir(aug_dir))
        cnt = 0
        print(f"Already augemented files: {auged_files}" )
        for file in os.listdir(raw_dir):
            if file and file[0] == ".":
                continue
            print(f"Total Augmented File Cnt: {cnt}")
            aug_file = f"aug_{file}"
            print(f"Augmenting {aug_file}...")
            if aug_file in auged_files:
                print(f"Already Parsed, Skip the annotated paper {file}")
                continue
            
            print(f"Paraphrase and relabelling from OpenAI...")
            sentences, sentenceIdx_to_tokenLabelDict, sentenceIdx_to_subNameDict = self.recover_and_label(raw_dir + file)
            paraphrased_result_lines = self.parapharase_and_relabel(sentences, sentenceIdx_to_tokenLabelDict, sentenceIdx_to_subNameDict)
            
            print(f"Write the result to file...")
            aug_file_path = os.path.join(aug_dir, aug_file)
            
            tokenf = open(aug_file_path, "w")
            tokenf.write("\n".join(paraphrased_result_lines))
            tokenf.close()
            # update the seen set to avoid duplicate aug
            auged_files.add(aug_file)
            

        # os.chdir(aug_dir if aug_dir else self.aug_dir)

# working_dir = "./Annotated_Data"
# data_augmenter = AnnotatedDataAug(working_dir)
# data_augmenter.data_aug()



