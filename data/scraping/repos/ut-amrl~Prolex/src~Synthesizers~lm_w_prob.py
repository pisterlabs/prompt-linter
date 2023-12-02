import torch
import openai
#from transformers import pipeline
#import time
#fill_masker = pipeline(model="bert-base-uncased", task="fill-mask")
def mult(l):
    if len(l) == 1:
        return l[0]
    else:
        l1 = l[0]
        l2 = l[1]
        m = []
        for i in range(len(l1)):
            for j in range(len(l2)):
                n = []
                n.append(l1[i][0]*l2[j][0])
                n += l1[i][1:]
                n += l2[j][1:]
                m.append(n)
        l.append(m)
        return mult(l[2:])

def most_likely_completion_scan(target_words, sentence, trace, fill_masker, hole=0):
    #Set up text_davinci input
    openai.api_key = "Add your open AI Key Here"
    assert False
    prompt = f"Given a candidate set of objects, return a reordered list of the candidate objects, in descending order of semantic relation to the *entire* trace. The return list should contain all of the candiate objects.\nExample 1:\n```\ncandidate_objects = ['Plant', 'Clothes', 'Cupboard', 'Bed', 'Table', 'Bucket']\n\ntrace = [Goto(Kitchen_1), action( Cup_1, Grab), action( Cup_2, Grab), action( Cup_3, Grab), Goto(Livingroom_1), action( Cup_4, Grab), action( Cup_5, Grab)]\n```\nOutput:\n```\nsorted_objects = ['Table', 'Cupboard', 'Plant', 'Bucket', 'Clothes', 'Bed']\n```\n\n\nExample 2:\n```\ncandidate_objects = ['Door', 'Sink', 'Fruit', 'Basket', 'Fan', 'Shelf']\n\ntrace = [Goto(Livingroom_1), action(Drawer_1, Open), action(Book_1, Grab), action(Pen_1, Grab), action(Drawer_1, Close), action(Book_1, Place), action(Pen_1, Place)]\n```\nOutput:\n```\nsorted_objects = ['Shelf', 'Basket', 'Door', 'Sink', 'Fruit', 'Fan']\n```\n\n\nWith the above examples in mind, return a reordered list of the following objects, in descending order of semantic relation to the trace provided below.\n```\ncandidate_objects = {target_words}\n\ntrace = {trace}\n```\nOutput:\n```\nsorted_objects ="
    
    response = openai.Completion.create(
            engine='text-davinci-003',
            temperature = 0,
            max_tokens = 200,
            prompt=prompt
    )

    reply_content = response.choices[0].text.strip()

    #print(reply_content)

    #print('\n\n')

    reply_content = reply_content.strip('[`')
    reply_content = reply_content.rstrip(']\n')
    
    #print(reply_content)

    #print('\n\n')

    ret_objs = []
    for res_str in reply_content.split(', '):
        res_str = res_str.strip("'")
        ret_objs.append(res_str)

    ret_objs_w_prob = []
    default_prob = 1.0
    for obj in ret_objs:
        ret_objs_w_prob.append((default_prob, obj))
        default_prob = default_prob/2

    #print(ret_objs_w_prob)

    return ret_objs_w_prob

"""
def most_likely_completion(target_words, sentence, fill_masker, hole = 0):
    #Get Embeddings:
    openai.api_key = "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"
    targ_embeddings = []
    for targ_word in target_words:
        response = openai.Embedding.create(
            input=f"{targ_word}",
            model="text-embedding-ada-002"
        )
        targ_embeddings.append(torch.tensor(response['data'][0]['embedding']))

    #Sentence Embedding:
    response = openai.Embedding.create(
        input=f"{sentence}",
        model="text-embedding-ada-002"
    )
    sentence_embedding = torch.tensor(response['data'][0]['embedding'])

    #Get Cosine Distance
    cos_sim = torch.nn.CosineSimilarity(dim=0)
    joint = []
    for i in range(len(targ_embeddings)):
        sim = cos_sim(sentence_embedding, targ_embeddings[i]).item()
        joint.append((sim, target_words[i]))

    completions = sorted(joint, key=lambda x: x[0], reverse=True)

    scores = []
    for score, _ in completions:
        scores.append(score)

    #print(f"Before Softmax: {completions}")

    epsilon = 1000

    scores = torch.tensor(scores)
    scores = (scores - torch.mean(torch.abs(scores)))*epsilon
    softmax = torch.nn.Softmax(dim=0)
    scores = softmax(scores)
    scores = scores.tolist()


    ret_comp = []
    for i in range(len(scores)):
            ret_comp.append((scores[i], completions[i][1]))

    #print(f"After Softmax: {ret_comp}")
    return ret_comp

"""

def most_likely_completion(target_words, sentence, fill_masker, hole = 0):
    #start_time = time.time()
    #print(target_words)

    if len(sentence) > 512:
        sentence = sentence[0:512]
        return []

    #print("Sentence: ", sentence, "\n")
    #print("Target: ", target_words, "\n")
    try:
        l = fill_masker(sentence, targets=target_words)
    except:
        return []
    #print("Return: ", l, "\n")
    if type(l[0]) == dict:
        l = [l]
    n_holes = len(l)
    joint = []

    for i in range(n_holes):
        l1 = []
        for pred in l[i]:
            l1.append([pred['score'], pred['token_str']])
        if l1 != []:
            joint.append(l1)
    # arr = mult(joint)
    # arr = sorted(arr, key=lambda x: x[0], reverse=True)
    completions = []
    if len(joint) > hole:
        completions = sorted(joint[hole], key=lambda x: x[0], reverse=True)
    # for i in range(5):
    #     print(f"Prediction {i+1} with a score of {arr[i][0]}:")
    #     j = 1
    #     s=""
    #     words = sentence.split(" ")
    #     for w in words:
    #         if w=="[MASK]":
    #             s+=arr[i][j] + " "
    #             j += 1
    #         else:
    #             s+=w + " "
    #     print(s[:-1])
    # print(completions)
    #end_time = time.time()
    #print("Time taken by LM: {} ms".format((end_time - start_time)*1000))

    scores = []
    for score, _ in completions:
        scores.append(score)

    #print(f"Before Softmax: {completions}")
    #print(f"Before Softmax: {completions}")

    epsilon = 1

    scores = torch.tensor(scores)
    #scores = (scores - torch.mean(torch.abs(scores)))*epsilon
    #softmax = torch.nn.Softmax(dim=0)
    #scores = softmax(scores)
    scores = scores.tolist()


    ret_comp = []
    for i in range(len(scores)):
            ret_comp.append((scores[i], completions[i][1]))

    #print(f"After Softmax: {ret_comp}")
    return ret_comp

    #scores = torch.tensor(scores)
    #scores = scores/torch.max(scores)
    #softmax = torch.nn.Softmax(dim=0)
    #scores = softmax(scores)
    #scores = scores.tolist()


    #ret_comp = []
    #for i in range(len(scores)):
    #        ret_comp.append((scores[i], completions[i][1]))

    #print(f"After Softmax: {ret_comp}")
    #return ret_comp

# most_likely_completion(["door", "box", "empty", "open", "close"],"Check if the [MASK] is [MASK] and close the door.")
"""
Only disadvantage of this approach is that joint probability assumes that these words are independent but they arent.
They have dependence. We can solve this by generating a prediction on all masks at once but that can't take targets as input.
"""


"""
Likelihood of a sentence using BERT
"""
# from pytorch_pretrained_bert import BertTokenizer, BertForMaskedLM
# import torch
# import pandas as pd
# import math	
# bertMaskedLM = BertForMaskedLM.from_pretrained('bert-base-uncased')
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# def get_score(sentence):
#     tokenize_input = tokenizer.tokenize(sentence)
#     tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
#     predictions=bertMaskedLM(tensor_input)
#     loss_fct = torch.nn.CrossEntropyLoss()
#     loss = loss_fct(predictions.squeeze(),tensor_input.squeeze()).data 
#     return math.exp(loss)
# print("Perplexity of \" Check if the door is open and close the door. \": ",get_score("Check if the door is open and close the door."))
# print("Perplexity of \" Check if the box is open and close the door. \": ", get_score("Check if the box is open and close the door."))
# print("Perplexity of \" Check if the door is empty and close the door. \": ", get_score("Check if the door is open and close the door. "))
# print("Perplexity of \" Check if the empty is open and close the door. \": ", get_score("Check if the empty is open and close the door. "))
