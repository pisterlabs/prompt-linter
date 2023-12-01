import torch
from tqdm import tqdm
import numpy as np
from Transfer.Models.Models import RobertaClassifierMultiHead
from Transfer.Datasets.Datasets import Kaggle_Toxicity,index2group
from Transfer.Datasets.Saliency_utils import attention,gradient_saliency,gradient_x_saliency,mask_attribution,gibbs,mask_tokens,\
    drop_token,drop_token_iterative,check_word_list,mask_word_vectors,prepend_labels,prepend_lm_saliency,drop_thresh
from transformers import RobertaTokenizer,BartTokenizer, BartForConditionalGeneration,RobertaForMaskedLM,BeamSearchScorer,StoppingCriteriaList,MaxLengthCriteria
from torch.utils.data import DataLoader,Subset
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import time
import json


def generate(removal_mode="attention",remove_label=None,n_batches=25,batch_size=4,max_length=256,q=0.9,model_type="Bart",
             generator_path="Bart_Unlabeled0",classifier_path="Roberta_test_identity_only",eval_classifier_path="Roberta_test_identity_only",
             target_label = None,sample_gibbs=False,add_label=False,overwrite_labels=False,
             contrast="mean",censor=False,n_beams=5,total_items=10,add_successes_only=False,
             attention_layer=11,shuffle_loader=True,save_experiment=None,save=None,gpt_mode=None,word_replacement_mode="Full",
             log_metrics=True,max_generations=100,thresh=0.25):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    assert gpt_mode is None or model_type=="GPT3"
    if model_type == "GPT3":
        import openai
        with open('key.txt') as f:
            lines = f.readlines()[0].replace("\n", "")
        openai.api_key = lines


    if model_type=="Bart":
        generator = BartForConditionalGeneration.from_pretrained("facebook/bart-large").to(device)
        generator.load_state_dict(torch.load(generator_path))
        tokenizer_bart = BartTokenizer.from_pretrained("facebook/bart-large")
    else:
        generator=RobertaForMaskedLM.from_pretrained("roberta-base")
    generator = generator.to(device)
    dataset = Kaggle_Toxicity(max_length=max_length)
    if not (removal_mode == "word_list"):
        model = RobertaClassifierMultiHead(24).to(device)
        model.load_state_dict(torch.load(classifier_path))

    model_eval = RobertaClassifierMultiHead(24).to(device)
    model_eval.load_state_dict(torch.load(eval_classifier_path))

    tokenizer_roberta = RobertaTokenizer.from_pretrained("roberta-base")

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_loader)
    t = tqdm(enumerate(train_loader), total=min(n_batches,len(train_loader)),dynamic_ncols=True)
    text_pairs = []
    total_removals = 0
    total_reversals = 0
    possible_reversals = 0
    total_generations = 0
    for step, texts in t:
        texts, labels = texts
        data = tokenizer_roberta(texts, return_tensors="pt", truncation=True, max_length=max_length, padding=True).to(device)
        if overwrite_labels:
            with torch.no_grad():
                labels = model_eval(data).detach()
        else:
            labels = labels.to(device)
        labels_source = labels[:,remove_label]>0
        total_generations += labels_source.sum()
        #print(total_generations)
        target_label_values = labels[:,target_label][labels_source]>0
        if labels_source.sum() == 0:
            continue
        data = data["input_ids"][labels_source]
        texts = [texts[i] for i in range(len(texts)) if labels_source[i]]
        if not (model_type is None or model_type=="GPT3"):
            if removal_mode == "attention":
                attributions = attention(data, model, batch_size=batch_size,layer=attention_layer)
            elif removal_mode =="grad":
                assert not remove_label is None
                attributions= gradient_saliency(data,model,batch_size=batch_size,target_index=remove_label)
            elif removal_mode =="grad_x":
                assert not remove_label is None
                attributions= gradient_x_saliency(data,model,batch_size=batch_size,target_index=remove_label)
            elif removal_mode == "drop":
                attributions = drop_token(data, model, batch_size=batch_size, target_index=remove_label)
            elif removal_mode == "drop_iter":
                attributions = drop_token_iterative(data, model, batch_size=batch_size, target_index=remove_label)
            elif removal_mode == "drop_thresh":
                attributions = drop_thresh(data, model, batch_size=batch_size, target_index=remove_label,hard_threshold=thresh)
            elif removal_mode == "bart_likelihood":
                attributions =  prepend_lm_saliency(texts,generator,tokenizer_bart,remove_label,contrast=contrast,censor_encoder=censor)
        if model_type=="Bart":
            if not (removal_mode == "word_list" or removal_mode=="word_vectors"):
                masked_text = [
                    mask_attribution(tokenizer_roberta.convert_ids_to_tokens(data[i]), attributions[i], q=q, delete_repeat=model_type=="Bart") for i
                    in range(len(attributions))]
            elif removal_mode=="word_list":
                masked_text = check_word_list(texts, target_index=remove_label, delete_repeat=model_type=="Bart")
            generation_batch = tokenizer_bart(masked_text, return_tensors="pt", truncation=True, max_length=max_length, padding=True).to(device)["input_ids"]
            if add_label:
                if n_beams is None:
                    generated_ids = generator.generate(prepend_labels(generation_batch, [target_label for i in range(len(generation_batch))], tokenizer_bart,all_labels=True), max_length=256)
                else:
                    generated_ids = []
                    for i in range(len(generation_batch)):
                        generation_candidates = generator.generate(
                            prepend_labels(generation_batch[i:i+1], [target_label for i in range(1)], tokenizer_bart,
                                           all_labels=True), max_length=max_length, num_beams=n_beams, num_return_sequences=n_beams)
                        generation_candidate_texts = tokenizer_bart.batch_decode(generation_candidates, skip_special_tokens=True)
                        label_candidates =  model(
                        tokenizer_roberta(generation_candidate_texts , return_tensors="pt", truncation=True, max_length=max_length, padding=True).to(device)).detach()
                        scores = label_candidates[:,target_label]-label_candidates[:,remove_label]
                        generated_ids.append(generation_candidates[torch.argmax(scores)])
                    maxlen = max([len(generated_ids[i]) for i in range(len(generated_ids))])
                    for i in range(len(generation_batch)):
                        generated_ids[i] = torch.concat([generated_ids[i],torch.tensor([tokenizer_bart.pad_token_id for i in range(maxlen-len(generated_ids[i]))]).to(generation_batch.device)])
                    generated_ids = torch.stack(generated_ids)
            else:
                assert n_beams is None, "not implemented"
                generated_ids = generator.generate(generation_batch, max_length=256)

            decoded = []
            for i in range(len(generated_ids)):
                decoded.append(tokenizer_bart.decode(generated_ids[i], skip_special_tokens=True))
        elif model_type == "Roberta":
            assert n_beams is None, "not implemented"
            generation_batch = mask_tokens(data, tokenizer_roberta, attributions, q=q).to(device)
            generated_ids = gibbs(generator,tokenizer_roberta,generation_batch,sample_tokens=sample_gibbs)
            decoded = []
            for i in range(len(generated_ids)):
                decoded.append(tokenizer_roberta.decode(generated_ids[i], skip_special_tokens=True))
        elif model_type == "GPT3":
            decoded = []
            masked_text = []
            if not gpt_mode == "davinci_postprocess":
                for i in range(len(texts)):
                    masked_text.append("No masking for GPT")
                    if gpt_mode == "curie_zero":
                        response = openai.Completion.create(model="text-curie-001",
                                                            prompt=get_basic_gpt_prompt(texts[i],
                                                                index2group(remove_label),index2group(target_label)),
                                                            temperature=0.7,
                                                            max_tokens=64,
                                                            top_p=1,
                                                            frequency_penalty=0,
                                                            presence_penalty=0)
                        decoded.append(response["choices"][0]["text"].replace("\n"," "))

                    if gpt_mode == "curie_agnostic":
                        response = openai.Completion.create(model="text-curie-001",
                                                            prompt=get_agnostic_few_shot_prompt(texts[i],
                                                                                        index2group(remove_label),
                                                                                        index2group(target_label)),
                                                            temperature=0.7,
                                                            max_tokens=64,
                                                            top_p=1,
                                                            frequency_penalty=0,
                                                            presence_penalty=0)
                        decoded.append(response["choices"][0]["text"][:-1].replace("\n", " "))

                    if gpt_mode == "curie_targeted":
                        response = openai.Completion.create(model="text-curie-001",
                                                            prompt=get_targeted_few_shot_prompt(texts[i],
                                                                                                index2group(
                                                                                                    remove_label),
                                                                                                index2group(
                                                                                                    target_label)),
                                                            temperature=0.7,
                                                            max_tokens=64,
                                                            top_p=1,
                                                            frequency_penalty=0,
                                                            presence_penalty=0)
                        decoded.append(response["choices"][0]["text"][:-1].replace("\n", " "))

                    if gpt_mode == "davinci_zero":
                        response = openai.Completion.create(model="text-davinci-001",
                                                            prompt=get_basic_gpt_prompt(texts[i],
                                                                                                index2group(
                                                                                                    remove_label),
                                                                                                index2group(
                                                                                                    target_label)),
                                                            temperature=0.7,
                                                            max_tokens=64,
                                                            top_p=1,
                                                            frequency_penalty=0,
                                                            presence_penalty=0)
                        decoded.append(response["choices"][0]["text"].replace("\n", " "))

                    if gpt_mode == "davinci_edit":
                        time.sleep(2)  # There is a rate limit on API requests (possibly only for editing?)
                        try:
                            response = openai.Edit.create(model="text-davinci-edit-001", input=texts[i],
                                                          instruction=get_edit_instructions(index2group(remove_label),index2group(target_label)),
                                                          temperature=0.7,
                                                          top_p=1)
                            decoded.append(response["choices"][0]["text"].replace("\n", " "))
                        except:
                            decoded.append(texts[i])
            else:
                if word_replacement_mode == "Full":
                    masked_text, decoded = check_word_list(texts, target_index=remove_label, replace_index=target_label)
                elif word_replacement_mode == "50":
                    masked_text, decoded = check_word_list(texts, target_index=remove_label, replace_index=target_label,list50=True)
                for i in range(len(decoded)):
                    try:
                        time.sleep(2)  # There is a rate limit on API requests (possibly only for editing?)
                        response = openai.Edit.create(model="text-davinci-edit-001", input=decoded[i],
                                                      instruction="Fix grammatical errors and logical inconsistencies",
                                                      temperature=0.7,
                                                      top_p=1)
                        decoded[i] = response["choices"][0]["text"].replace("\n", " ")
                    except:
                        decoded[i] = decoded[i]

        elif model_type == None:
            #print("Warning: model type None produces word-replacements only!")
            if word_replacement_mode == "Full":
                masked_text,decoded = check_word_list(texts, target_index=remove_label,replace_index=target_label)
            elif word_replacement_mode == "50":
                masked_text, decoded = check_word_list(texts, target_index=remove_label, replace_index=target_label,
                                                       list50=True)
        with torch.no_grad():
            labels_new = model_eval(
                tokenizer_roberta(decoded, return_tensors="pt", truncation=True, max_length=max_length, padding=True).to(
                    device)).detach()
        total_removals += (labels_new[:, remove_label] < 0).float().sum().item()
        total_reversals += ((labels_new[:, target_label] > 0) > target_label_values).float().sum().item()
        possible_reversals += (target_label_values == 0).float().sum().item()

        for i in range(len(texts)):
            if add_successes_only:
                if (labels_new[i, remove_label] < 0) and (labels_new[i, target_label] > 0):
                    text_pairs.append((texts[i], masked_text[i], decoded[i]))
            else:
                text_pairs.append((texts[i], masked_text[i], decoded[i]))
        if step>n_batches:
            break
        if len(text_pairs) >= total_items:
            text_pairs = text_pairs[:total_items]
            break
        if total_generations >= max_generations:
            break

    if log_metrics:
        generator = None
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        generator = GPT2LMHeadModel.from_pretrained('gpt2').to(device)

    print(len(text_pairs),total_generations,step*batch_size)
    if not save_experiment is None:
        _make_json([text_pair[0] for text_pair in text_pairs],[text_pair[2] for text_pair in text_pairs],
                   save_experiment+".jsonl",
                   groups_a=[index2group(remove_label) for i in range(len(text_pairs))],
                   groups_b=[index2group(target_label) for i in range(len(text_pairs))])
    if not save is None:
        with open(save + '_base.jsonl', 'w') as outfile:
            for entry in [text_pair[0] for text_pair in text_pairs]:
                json.dump(entry, outfile)
                outfile.write('\n')
        with open(save + '_perturbed.jsonl', 'w') as outfile:
            for entry in [text_pair[2] for text_pair in text_pairs]:
                json.dump(entry, outfile)
                outfile.write('\n')

    return [text_pair[0] for text_pair in text_pairs],[text_pair[2] for text_pair in text_pairs], [index2group(remove_label) for i in range(len(text_pairs))], [index2group(target_label) for i in range(len(text_pairs))]


def get_basic_gpt_prompt(text,attribute_base,attribute_target):
    return "Please rewrite the following sentence to be about "+attribute_target+" rather than "+attribute_base+ ":" + "\n" +text

def get_agnostic_few_shot_prompt(text,attribute_base,attribute_target):
    context = """Here is some text: {When the doctor asked Linda to take the medicine, he smiled and gave her a lollipop.}.
    Here is a rewrite of the text, which is more scary. {When the doctor told Linda to take the medicine, 
    there had been a malicious gleam in her eye that Linda didn’t like at all.} 
    Here is some text: {they asked loudly, over thesound of the train.}. 
    Here is a rewrite of the text, which is more intense. {they yelled aggressively, over the clanging of the train.} 
    Here is some text: {When Mohammed left the theatre, it was already dark out}. 
    Here is a rewrite of the text, which is more about the movie itself. {The movie was longer than Mohammed had expected,
    and despite the excellent ratings he was a bit disappointed when he left the theatre.} 
    Here is some text: {next to the path}. 
    Here is a rewrite of the text, which is about France. {next to la Siene} 
    Here is some text: {The man stood outside the grocery store, ringing the bell.}. 
    Here is a rewrite of the text, which is about clowns. {The man stood outside the circus, holding a bunch of balloons.} 
    Here is some text: {the bell ringing}. 
    Here is a rewrite of the text, which is more flowery. {the peales of the jangling bell} 
    Here is some text: {against the tree}. 
    Here is a rewrite of the text, which is include the word "snow". {against the snow-covered bark of the tree}
    Here is some text: {
    """
    return context + text + "}" + "Here is a rewrite of the text, which is about" +attribute_target+" rather than "+attribute_base+ ": {"
def get_targeted_few_shot_prompt(text,attribute_base,attribute_target):
    context = """Here is some text: {When the doctor asked Linda to take the medicine, he smiled and gave her a lollipop.}.
    Here is a rewrite of the text, which is about men rather than women. {When the doctor asked Peter to take the medicine, he smiled and gave him a lollipop.} 
    Here is some text: {When the Imam left the theatre, it was already dark out}. 
    Here is a rewrite of the text, which is about women rather than muslim people. {When she left the theatre, it was already dark out} 
    Here is some text: {The man stood outside the synagogue, ringing the bell.}. 
    Here is a rewrite of the text, which is about christian people rather than jewish people. {The man stood outside the church, ringing the bell.} 
    Here is some text: {They met at the yearly pride parade}. 
    Here is a rewrite of the text, which is about asian people rather than homosexual people. {They met at the yearly celebrations for the chinese new year.}
    Here is some text: {
    """
    return context + text + "}" + "Here is a rewrite of the text, which is about" +attribute_target+" rather than "+attribute_base+ ". {"

def get_edit_instructions(attribute_base,attribute_target):
    return "Rewrite the text to be about "+attribute_target + " rather than " + attribute_base

