import os
import string
import json
import torch
import numpy as np
import openai
import random
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel
from nemo.collections.nlp.models import PunctuationCapitalizationModel
import argparse
from tqdm import tqdm
import spacy
from sentence_transformers import SentenceTransformer
import multiprocessing as mp
import _io

nlp = spacy.load('en_core_web_sm')
sent_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-mpnet-base-v2")

def get_next_character(text_list, index1, index2):
    '''
    Gets next character from a text list. Index 1 -> rows (entry), Index 2 -> cols (word).
    '''
    if index1 == len(text_list):
        return None, index1, index2
    if index2 == len(text_list[index1]):
        return get_next_character(text_list, index1+1, 0)
    if text_list[index1][index2].isspace():
        return get_next_character(text_list, index1, index2+1)
    return text_list[index1][index2], index1, index2

def align_after_postprocess(postprocessed, original):
    '''
    
    '''
    index_map = {}
    speech_segment_index = 0
    within_segment_index = 0
    p_index = 0
    postprocessed_l = postprocessed # .lower()
    while p_index < len(postprocessed_l):
        if postprocessed_l[p_index].isspace():
            p_index += 1
            continue
        char, speech_segment_index, within_segment_index = get_next_character(original["text"], speech_segment_index, within_segment_index)
        if char is not None:
            _, next_speech_segment_index, next_within_segment_index = get_next_character(original["text"], speech_segment_index, within_segment_index+1)
            if postprocessed_l[p_index].upper().lower() == char.upper().lower() or postprocessed_l[p_index:p_index+2].upper().lower() == char.upper().lower():
                index_map[p_index] = (speech_segment_index, within_segment_index)
                speech_segment_index = next_speech_segment_index
                within_segment_index = next_within_segment_index
        p_index += 1
    return index_map

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def encode_section(sent_model, sents, start, end):
    section = ' '.join(sents[start:end])
    return {(start, end): sent_model.encode([section])[0]}

def remove_punctuation(text):
    new_text = text
    for c in string.punctuation:
        new_text = new_text.replace(c, '')
    return new_text

def align_text(text, original_text, steps, sent_model, num_workers, dtw=True, dtw_window_size=10000000000, dtw_start_offset=False):
    doc = nlp(text)
    sents = [str(sent) for sent in list(doc.sents)]
    steps = steps[:len(sents)]
    step_embs = sent_model.encode(steps)
    text = text.replace('ı', 'i')
    if dtw:
        dtw_matrix = np.zeros((len(steps)+1, len(sents)+1, len(sents)+1))
        for i in range(len(steps)+1):
            for start in range(len(sents)+1):
                for end in range(len(sents)+1):
                    dtw_matrix[i,start,end] = -np.inf
        dtw_matrix[0,0,0] = 0
        pointers = -1*np.ones((len(steps)+1, len(sents)+1, len(sents)+1), dtype=np.int32)
        pointer_scores = -np.inf*np.ones((len(steps)+1, len(sents)+1, len(sents)+1), dtype=np.float32)
        start_sent_index = 0
        if dtw_start_offset:
            single_sent_emb = np.stack([sent_model.encode([sent])[0,:] for sent in sents])
            start_scores = (step_embs[:1,:]*single_sent_emb).sum(1)
            start_sent_index = min(max(0, start_scores.argmax()-1), len(sents)-len(steps))
            dtw_matrix[0,start_sent_index,start_sent_index] = 0
        section_emb = {}
        if num_workers == 1:
            batch = []
            for start in range(start_sent_index, len(sents)):
                for end in range(start+1, min(start+dtw_window_size+1, len(sents)+1)):
                    section = ' '.join(sents[start:end])
                    batch.append((start, end, section))
                    if len(batch) == 16 or (start == len(sents)-1 and end == len(sents)):
                        inputs = [item[-1] for item in batch]
                        outputs = sent_model.encode(inputs)
                        for item, output in zip(batch, outputs):
                            section_emb[item[:2]] = output
                        batch = []
            if len(batch) > 0:
                inputs = [item[-1] for item in batch]
                outputs = sent_model.encode(inputs)
                for item, output in zip(batch, outputs):
                    section_emb[item[:2]] = output
        else:
            with mp.Pool(num_workers) as pool:
                section_emb_list = pool.starmap(encode_section, [(sent_model, sents, start, end) for start in range(0, len(sents)) for end in range(start+1, min(start+dtw_window_size+1, len(sents)+1))])
            for emb_dict in section_emb_list:
                section_emb.update(emb_dict)
        for i in range(1, len(steps)+1):
            for start in range(start_sent_index, len(sents)):
                for end in range(start+1, min(start+dtw_window_size+1, len(sents)+1)):
                    section = ' '.join(sents[start:end])
                    sentence_emb = section_emb[(start,end)] # sent_model.encode([section])[0]
                    step_emb = step_embs[i-1] # sent_model.encode([steps[i-1]])[0]
                    similarity = (sentence_emb*step_emb).sum().item()
                    best_prev_segment = dtw_matrix[i-1,:,start].argmax().item()
                    prev_segment_score = dtw_matrix[i-1,:,start].max().item()
                    # if prev_segment_score > dtw_matrix[i-1,start,end].item():
                    #     pointers[i,start,end] = best_prev_segment
                    # else:
                    #     pointers[i,start,end] = start
                    pointers[i,start,end] = best_prev_segment
                    pointer_scores[i,start,end] = prev_segment_score
                    last_max = np.max([prev_segment_score]) # , dtw_matrix[i-1,start,end]])
                    dtw_matrix[i,start,end] = similarity+last_max
            # print('good', i, [j for j in range(dtw_matrix.shape[1]) if dtw_matrix[i,j,:].max().item() > -np.inf])
        end = dtw_matrix.shape[1]-1
        index = dtw_matrix.shape[0]-1
        start = dtw_matrix[index,:,end].argmax().item()
        print(dtw_matrix[index,:,:end].max().item())
        segments = {index: (start, end)}
        index -= 1
        while index > 0:
            # print(index+1, start, end)
            new_start = int(pointers[index+1,start,end])
            print(pointer_scores[index+1,start,end])
            if new_start != start:
                end = start
                start = new_start
            # else:
            #     print('bad', pointers[index+1,start,end], pointer_scores[index+1,start,end])
            segments[index] = (start, end)
            index -= 1
        print(start_sent_index, segments)
    else:
        sent_emb = sent_model.encode(sents)
        scores = torch.matmul(torch.from_numpy(step_embs), torch.from_numpy(sent_emb).t())
        matched_sentences = scores.argmax(dim=-1).tolist()
        segments = {}
        for i in range(1, len(steps)+1):
            print(steps[i-1], '|||', sents[matched_sentences[i-1]])
            segments[i] = (max(0, matched_sentences[i-1]-1), min(len(sents), matched_sentences[i-1]+2))
    # text_sans_punct = remove_punctuation(text)
    # assert text_sans_punct.lower() == ' '.join(original_text['text'])
    postprocess_alignment = align_after_postprocess(text, original_text)
    # print(segments)
    # print(postprocess_alignment)
    aligned_segments = {}
    sents = list(doc.sents)
    # print(text)
    # print(original_text)
    # print(' '.join(original_text['text']))
    # print(max(list(postprocess_alignment.keys())), [sents[segments[index][0]].start_char for index in segments], [text[sents[segments[index][0]].start_char:sents[segments[index][1]-1].end_char] for index in segments])
    for index in segments:
        while str(sents[segments[index][0]]).isspace():
            segments[index] = (segments[index][0]-1, segments[index][1])
        start = sents[segments[index][0]].start_char
        while start not in postprocess_alignment and start < len(text):
            start += 1
        if start not in postprocess_alignment:
            print('A', sents[segments[index][0]])
            print('B', text[sents[segments[index][0]].start_char:], sents[segments[index][0]].start_char)
            print('C', text)
            print('D', ' '.join(original_text['text']))
            print(sents[segments[index][0]].start_char, sorted(list(postprocess_alignment.keys()))[-50:])
        assert start in postprocess_alignment
        end = sents[segments[index][1]-1].end_char-1
        while end not in postprocess_alignment and end >= 0:
            end -= 1
        assert end in postprocess_alignment
        aligned_segments[index] = postprocess_alignment[start]+postprocess_alignment[end]
        print('aligned', ' '.join(original_text['text'][aligned_segments[index][0]:aligned_segments[index][2]+1]), sents[segments[index][0]:segments[index][1]])
    return aligned_segments

def remove_repeat_ngrams(text_list, min_n=3, max_n=8, return_segment_ids=False):
    assert isinstance(text_list, list)
    tokens = []
    segment_ids = []
    for segment_id, segment in enumerate(text_list):
        segment_tokens = segment.split()
        for token in segment_tokens:
            if len(token) > 0:
                tokens.append(token)
                segment_ids.append(segment_id)
    inside_segment = False
    num_streak_tokens = 0
    new_tokens = []
    new_segment_ids = []
    indices_added = set()
    for i in range(len(tokens)):
        redundant = False
        for j in range(max_n, min_n-1, -1):
            if i+1 >= j*2 and tokens[i+1-j:i+1] == tokens[i+1-j*2:i+1-j]:
                # print('here', tokens[i+1-j*2:i+1])
                inside_segment = True
                num_streak_tokens = min_n
                for k in range(1, j):
                    if i-k in indices_added:
                        new_tokens.pop()
                        new_segment_ids.pop()
                        indices_added.remove(i-k)
                redundant = True
                break
        if not redundant:
            new_tokens.append(tokens[i])
            indices_added.add(i)
            new_segment_ids.append(segment_ids[i])
    if return_segment_ids:
        return ' '.join(new_tokens), new_segment_ids
    return ' '.join(new_tokens)

def process_video(video_id, args, input_steps, transcripts, tokenizer, punct_cap_model, output_queue):
    '''
    Main function that processes the video. Takes in arguments:
    - video_id: 
    - args:
    - input_steps:
    - transcripts:
    - tokenizer:
    - punct_cap_model:
    - output_queue:
    '''
    prompt = "Write the steps of the task that the person is demonstrating, based on the noisy transcript.\nTranscript: |||1\nSteps:\n1."
    print('here3')
    # Indexes into transcripts if argument is passed, else processes it
    if transcripts is not None:
        original = transcripts[video_id]
    else:
        f = open(os.path.join(args.transcripts_path, video_id+".csv"))
        lines = f.readlines()
        original = {"text": [], "start": [], "end": []}
        for line in lines[1:]:
            parts = line.split(',')
            original["start"].append(float(parts[0]))
            original["end"].append(float(parts[1]))
            original["text"].append(parts[-1].strip())
    transcript = " ".join(original["text"])
    # Removes repeated n-grams
    deduplicated_text, new_segment_ids = remove_repeat_ngrams(original["text"], min_n=3, max_n=9, return_segment_ids=True)
    deduplicated_tokens = deduplicated_text.split()
    original["text"] = [[] for _ in range(len(original["text"]))]
    for token, new_id in zip(deduplicated_tokens, new_segment_ids):
        original["text"][new_id].append(token)
    original["text"] = [" ".join(lst) for lst in original["text"]]
    transcript = " ".join(original["text"])
    # Creates the output path, adds capitalization and other formatting if specified
    if not args.no_formatting:
        if args.formatted_transcripts_path is not None:
            fname = os.path.join(args.formatted_transcripts_path, video_id+".txt")
        if args.formatted_transcripts_path is not None and os.path.exists(fname):
            f = open(fname)
            transcript = f.readlines()[0]
        else:
            transcript = punct_cap_model.add_punctuation_capitalization([transcript])[0]
    # Tokenizes transcript and saves it as `tokens`
    tokens = tokenizer(transcript)
    print(video_id, len(transcript), len(tokens["input_ids"]))
    while len(tokens["input_ids"]) > 1600:
        transcript = transcript[:-100]
        tokens = tokenizer(transcript)
    if args.input_steps_path is not None:
        if video_id not in input_steps:
            return
        steps = input_steps[video_id]["steps"]
    else:
        if video_id in finished:
            return
        input_text = prompt.replace("|||1", transcript)
        steps = []
        num_attempts = 0
        while len(steps) == 0:
            response = openai.Completion.create(
                            engine="text-babbage-001",
                            prompt=input_text,
                            temperature=0.7,
                            max_tokens=256,
                            top_p=1,
                            frequency_penalty=0,
                            presence_penalty=0
                        )
            output = response["choices"][0]["text"].strip()
            num_attempts += 1
            steps = output.split("\n")
            if all(["." in step for step in steps[1:]]):
                steps = steps[:1]+[step[step.index(".")+1:].strip() for step in steps[1:]]
            elif num_attempts < args.max_attempts:
                steps = []
    output_dict = {"video_id": video_id, "steps": steps, "transcript": transcript}
    if not args.no_align:
        segments = align_text(transcript, original, steps, sent_model, args.num_workers, not args.no_dtw, args.dtw_window_size)
        print(segments)
        output_dict["segments"] = segments
    if isinstance(output_queue, _io.TextIOWrapper):
        output_queue.write(json.dumps(output_dict)+'\n')
    else:
        output_queue.put(json.dumps(output_dict)+'\n')

def output_listener(output_queue, output_filename):
    mode = 'a+' if os.path.exists(output_filename) else 'w'
    with open(output_filename, 'a+') as fout:
        while True:
            output = output_queue.get()
            if output == 'kill':
                break
            fout.write(output)
            fout.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_list_path")
    parser.add_argument("--transcripts_path")
    parser.add_argument("--formatted_transcripts_path")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=None)
    parser.add_argument("--max_attempts", type=int, default=1)
    parser.add_argument("--no_formatting", action="store_true")
    parser.add_argument("--output_path")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no_align", action="store_true")
    parser.add_argument("--input_steps_path", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--no_dtw", action="store_true")
    parser.add_argument("--dtw_window_size", type=int, default=1000000)
    args = parser.parse_args()
    
    '''
    Specify device, CPU vs. GPU
    '''
    if not args.no_align:
        if args.cpu:
            sent_model = SentenceTransformer('sentence-transformers/paraphrase-mpnet-base-v2').cpu()
        else:
            sent_model = SentenceTransformer('sentence-transformers/paraphrase-mpnet-base-v2').cuda()
    # sent_model = AutoModel.from_pretrained('sentence-transformers/paraphrase-mpnet-base-v2').cuda()

    '''
    Args no formatting - load pretrained punctuation capitalization model
    '''
    if not args.no_formatting:
        punct_cap_model = PunctuationCapitalizationModel.from_pretrained("punctuation_en_bert")
        if args.cpu:
            punct_cap_model = punct_cap_model.cpu()
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    '''
    Opens list of videos
    '''
    f = open(args.video_list_path)
    lines = f.readlines()
    video_ids = [line.strip().split()[0].split('.')[0] for line in lines]
    '''
    Loads transcripts
    '''
    transcripts = None
    if args.transcripts_path[-5:] == ".json":
        f = open(args.transcripts_path)
        transcripts = json.load(f)
    '''
    Video End-index, can be used to truncate # videos read
    '''
    if args.end_index is not None:
        video_ids = video_ids[:args.end_index]
    '''
    Video Start-index
    '''
    video_ids = video_ids[args.start_index:]
    '''
    Ending: output is read and the video id is added to set "finished"
    '''
    finished = set()
    if os.path.exists(args.output_path):
        fout = open(args.output_path)
        written_lines = fout.readlines()
        fout.close()
        for line in written_lines:
            try:
                datum = json.loads(line)
                finished.add(datum['video_id'])
            except:
                pass
        fout = open(args.output_path, 'a')
    else:
        fout = open(args.output_path, 'w')
    '''
    Reads input_steps
    '''
    input_steps = None
    if args.input_steps_path is not None:
        f = open(args.input_steps_path)
        lines = f.readlines()
        input_steps = [json.loads(line) for line in lines]
        input_steps = {datum["video_id"]: datum for datum in input_steps}
    """manager = mp.Manager()
    q = manager.Queue()
    pool = mp.Pool(args.num_workers+2)
    watcher = pool.apply_async(output_listener, (q, args.output_path))
    print('here1', pool._processes)
    jobs = []"""
    '''
    Goes through list of all video_ids, if video is in set finished, skip and move to next unfinished video
    '''
    for video_id in tqdm(video_ids):
        if video_id in finished:
            continue
        # job = pool.apply_async(process_video, (video_id, args, input_steps, transcripts, tokenizer, punct_cap_model, q))
        '''
        Call process_video here
        '''
        process_video(video_id, args, input_steps, transcripts, tokenizer, punct_cap_model, fout)
        # print('here', len(jobs))
        # jobs.append(job)
    """for job in jobs:
        job.get()
    q.put('kill')
    pool.close()
    pool.join()"""
    fout.close()
