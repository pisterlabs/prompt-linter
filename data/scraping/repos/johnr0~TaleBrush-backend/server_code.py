from flask import request, url_for
from flask_api import FlaskAPI, status, exceptions
from flask_cors import CORS, cross_origin
import torch
import json

import numpy as np
import torch

from modeling_gptneo import GPTNeoForCausalLM
from modeling_gpt2 import GPT2LMHeadModel


from transformers import (
    GPTNeoConfig,
    GPT2Config,
    GPT2Tokenizer
)
import transformers

from nltk import sent_tokenize
import nltk
nltk.download('punkt')

### Loading the model
code_desired = "true"
code_undesired = "false"
model_type = 'gpt2'
gen_type = "gedi"
gen_model_name_or_path = "EleutherAI/gpt-neo-2.7B"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_CLASSES = {"gptneo": (GPTNeoConfig, GPTNeoForCausalLM, GPT2Tokenizer), "gpt2": (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),}
config_class_n, model_class_n, tokenizer_class_n = MODEL_CLASSES["gptneo"]
config_class_2, model_class_2, tokenizer_class_2 = MODEL_CLASSES["gpt2"]
tokenizer = tokenizer_class_n.from_pretrained('EleutherAI/gpt-neo-2.7B', do_lower_case=False, additional_special_tokens=['[Prompt]'])

model = model_class_n.from_pretrained(gen_model_name_or_path, load_in_half_prec=True)
model = model.to(device)
model = model.float()
model.config.use_cache=True
model.resize_token_embeddings(len(tokenizer))

gedi_model_name_or_path = 'fortune_gedi'
gedi_model = model_class_2.from_pretrained(gedi_model_name_or_path)
gedi_model.to(device)
gedi_model.resize_token_embeddings(len(tokenizer))
gedi_model.resize_token_embeddings(50258)
wte = gedi_model.get_input_embeddings()
wte.weight.requires_grad=False
wte.weight[len(tokenizer)-1, :]= wte.weight[len(tokenizer)-2, :]
gedi_model.set_input_embeddings(wte)

embed_cont = torch.load('./result_embedding_cont')
embed_infill_front = torch.load('./result_embedding_infill_front')
embed_infill_back = torch.load('./result_embedding_infill_back')
embed_recognition = torch.load('./result_embedding_recognition')
recognition_score = torch.load('./recog_score')
model.set_input_embeddings(embed_cont.wte)

# setting arguments for generation
#max generation length
gen_length = 40
#omega from paper, higher disc_weight means more aggressive topic steering
disc_weight = 30
#1 - rho from paper, should be between 0 and 1 higher filter_p means more aggressive topic steering
filter_p = 0.8
#tau from paper, preserves tokens that are classified as correct topic
target_p = 0.8
#hyperparameter that determines class prior, set to uniform by default
class_bias = 0

if gen_length>1024:
  length = 1024
else:
  length = gen_length

def cut_into_sentences(text, do_cleanup=True):
    """
    Cut text into sentences. \n are also regarded as a sentence.
    :param do_cleanup: if True, do cleanups.
    :param text: input text.
    :return: sentences.
    """
    all_sentences = []
    # print(text)
    # sentences_raw = text.split("\n")
    
    text = text.replace("[Prompt] [Prompt] [Prompt] [Prompt] ", "[Prompt] [Prompt] [Prompt] ")
    sentences_raw = text.split('[Prompt] [Prompt] [Prompt]')
    text = sentences_raw[len(sentences_raw)-1]
    text = text.replace("Start:", " ")
    text = text.replace("Characters:", " ")
    text = text.replace("Story after start:", " ")
    sentences_raw = [text.replace("\n", " ")]
    result = []

    for item in sentences_raw:
        sentence_in_item = sent_tokenize(item)
        for item2 in sentence_in_item:
            all_sentences.append(item2.strip())

    if do_cleanup:
        for item in all_sentences:
            item = item.replace('<|endoftext|>', '')
            if len(item) > 2:
                result.append(item)
    else:
        result = all_sentences
    return result



def generate_one_sentence(sentence, control, length=50, disc_weight=30, temperature=0.8, gpt3_id=None):
    """
    Generate one sentence based on input data.
    :param sentence: (string) context (prompt) used.
    :param topic: (dict) {topic: weight, topic:weight,...} topic that the sentence need to steer towards.
    :param extra_args: (dict) a dictionary that certain key will trigger additional functionality.
        disc_weight: Set this value to use a different control strength than default.
        get_gen_token_count: Return only how many tokens the generator has generated (for debug only).
    :return: sentence generated, or others if extra_args are specified.
    """
    secondary_code = control

    if sentence == "":
        print("Prompt is empty! Using a dummy sentence.")
        sentence = "."

    # Specify prompt below
    prompt = sentence

    # Calculate oroginal input length.
    length_of_prompt = len(sentence)

    start_len = 0
    text_ids = tokenizer.encode(prompt)
    length_of_prompt_in_tokens = len(text_ids)
    # print('text ids', text_ids)
    
    encoded_prompts = torch.LongTensor(text_ids).unsqueeze(0).to(device)

    if type(control) is str:
        multi_code = tokenizer.encode(secondary_code)
    elif type(control) is dict:
        multi_code = {}
        for item in secondary_code:
            encoded = tokenizer.encode(item)[0]  # only take the first one
            multi_code[encoded] = secondary_code[item]
    else:
        raise NotImplementedError("topic data type of %s not supported... Supported: (str,dict)" % type(control))

    # If 1, generate sentences towards a specific topic.
    attr_class = 1
    print(multi_code)

    if int(control)!=-1:
      if gpt3_id is None:
        generated_sequence = model.generate(input_ids=encoded_prompts,
                                                  pad_lens=None,
                                                  max_length=length + length_of_prompt_in_tokens,
                                                  top_k=None,
                                                  top_p=None,
                                                  repetition_penalty=1.2,
                                                  rep_penalty_scale=10,
                                                  eos_token_ids=tokenizer.eos_token_id,
                                                  pad_token_id=tokenizer.eos_token_id,
                                                  bad_token_ids = tokenizer.all_special_ids,
                                                  do_sample=True,
                                                  temperature = temperature,
                                                  penalize_cond=True,
                                                  gedi_model=gedi_model,
                                                  tokenizer=tokenizer,
                                                  disc_weight=disc_weight,
                                                  filter_p=filter_p,
                                                  target_p=target_p,
                                                  class_bias=class_bias,
                                                  attr_class=attr_class,
                                                  code_0=code_undesired,
                                                  code_1=code_desired,
                                                  multi_code=multi_code,
                                                  )
      else: 
        generated_sequence = model.generate(input_ids=encoded_prompts,
                                                  pad_lens=None,
                                                  max_length=length + length_of_prompt_in_tokens,
                                                  top_k=None,
                                                  top_p=None,
                                                  repetition_penalty=1.2,
                                                  rep_penalty_scale=10,
                                                  eos_token_ids=tokenizer.eos_token_id,
                                                  pad_token_id=tokenizer.eos_token_id,
                                                  bad_token_ids = tokenizer.all_special_ids,
                                                  do_sample=True,
                                                  temperature = temperature,
                                                  penalize_cond=True,
                                                  gedi_model=gedi_model,
                                                  tokenizer=tokenizer,
                                                  disc_weight=disc_weight,
                                                  filter_p=filter_p,
                                                  target_p=target_p,
                                                  class_bias=class_bias,
                                                  attr_class=attr_class,
                                                  code_0=code_undesired,
                                                  code_1=code_desired,
                                                  multi_code=multi_code,
                                                  gpt3_api_key=gpt3_id,
                                                  )
      text = tokenizer.decode(generated_sequence.tolist()[0])
    else:
      if gpt3_id is None:
        generated_sequence = model.generate(input_ids=encoded_prompts,
                                                  pad_lens=None,
                                                  max_length=length + length_of_prompt_in_tokens,
                                                  top_k=None,
                                                  top_p=None,
                                                  repetition_penalty=1.2,
                                                  rep_penalty_scale=10,
                                                  eos_token_ids=tokenizer.eos_token_id,
                                                  pad_token_id=tokenizer.eos_token_id,
                                                  bad_token_ids = tokenizer.all_special_ids,
                                                  do_sample=True,
                                                  temperature = temperature, 
                                                  penalize_cond=True,
                                                  gedi_model=None,
                                                  tokenizer=tokenizer,
                                                  disc_weight=disc_weight,
                                                  class_bias=class_bias,
                                                  attr_class=attr_class,
                                                  )
        text = tokenizer.decode(generated_sequence.tolist()[0])
        

        
      else:
        import openai
        openai.api_key = gpt3_id
        completion = openai.Completion()
        response = completion.create(prompt=prompt,
                                 engine="curie",
                                 max_tokens=length,
                                 temperature=temperature,)
        text = response["choices"][0]["text"]

    
    text = cut_into_sentences(text)
    if len(text) == 0:
        print("Warning! No text generated.")
        return ""
    all_gen_text = text[0]
    return all_gen_text



import numpy as np

def continuing_generation(prompts, generation_controls, characters, temperatures, gpt3_id=None, disc_weight=30):
  """
  Explanations on controls
  prompts: The prompt to be input. This is a list of sentences. 
  generation_controls: Generation control in the list. If no control is given, -1 is given.
  
  """
  model.set_input_embeddings(embed_cont)
  prompts = list(prompts)
  generated = []

  character_prepend = '[Prompt][Prompt][Prompt]'
  for idx, character in enumerate(characters):
    if idx==0:
      character_prepend = character_prepend+character
    else:
      character_prepend = character_prepend+' '+character
    if idx != len(characters)-1:
      character_prepend = character_prepend + ','

  prompt_start_idx = 0
  for c_idx, generation_control in enumerate(generation_controls):
    
    temperature = temperatures[c_idx]
    while True:
      prompt_postpend = '[Prompt][Prompt][Prompt]'
      # prompt_postpend = 'Story: '

      for i in range(prompt_start_idx, len(prompts)):
        prompt_postpend = prompt_postpend + prompts[i]
        if i != len(prompts)-1:
          prompt_postpend = prompt_postpend + ' '
          # continue
        else:
          prompt_postpend = prompt_postpend
      
      prompt_input = prompt_postpend+character_prepend+ '[Prompt][Prompt][Prompt]'
      prompt_encoded = tokenizer.encode(prompt_input)
      length_of_prompt_in_tokens = len(prompt_encoded)
      if length_of_prompt_in_tokens>2048:
        prompt_start_idx = prompt_start_idx + 1
      else:
        break
    print(prompt_input, generation_control)
    gen_sent = generate_one_sentence(prompt_input, generation_control, temperature=temperature, gpt3_id=gpt3_id, disc_weight=disc_weight)
    prompts.append(gen_sent)
    generated.append(gen_sent)
  
  for gen in generated:
    print('gen:', gen)
    print()
  return generated



def infilling_generation(pre_prompts, post_prompts, generation_controls, characters, temperatures, is_front, gpt3_id=None, disc_weight=30):
  """
  Explanations on controls
  prompts: The prompt to be input. This is a list of sentences. 
  generation_controls: Generation control in the list. If no control is given, -1 is given.
  
  """

  pre_prompts = list(pre_prompts)
  post_prompts = list(post_prompts)
  right = ''
  for idx, pp in enumerate(post_prompts):
    right = right + pp
    if idx!=len(post_prompts)-1:
      right = right + ' '
  left = ''
  for idx, pp in enumerate(pre_prompts):
    left = left + pp
    if idx!=len(post_prompts)-1:
      left = left + ' '
  generated = ['']*len(generation_controls)

  # gen_counter = 0
  for gen_counter in range(len(generation_controls)):
    if is_front:
      generation_control = generation_controls[int(gen_counter/2)]
      temperature = temperatures[int(gen_counter/2)]
      model.set_input_embeddings(embed_infill_front)
      prompt_input = '[Prompt][Prompt][Prompt]'+right+'[Prompt][Prompt][Prompt]'+left+'[Prompt][Prompt][Prompt][Prompt]'
      
      gen_sent = generate_one_sentence(prompt_input, generation_control, temperature=temperature, gpt3_id=gpt3_id, disc_weight=disc_weight)
      generated[int(gen_counter/2)] =gen_sent
      print(gen_sent)
      left = left + ' ' + gen_sent
    else:
      generation_control = generation_controls[len(generated)-1-int(gen_counter/2)]
      temperature = temperatures[len(generated)-1-int(gen_counter/2)]
      model.set_input_embeddings(embed_infill_back)
      prompt_input = '[Prompt][Prompt][Prompt]'+left+'[Prompt][Prompt][Prompt]'+right + '[Prompt][Prompt][Prompt][Prompt]' 
      gen_sent = generate_one_sentence(prompt_input, generation_control, temperature=temperature, gpt3_id=gpt3_id, disc_weight=disc_weight)
      generated[len(generated)-1-int(gen_counter/2)] =gen_sent
      print(gen_sent)
      right = gen_sent+' '+right

  for gen in generated:
    print('gen', gen)
    print()
  return generated

def recognize_sentence_fortune(pre_context, character, target_sentence):
  rec_input = "[Prompt][Prompt][Prompt]"+pre_context+"[Prompt][Prompt][Prompt]"+character+"[Prompt][Prompt][Prompt]"+target_sentence
  
  with torch.no_grad():
    model.set_input_embeddings(embed_recognition)
    tokenized_input = tokenizer.encode(rec_input)
    tokenized_input = torch.LongTensor(tokenized_input).unsqueeze(0).to(device)
    output = model.transformer(tokenized_input)
    op= output[0].type(torch.half)
    
    # op=output[0].type(torch.FloatTensor).to(device)
    logits = recognition_score(op)
    
    to_return = float(logits[0][len(tokenized_input[0])-1][0])
    if to_return > 1:
      to_return = 1
    elif to_return <0:
      to_return = 0
    return to_return


app = FlaskAPI(__name__)
# run_with_ngrok(app)
CORS(app, resources={r"/*": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'

# Below is temporary function with sentiment analysis.
# Hence, it needs to be updated later.
@app.route('/labelSentence', methods=['GET', 'POST'])
@cross_origin(origin='http://10.168.233.218:7082',headers=['Content-Type'])
def sentence_analysis():
    if request.method == 'POST':
        print(request.data)
        sentence = request.data['sentence']
        pre_context = request.data['pre_context']
        character = request.data['character']
        # print(images, group_model, l2t, dec)

        value = recognize_sentence_fortune(pre_context, character, sentence)
        value = value * 100

        return {'value': value}

@app.route('/continuingGeneration', methods=['GET', 'POST'])
@cross_origin(origin='http://10.168.233.218:7082',headers=['Content-Type'])
def continuingGeneration():
  if request.method == 'POST':
      pre_context = json.loads(request.data['pre_context'])
      controls = json.loads(request.data['controls'])
      characters = json.loads(request.data['characters'])
      temperature = json.loads(request.data['temperature'])
      print(pre_context)
      print(controls)
      print(characters)
      print(temperature)

      # TODO update below
      generated = continuing_generation(pre_context, controls, characters, temperature, gpt3_id=None, disc_weight=30)

      # generated = ['This is a generated sentence'] * len(controls)
      values = []
      for gen in generated:
        pre_context_concat = ''
        # start_id = 0
        # start_id = len(pre_context)-2
        # if start_id<0:
        #   start_id=0
        # for idx in range(start_id, len(pre_context)):
        #   pre_context_concat = pre_context_concat + pre_context[idx]
        value = recognize_sentence_fortune(pre_context_concat, characters[0], gen)
        pre_context.append(gen)
        values.append(value*100)

      return {'generated': json.dumps(generated), 'values': json.dumps(values)}

@app.route('/infillingGeneration', methods=['GET', 'POST'])
@cross_origin(origin='http://10.168.233.218:7082',headers=['Content-Type'])
def infillingGeneration():
  if request.method == 'POST':
      pre_context = json.loads(request.data['pre_context'])
      post_context = json.loads(request.data['post_context'])
      controls = json.loads(request.data['controls'])
      characters = json.loads(request.data['characters'])
      temperature = json.loads(request.data['temperature'])
      is_front = request.data['is_front']
      print(pre_context)
      print(post_context)
      print(controls)
      print(characters)
      print(temperature)

      # TODO update below
      generated = infilling_generation(pre_context, post_context, controls, characters, temperature, is_front, gpt3_id=None, disc_weight=30)

      # generated = ['This is a generated sentence'] * len(controls)
      # it needs to be updated
      values = sentences_analysis(generated)

      return {'generated': json.dumps(generated), 'values': json.dumps(values)}

if __name__=="__main__":
    app.run(host='0.0.0.0', port=11080)