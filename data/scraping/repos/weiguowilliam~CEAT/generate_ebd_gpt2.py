import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import torch

# import spacy
from transformers import BertModel, BertTokenizer
from transformers import GPT2Tokenizer, GPT2LMHeadModel,GPT2Model
# from allennlp.commands.elmo import ElmoEmbedder
from transformers import OpenAIGPTTokenizer, OpenAIGPTModel
import logging
logging.getLogger('transformers.tokenization_utils').disabled = True
import numpy as np
import json
import pickle
import datetime
# import spacy
# from allennlp.commands.elmo import ElmoEmbedder
torch.cuda.is_available()


tokenizer_gpt2 = GPT2Tokenizer.from_pretrained('gpt2')
model_gpt2 = GPT2LMHeadModel.from_pretrained('gpt2', output_hidden_states=True)
model_gpt2.eval()
model_gpt2.to('cuda')

tokenizer_bert = BertTokenizer.from_pretrained('bert-base-cased')
model_bert = BertModel.from_pretrained('bert-base-cased')
model_bert.eval()
model_bert.to('cuda')

tokenizer_gpt = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
model_gpt = OpenAIGPTModel.from_pretrained('openai-gpt')
model_gpt.eval()
model_gpt.to('cuda')

# weat 1
flowers = ['aster', 'clover', 'hyacinth', 'marigold', 'poppy', 'azalea', 'crocus', 'iris', 'orchid', 'rose', 'bluebell', 'daffodil', 'lilac', 'pansy', 'tulip', 'buttercup', 'daisy', 'lily', 'peony', 'violet', 'carnation', 
'magnolia', 'petunia', 'zinnia','gladiola'] #'gladiola' deleted since it not appear
insects = ['ant', 'caterpillar', 'flea', 'locust', 'spider', 'bedbug', 'centipede', 'fly', 'maggot', 'tarantula',
'bee', 'cockroach', 'gnat', 'mosquito', 'termite', 'beetle', 'cricket', 'hornet', 'moth', 'wasp', 
'dragonfly', 'horsefly', 'roach', 'weevil','blackfly'] # 'blackfly' deleted for sysmetric since it only appears 1 time.
pleasant = ['caress', 'freedom', 'health', 'love', 'peace', 'cheer', 'friend', 'heaven', 'loyal', 'pleasure', 'diamond', 'gentle', 'honest', 
'lucky', 'rainbow', 'diploma', 'gift', 'honor', 'miracle', 'sunrise', 'family',
'happy', 'laughter', 'paradise', 'vacation']
unpleasant = ['abuse', 'crash', 'filth', 'murder', 'sickness', 'accident', 'death', 'grief', 'poison', 'stink',
'assault', 'disaster', 'hatred', 'pollute', 'tragedy', 'divorce', 'jail', 'poverty', 'ugly', 'cancer', 'kill', 'rotten',
'vomit', 'agony', 'prison']

#weat 2
instruments = ['bagpipe', 'cello', 'guitar', 'lute', 'trombone', 'banjo', 'clarinet', 'harmonica', 'mandolin',
'trumpet', 'bassoon', 'drum', 'harp', 'oboe', 'tuba', 'bell', 'fiddle', 'harpsichord', 'piano', 'viola', 'bongo',
'flute', 'horn', 'saxophone', 'violin']
weapons = ['arrow', 'club', 'gun', 'missile', 'spear', 'axe', 'dagger', 'harpoon', 'pistol', 'sword', 'blade',
'dynamite', 'hatchet', 'rifle', 'tank', 'bomb', 'firearm', 'knife', 'shotgun', 'teargas', 'cannon', 'grenade',
'mace', 'slingshot', 'whip']
pleasant, unpleasant

#weat 3
#weat 3
european_3 = ['Adam', 'Harry', 'Roger', 'Alan', 
'Ryan', 'Andrew',  'Jack', 'Matthew', 'Stephen', 'Brad', 'Greg' , 'Paul', 
'Jonathan', 'Peter',  'Amanda', 'Courtney',  'Melanie', 'Katie', 'Kristin', 'Nancy', 'Stephanie', 
'Ellen', 'Lauren', 'Colleen', 'Emily', 'Megan', 'Rachel','Betsy','Justin','Frank','Josh','Heather'] #delte random: 'Betsy','Justin','Frank','Josh','Heather'

african_3 = [ 'Alonzo',   'Theo', 'Alphonse', 'Jerome',
'Leroy',  'Torrance', 'Darnell', 'Lamar', 'Lionel', 'Tyree', 'Deion', 'Lamont', 'Malik',
'Terrence', 'Tyrone',  'Lavon', 'Marcellus',  'Wardell', 'Nichelle',
 'Ebony',  'Shaniqua',  'Jasmine',
'Tanisha', 'Tia',  'Latoya', 'Yolanda',  'Yvette','Jamel','Shereen', 'Latisha','Lakisha','Malika'] #delete 'Jamel','Shereen', 'Latisha','Lakisha','Malika',  because not appear
pleasant_3 = [ 'caress', 'freedom', 'health', 'love', 'peace', 'cheer', 'friend', 'heaven', 'loyal', 'pleasure', 
'diamond', 'gentle', 'honest', 'lucky', 'rainbow', 'diploma', 'gift', 'honor', 'miracle', 'sunrise', 'family', 'happy', 'laughter', 'paradise', 'vacation']

unpleasant_3 = ['abuse', 'crash', 'filth', 'murder', 'sickness', 'accident', 'death', 'grief', 'poison', 'stink', 'assault', 'disaster', 
'hatred', 'pollute', 'tragedy', 'bomb', 'divorce', 'jail', 'poverty', 'ugly', 'cancer', 'evil', 'kill', 'rotten', 'vomit'] 
# bomb, evil in 3 not in 1, agony prison in 1 not in 3. #pleasant 3 = pleasant 1

#weat 4
european_4 = ['Brad', 'Brendan', 'Geoffrey', 'Greg', 'Brett',  'Matthew', 
'Todd', 'Allison', 'Anne', 'Carrie', 'Emily', 'Jill', 'Laurie',  'Meredith', 'Sarah','Neil'] #delete 'Neil' sysmetric
african_4 = ['Darnell', 'Hakim', 'Jermaine', 'Kareem', 'Jamal', 'Leroy', 'Rasheed',
 'Tyrone', 'Aisha', 'Ebony', 'Keisha', 'Kenya',   'Latoya', 'Tamika',
'Tanisha','Lakisha'] #delete Lakisha because not appear
pleasant_3, unpleasant_3

#weat 5
european_4, african_4
pleasant_5 = ['joy', 'love', 'peace', 'wonderful', 'pleasure', 'friend', 'laughter', 'happy']
unpleasant_5 = ['agony', 'terrible', 'horrible', 'nasty', 'evil', 'war', 'awful', 'failure']

#weat  6
male = ['John', 'Paul', 'Mike', 'Kevin', 'Steve', 'Greg', 'Jeff', 'Bill']
female = ['Amy', 'Joan', 'Lisa', 'Sarah', 'Diana', 'Kate', 'Ann', 'Donna']
career = ['executive', 'management', 'professional', 'corporation', 'salary', 'office', 'business', 'career']
family = ['home', 'parents', 'children', 'family', 'cousins', 'marriage', 'wedding', 'relatives']

#weat 7
math = ['math', 'algebra', 'geometry', 'calculus', 'equations', 'computation', 'numbers', 'addition']
arts = ['poetry', 'art', 'dance', 'literature', 'novel', 'symphony', 'drama', 'sculpture']
male_term = ['male', 'man', 'boy', 'brother', 'he', 'him', 'his', 'son']
female_term = ['female', 'woman', 'girl', 'sister', 'she', 'her', 'hers', 'daughter']

#weat 8
science = ['science', 'technology', 'physics', 'chemistry', 'Einstein', 'NASA', 'experiment', 'astronomy']
arts_8 = ['poetry', 'art', 'Shakespeare', 'dance', 'literature', 'novel', 'symphony', 'drama']
male_term_8 = ['brother', 'father', 'uncle', 'grandfather', 'son', 'he', 'his', 'him']
female_term_8 = ['sister', 'mother', 'aunt', 'grandmother', 'daughter', 'she', 'hers', 'her']

#weat 9
mental_disease = ['sad', 'hopeless', 'gloomy', 'tearful', 'miserable', 'depressed']
physical_disease = ['sick', 'illness', 'influenza', 'disease', 'virus', 'cancer']
temporary = ['impermanent', 'unstable', 'variable', 'fleeting', 'short', 'brief', 'occasional' ]
permanent = [ 'stable', 'always', 'constant', 'persistent', 'chronic', 'prolonged', 'forever']

#weat 10
young_name = ['Tiffany', 'Michelle', 'Cindy', 'Kristy', 'Brad', 'Eric', 'Joey', 'Billy']
old_name = [ 'Ethel', 'Bernice', 'Gertrude', 'Agnes', 'Cecil', 'Wilbert', 'Mortimer', 'Edgar']
pleasant_5, unpleasant_5

african_female = ['Yvette','Aisha','Malika','Latisha','Keisha','Tanisha',
'Tamika', 'Yolanda','Nichelle','Latoya','Lakisha','Shereen','Shaniqua',
'Jasmine','Tia'][0:12]
# delete:  Kenya

african_male = ['Lionel', 'Wardell',  'Jamel', 'Marcellus',
  'Jamal', 'Leroy',  'Tyrone', 'Alphonse'
, 'Hakim', 'Terrence', 'Jerome', 'Alonzo'][0:12]
#delete: Deion, Ebony, Kareem,Lamar,Lavon,Malik,Rasheed,Jermaine,
# Tyree,Lamont,Darnell,Torrance,Theo

european_female = ['Melanie','Colleen','Ellen','Emily','Sarah','Rachel',
'Carrie','Stephanie','Megan','Nancy', 'Katie','Heather','Betsy',
'Kristin'][0:12]
#delete: Amanda

european_male = ['Frank',   'Roger', 'Neil', 'Geoffrey',
 'Brad', 'Stephen', 'Peter', 'Josh', 'Andrew',  'Jack', 
'Matthew', 'Jonathan',  'Greg',
  'Justin', 'Alan',    'Adam',
    'Harry',  'Paul'][0:12]
#delete: Lauren,Jill,Brendan,Meredith,Allison,Todd,Ryan,Courtney,Laurie,Brett,Anne

mexican_female = ['Maria','Yesenia','Adriana','Liset','Mayra','Alma',
'Carolina','Iliana','Sonia',
                    'Karina','Alejandra','Brenda','Vanessa','Diana'][0:12]
# delete: Ana
mexican_male = ['Jesús','Rigoberto','César','Rogelio','José','Pedro',
'Antonio','Alberto','Alejandro',
                  'Alfredo','Juan','Miguel','Ricardo'][0:12]

af_bias = ['loud','ghetto','unrefined','athletic','unintelligent']+['bigbutt','overweight','confident','darkskinned','promiscuous','unfeminine','aggressive','chicken']
em_bias_foraf = ['rich', 'intelligent', 'arrogant', 'status', 'blond', 'racist', 'American', 'leader', 'privileged', 'attractive', 'tall', 'sexist', 'successful']

af_unique_bias = ['bigbutt','overweight','confident','darkskinned','promiscuous','unfeminine','aggressive','chicken']
em_unique_bias_foraf = ['rich', 'tall', 'intelligent', 'arrogant', 'successful', 'status', 'blond', 'racist']

lf_bias = ['feisty','curvy', 'loud',  'cook', 'darkskinned', 'uneducated', 'hardworker' ,'promiscuous','unintelligent','short','sexy', 'maids']
em_bias_forlf = ['rich', 'intelligent', 'arrogant', 'status', 'blond', 'racist', 'American', 'leader', 'privileged',  'tall', 'sexist', 'successful']

lf_unique_bias = ['feisty','curvy','cook','promiscuous','sexy','maids']
em_unique_bias_forlf = ['rich', 'tall', 'intelligent', 'assertive', 'arrogant', 'successful']

def short_sen(sen,wd):
    """
    shorten the raw comment, take only 9 words including the target word
    """
    wds = sen.split()
    wd_idx = wds.index(wd)
    if len(wds) >=9:
        if wd_idx < 4:
            wds_used = wds[:9]
        elif (len(wds) - wd_idx - 1 < 4):
            wds_used = wds[-9:]
        else:
            wds_used = wds[(wd_idx-4):(wd_idx+4)]
        new_sen = ' '.join(wds_used)
    else:
        new_sen = sen
    return new_sen



def gpt2(wd_lst,out_name):
    # load
    sen_dict = pickle.load(open('sen_dic_2.pickle','rb'))
    wd_idx_dict = {wd:[] for wd in wd_lst}
    out_dict = {wd:[] for wd in wd_lst}
    # error_sen_dict = {wd:[] for wd in sen_dict}

    # generate wd index dictionary
    for wd in wd_lst:
        current_idx = torch.tensor(tokenizer_gpt2.encode(wd,add_prefix_space=True)).unsqueeze(0).tolist()[0]
        wd_idx_dict[wd] = current_idx
    
    # generate embeddings
    i = 0
    for wd in wd_lst:
        target = wd_idx_dict[wd][-1]
        for idx,sen in enumerate(sen_dict[wd]):
            i += 1
            if idx == 1000:
                break
            if i%2000 == 0:
                now = datetime.datetime.now()
                print(now.strftime("%Y-%m-%d %H:%M:%S"))
                print(str(i)+' finished.')
            # try:
            #     input_ids = torch.tensor(tokenizer_gpt2.encode(sen,add_prefix_space=True)).unsqueeze(0) 
            #     input_ids = input_ids.to('cuda')
            #     exact_idx = input_ids.tolist()[0].index(target)
            #     outputs = model_gpt2(input_ids)
            #     exact_state_vector = outputs[2][-1][0,int(exact_idx),:].cpu().detach().numpy() 
            #     out_dict[wd].append(exact_state_vector)
            # except:
                # error_sen_dict[wd].append(sen)
            sen = short_sen(sen,wd)            
            input_ids = torch.tensor(tokenizer_gpt2.encode(sen,add_prefix_space=True)).unsqueeze(0) 
            input_ids = input_ids.to('cuda')
            exact_idx = input_ids.tolist()[0].index(target)
            outputs = model_gpt2(input_ids)
            exact_state_vector = outputs[2][-1][0,int(exact_idx),:].cpu().detach().numpy() 
            out_dict[wd].append(exact_state_vector)            
    n = 'gpt2_'+out_name+'.pickle'
    pickle.dump(out_dict,open(n,'wb'))

    # pickle.dump(error_sen_dict,open('gpt2_error_sen.pickle','wb'))



lst = flowers + insects + pleasant + unpleasant

# now = datetime.datetime.now()
# print(now.strftime("%Y-%m-%d %H:%M:%S"))
# bert(lst,'weat1')
# print("bert finish")

# now = datetime.datetime.now()
# print(now.strftime("%Y-%m-%d %H:%M:%S"))
# gpt(lst,'weat1')
# print("gpt finish")

now = datetime.datetime.now()
print(now.strftime("%Y-%m-%d %H:%M:%S"))
gpt2(lst,'weat1')
print("gpt2 finish")

# now = datetime.datetime.now()
# print(now.strftime("%Y-%m-%d %H:%M:%S"))
# elmo(lst,'weat1')
# print("elmo finish")