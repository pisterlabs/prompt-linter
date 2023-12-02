#!/usr/bin/env python
# coding: utf-8

# !pip install -qqq jamo nltk jieba spacy paddlepaddle==2.5.2 -i https://mirror.baidu.com/pypi/simple torch==2.1.0 transformers[torch] scipy evaluate datasets nltk keybert sentence-transformers symspellpy autocorrect spacy sentencepiece pyspellchecker typing konlpy symspellpy-ko jamo kiwipiepy janome sudachipy sudachidict_core langdetect emoji demoji tweetnlp selenium 'git+https://github.com/SKTBrain/KoBERT.git#egg=kobert_tokenizer&subdirectory=kobert_hf'
# !python -m spacy download ko_core_news_lg
# !python -m spacy download ja_core_news_lg
# !python -m spacy download en_core_web_lg
# !python -m spacy download zh_core_web_lg

# In[1]:


import re
import gc
import math
import jamo
import nltk
import time
import jieba
import spacy
# import MeCab
import pickle
import demoji
import random
import warnings
import numpy as np
import pandas as pd
import pkg_resources
from scipy import spatial
from konlpy.tag import Okt
from tqdm.auto import tqdm
from konlpy.tag import Kkma
import jieba.posseg as pseg
from khaiii import KhaiiiApi
from konlpy.tag import Komoran
from nltk.corpus import stopwords
from spellchecker import SpellChecker
from janome.tokenizer import Tokenizer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from symspellpy import SymSpell, Verbosity
from sklearn.feature_extraction.text import TfidfVectorizer
from symspellpy_ko import KoSymSpell, Verbosity as ko_verbo
# from langdetect import detect, detect_langs
from lingua import Language, LanguageDetectorBuilder


%cd '/home/jason/Desktop/yeardream/mnc-avo'
import momory_keyword_extraction

from momory_keyword_extraction.utils import (nounify,
                                             get_posneg,
                                             get_sentences,
                                             determine_language,
                                             preprocessing_pipeline
                                             )
from momory_keyword_extraction.parsers import get_parser
from momory_keyword_extraction.preprocessing import (remove_emoji,
                                                     unicode_check,
                                                     filter_korean,
                                                     remove_overused_characters,
                                                     )
from momory_keyword_extraction.spellcheckers import get_spellchecker
from momory_keyword_extraction.constants import languages

%cd '/home/jason/Desktop/yeardream/jamo_con'
from han_util_unicode import join_jamos, split_syllables
%cd '/home/jason/Desktop/yeardream'
avo = '/home/jason/Desktop/yeardream/data'

# In[2]:


kor_text_file = pd.read_csv('./jamo_con/wordslistUnique.txt', encoding='utf-8')
kor_name_list = pd.read_csv('./data/name_list.txt', encoding='utf-8')
ko_base_dict = pd.read_csv('./jamo_con/ko_50k.txt')

okt = Okt()
kkma = Kkma()
tqdm.pandas()
komoran = Komoran()
khaiii_api = KhaiiiApi()
jieba.enable_paddle()
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
warnings.filterwarnings('ignore')
# ja_mecab = MeCab.Tagger('-0chasen')
eng_lemmatizer = WordNetLemmatizer()
ko_symspell = KoSymSpell(prefix_length=7)
nltk.download('averaged_perceptron_tagger')
sym_spell = SymSpell(max_dictionary_edit_distance=3, prefix_length=7)
ko_symspell.load_korean_dictionary(decompose_korean=True, load_bigrams=True)
freq_dict = pkg_resources.resource_filename("symspellpy", "symspell_freq_dict.txt")
sym_spell.load_dictionary(freq_dict, term_index=0, count_index=1)
languages = [Language.ENGLISH, Language.INDONESIAN, Language.KOREAN, Language.CHINESE, Language.JAPANESE, Language.TAGALOG]
lingua_detector = LanguageDetectorBuilder.from_languages(*languages).build()

ko_sw_temp = '아 휴 아이구 아이쿠 아이고 어 나 우리 저희 따라 의해 을 를 에 의 가 으로 로 에게 뿐이다 의거하여 근거하여 입각하여 기준으로 예하면 예를 들면 예를 들자면 저 소인 소생 저희 지말고 하지마 하지마라 다른 물론 또한 그리고 비길수 없다 해서는 안된다 뿐만 아니라 만이 아니다 만은 아니다 막론하고 관계없이 그치지 않다 그러나 그런데 하지만 든간에 논하지 않다 따지지 않다 설사 비록 더라도 아니면 만 못하다 하는 편이 낫다 불문하고 향하여 향해서 향하다 쪽으로 틈타 이용하여 타다 오르다 제외하고 이 외에 이 밖에 하여야 비로소 한다면 몰라도 외에도 이곳 여기 부터 기점으로 따라서 할 생각이다 하려고하다 이리하여 그리하여 그렇게 함으로써 하지만 일때 할때 앞에서 중에서 보는데서 으로써 로써 까지 해야한다 일것이다 반드시 할줄알다 할수있다 할수있어 임에 틀림없다 한다면 등 등등 제 겨우 단지 다만 할뿐 딩동 댕그 대해서 대하여 대하면 훨씬 얼마나 얼마만큼 얼마큼 남짓 여 얼마간 약간 다소 좀 조금 다수 몇 얼마 지만 하물며 또한 그러나 그렇지만 하지만 이외에도 대해 말하자면 뿐이다 다음에 반대로 반대로 말하자면 이와 반대로 바꾸어서 말하면 바꾸어서 한다면 만약 그렇지않으면 까악 툭 딱 삐걱거리다 보드득 비걱거리다 꽈당 응당 해야한다 에 가서 각 각각 여러분 각종 각자 제각기 하도록하다 와 과 그러므로 그래서 고로 한 까닭에 하기 때문에 거니와 이지만 대하여 관하여 관한 과연 실로 아니나다를가 생각한대로 진짜로 한적이있다 하곤하였다 하 하하 허허 아하 거바 와 오 왜 어째서 무엇때문에 어찌 하겠는가 무슨 어디 어느곳 더군다나 하물며 더욱이는 어느때 언제 야 이봐 어이 여보시오 흐흐 흥 휴 헉헉 헐떡헐떡 영차 여차 어기여차 끙끙 아야 앗 아야 콸콸 졸졸 좍좍 뚝뚝 주룩주룩 솨 우르르 그래도 또 그리고 바꾸어말하면 바꾸어말하자면 혹은 혹시 답다 및 그에 따르는 때가 되어 즉 지든지 설령 가령 하더라도 할지라도 일지라도 지든지 몇 거의 하마터면 인젠 이젠 된바에야 된이상 만큼 어찌됏든 그위에 게다가 점에서 보아 비추어 보아 고려하면 하게될것이다 일것이다 비교적 좀 보다더 비하면 시키다 하게하다 할만하다 의해서 연이서 이어서 잇따라 뒤따라 뒤이어 결국 의지하여 기대여 통하여 자마자 더욱더 불구하고 얼마든지 마음대로 주저하지 않고 곧 즉시 바로 당장 하자마자 밖에 안된다 하면된다 그래 그렇지 요컨대 다시 말하자면 바꿔 말하면 즉 구체적으로 말하자면 시작하여 시초에 이상 허 헉 허걱 바와같이 해도좋다 해도된다 게다가 더구나 하물며 와르르 팍 퍽 펄렁 동안 이래 하고있었다 이었다 에서 로부터 까지 예하면 했어요 해요 함께 같이 더불어 마저 마저도 양자 모두 습니다 가까스로 하려고하다 즈음하여 다른 다른 방면으로 해봐요 습니까 했어요 말할것도 없고 무릎쓰고 개의치않고 하는것만 못하다 하는것이 낫다 매 매번 들 모 어느것 어느 로써 갖고말하자면 어디 어느쪽 어느것 어느해 어느 년도 라 해도 언젠가 어떤것 어느것 저기 저쪽 저것 그때 그럼 그러면 요만한걸 그래 그때 저것만큼 그저 이르기까지 할 줄 안다 할 힘이 있다 너 너희 당신 어찌 설마 차라리 할지언정 할지라도 할망정 할지언정 구토하다 게우다 토하다 메쓰겁다 옆사람 퉤 쳇 의거하여 근거하여 의해 따라 힘입어 그 다음 버금 두번째로 기타 첫번째로 나머지는 그중에서 견지에서 형식으로 쓰여 입장에서 위해서 단지 의해되다 하도록시키다 뿐만아니라 반대로 전후 전자 앞의것 잠시 잠깐 하면서 그렇지만 다음에 그러한즉 그런즉 남들 아무거나 어찌하든지 같다 비슷하다 예컨대 이럴정도로 어떻게 만약 만일 위에서 서술한바와같이 인 듯하다 하지 않는다면 만약에 무엇 무슨 어느 어떤 아래윗 조차 한데 그럼에도 불구하고 여전히 심지어 까지도 조차도 하지 않도록 않기 위하여 때 시각 무렵 시간 동안 어때 어떠한 하여금 네 예 우선 누구 누가 알겠는가 아무도 줄은모른다 줄은 몰랏다 하는 김에 겸사겸사 하는바 그런 까닭에 한 이유는 그러니 그러니까 때문에 그 너희 그들 너희들 타인 것 것들 너 위하여 공동으로 동시에 하기 위하여 어찌하여 무엇때문에 붕붕 윙윙 나 우리 엉엉 휘익 윙윙 오호 아하 어쨋든 만 못하다 하기보다는 차라리 하는 편이 낫다 흐흐 놀라다 상대적으로 말하자면 마치 아니라면 쉿 그렇지 않으면 그렇지 않다면 안 그러면 아니었다면 하든지 아니면 이라면 좋아 알았어 하는것도 그만이다 어쩔수 없다 하나 일 일반적으로 일단 한켠으로는 오자마자 이렇게되면 이와같다면 전부 한마디 한항목 근거로 하기에 아울러 하지 않도록 않기 위해서 이르기까지 이 되다 로 인하여 까닭으로 이유만으로 이로 인하여 그래서 이 때문에 그러므로 그런 까닭에 알 수 있다 결론을 낼 수 있다 으로 인하여 있다 어떤것 관계가 있다 관련이 있다 연관되다 어떤것들 에 대해 이리하여 그리하여 여부 하기보다는 하느니 하면 할수록 운운 이러이러하다 하구나 하도다 다시말하면 다음으로 에 있다 에 달려 있다 우리 우리들 오히려 하기는한데 어떻게 어떻해 어찌됏어 어때 어째서 본대로 자 이 이쪽 여기 이것 이번 이렇게말하자면 이런 이러한 이와 같은 요만큼 요만한 것 얼마 안 되는 것 이만큼 이 정도의 이렇게 많은 것 이와 같다 이때 이렇구나 것과 같이 끼익 삐걱 따위 와 같은 사람들 부류의 사람들 왜냐하면 중의하나 오직 오로지 에 한하다 하기만 하면 도착하다 까지 미치다 도달하다 정도에 이르다 할 지경이다 결과에 이르다 관해서는 여러분 하고 있다 한 후 혼자 자기 자기집 자신 우에 종합한것과같이 총적으로 보면 총적으로 말하면 총적으로 대로 하다 으로서 참 그만이다 할 따름이다 쿵 탕탕 쾅쾅 둥둥 봐 봐라 아이야 아니 와아 응 아이 참나 년 월 일 령 영 일 이 삼 사 오 육 륙 칠 팔 구 이천육 이천칠 이천팔 이천구 하나 둘 셋 넷 다섯 여섯 일곱 여덟 아홉 령 영 이 있 하 것 들 그 되 수 이 보 않 없 나 사람 주 아니 등 같 우리 때 년 가 한 지 대하 오 말 일 그렇 위하 때문 그것 두 말하 알 그러나 받 못하 일 그런 또 문제 더 사회 많 그리고 좋 크 따르 중 나오 가지 씨 시키 만들 지금 생각하 그러 속 하나 집 살 모르 적 월 데 자신 안 어떤 내 내 경우 명 생각 시간 그녀 다시 이런 앞 보이 번 나 다른 어떻 여자 개 전 들 사실 이렇 점 싶 말 정도 좀 원 잘 통하 놓'
ko_yok = '시발 씨발 ㅅㅂ 시부레 씨바 씹팔 씨팔 씨부레 ㅆㅂ 씨파 시벌 씨벌 야발 쒸발 쉬발 씌발 싀발 슈발 찌발 쓔발 쓔팔 슈팔 쑤발 ㅆ 썅 샹 개새끼 개새 ㄱㅅㄲ 개새기 셰끼 개쉐키 개섀키 개섀끼 개세기 개쉐키 개쉬키 개시키 ^|발 C발 씨foot 씨8 sibal Tlqkf tlqkf rotoRl 병신 ㅂㅅ 빙신 븅신 븅 붕신 '
ko_stopwords = [i for i in ko_sw_temp.split()]

# In[3]:


# def determine_language(text) -> dict:
#     current_language = ''
#     lang_count = {}
#     for char in text:
#     # iterates over each character
#         language = unicode_check(char):
#         lang_count[language] += 1

#     len_char = sum(lang_count.values())
#     for key, value in lang_count.items():
#         lang_count[key] = value/sum(lang_count.values())

#     for key, value in lang_count.items():
#         if value > 0.8:
#             current_language = key
#         else:
#             continue
#     if current_language == '':
#         current_language = max(lang_count, key=lang_count.get)
#     if current_language == 'en':
#         possible_lang = detect_langs(text)
                

# In[41]:


# possible_lang = lingua_detector.detect_language_of('hello, 안녕하세요, 这是中文字 请翻译')
# possible_lang

# In[5]:


# str(possible_lang).split('.')[1]

# In[6]:


ko_word_token = {}
for row_num in range(ko_base_dict.shape[0]):
    word = ko_base_dict['word token'].iloc[row_num].split()[0]
    token = ko_base_dict['word token'].iloc[row_num].split()[1]
    ko_word_token[word] = token

# In[7]:


kor_dict = set()
for i in kor_text_file['ㄱ']:
    kor_dict.add(i)

# In[8]:


kor_name_dict = {}
kor_surname_dict = {}
for idx, word in enumerate(kor_name_list['name']):
    kor_name_dict[word[-2:]] = idx
    if '✔' not in word:
        kor_surname_dict[word[:-2]] = idx +1000000
    else:
        kor_surname_dict[word[1:-2]] = idx + 1000000

# In[9]:


main_file = pd.read_pickle(f"{avo}/2023_9_30_2023_10_08.pkl")
main_file = main_file.rename(columns={'id':'text_id', 'userId':'id'})
user_file = pd.read_pickle(f"{avo}/user.pkl")
text_filter = pd.read_pickle(f"{avo}/text_filter.pkl")
# text_filter.head(5)

# In[10]:


main_user_file = pd.merge(main_file, user_file, on='id', how='outer')
main_user_file = main_user_file.rename(columns={'emotion':'default_emotion', 'emotion2':'gpt_emotion', 'name':'user_input_emotion'})
# main_user_file.head(5)

# In[11]:


deleted_text_id = list(main_user_file[(main_user_file['deletedAt'].notnull()) &
                                 (main_user_file['updatedAt']!=main_user_file['createdAt'])]['createdAt'])
main_user_file = main_user_file[~main_user_file.isin(deleted_text_id).any(axis=1)]
df = main_user_file.sort_values(by=['createdAt', 'updatedAt'], ascending=[True, False])
df = df.drop_duplicates(subset='createdAt', keep='first')
df = df.drop(columns=['deletedAt', 'gender'])

# df.head(5)

# In[12]:


# df.info()

# In[13]:


byear = []
for value in df['birth_year']:
    if math.isnan(value):
        byear.append('-')
        continue
    elif value == '-':
        byear.append('-')
        continue
    else:
        value = int(value)
        byear.append(value)

df['birth_year'] = pd.Series(byear)
# df.head(5)

# In[14]:


# text = '내일은 10월 4일 수요일 [SEP] 내일부터 난 다이어트를시작한다 [SEP] [SEP] 아침은 바쁘니 거르고 [SEP] 점심은 학교에서 나오는 급식으로 떼우고 [SEP] 저녁은 절대 안먹는다 [SEP] [SEP] 만약 배고프면 과일 조금 [SEP]간식도절대안먹는다 [SEP] [SEP] [SEP] 진짜로 진짜로 진짜로 진짜로 살 빼야 한다' + '\n규미니' + '\n이 두 헌책은' + '\n절대' + '\n스타벅스 스벅'

cor_text = ''
for word in khaiii_api.analyze(text):
    # print(word.lex)
    # cor_text = cor_text +' '.join([str(m) for m in word.morphs]) + '\n'
    list_ = [str(m) for m in word.morphs]
    # print(list_)
    # print(word.morphs[0])
# print(cor_text)

# print([str(w).split('\t')[1].split(' + ')[0].split('/')[1] for w in khaiii_api.analyze('절대안먹겠다')][0])

# In[15]:


class Preprocessor:
    def __init__(self, df) -> None:
        self.df = df
        self.sep = '[SEP]'
    def emoji_text_cleanser(self, text):
        if text == None:
            return text
        else:
            emoji_dict = demoji.findall(text)
            for emoji in emoji_dict.keys():
                if emoji in text:
                    text = text.replace(emoji, '')
            text = text.replace('  ', ' ')
            return text.strip()
            
    def emoji_col_cleanser(self, text):
        if text == None:
            return text
        else:
            emoji_dict = demoji.findall(text)
            for emoji in emoji_dict.keys():
                if emoji in text:
                    text = text.replace(emoji, emoji_dict[emoji].split(":")[0])
            # text = text.replace('  ', ' ')
            return text.strip()

    def process_sentences(self, text):
        processed_sentences = []
        # for sentence in text:
        # 마침표와 그 뒤의 공백을 모두 제거
        pattern = r'[\.\s*]'
        processed_sentence = re.sub(pattern, '', text)
    
            # 마지막 마침표 유지
        if not processed_sentence.endswith('.'):
            processed_sentence += '.'
        # processed_sentences.append(processed_sentence)
        return processed_sentence

    def initial_joiner(self, row):
        text = row['text_cleansed']
        lang = row['language']

        def remove_misplaced_dots(text):
            # This regex pattern matches any letter (including non-English ones) followed by a full stop,
            # and it will be applied repetitively to capture consecutive occurrences.
            pattern = re.compile(r'(?<=\b\w)\.(?=\w\b^\s)')
            
            # Replace the identified patterns with an empty string
            cleaned_text = pattern.sub('', text)

            # Return the cleaned text
            return cleaned_text.strip()

        text = remove_misplaced_dots(text)
        pattern = re.compile(r'(?<=\b\w^\s)\.')
        text = pattern.sub(f' {self.sep} ', text)
        text = text.replace('.\n', f' {self.sep} ')
        text = text.replace('\n', f' {self.sep} ')
        text = text.replace('...', f' {self.sep} ')
        # text = text.replace(' .', '')

        if lang == 'KOREAN':
            tex_ = ''
            for word in text.split():
                if (word.endswith('다')) or (word.endswith('요')) or (word.endswith('음')) or (word.endswith('슴')):
                    tex_ = tex_ + word + ' ' + f' {self.sep} '
                else:
                    tex_ = tex_ + ' ' + word

            pattern = r'[^a-zA-Zㄱ-ㅣ가-힣0-9\[\]\{\}\(\)\<\>\s]'
            text_ = re.sub(pattern, '', tex_)
            try:
                s = split_syllables(text_)
                s = join_jamos(s)
            except KeyError:
                print(text)
            s = s.replace('  ', ' ')
            return s.strip()

        else:
            return text.strip()
    
    def spacing(self, row):
        text = row['each_sentence']
        language = row['language']
        first_bracket = ["(", "[", "{", "<"]
        symbs_ = ['SF', 'SP', 'SE', 'SO', 'NR', 'NNB']
        apps = ["'", '"']
        cor_list = ''
        prev_tag = ''

        if language == "KOREAN":
            for johab in khaiii_api.analyze(text):
                list_khaiii = [str(m) for m in johab.morphs]
                for words in list_khaiii:                            
                    word = words. split('/')[0]
                    tag = words.split('/')[1]
                    
                    if cor_list != '':
                        prev_word = cor_list.split()[-1]
                        prev_word_tag = [str(w).split('\t')[1].split()[0].split('/')[1] for w in khaiii_api.analyze(prev_word)][0]
                        
                        if prev_word_tag == 'XPN':
                            cor_list = cor_list + word
                            prev_tag = tag
                            continue

                        # if prev_word_tag == 'NNG' and tag == 'VV' and len(prev_word)==1:
                        #     temp = ' '.join(cor_list.split()[:-1]) + '  ' + prev_word + word[0] + '  ' + word[1:]
                        #     prev_tag = tag
                        #     continue
                            
                        
                    if prev_tag == 'NNG' and tag == 'VV':
                        cor_list = cor_list + word
                        prev_tag = tag
                        continue
                        
                    if word in first_bracket:
                        # 선행 괄호를 먼저 처리해 준다.
                        # 후행 괄호는 하단에서 처리를 해준다.
                        cor_list = cor_list + ' ' + word
                        prev_tag = tag
                        continue

                    if word in apps:
                        if str(cor_list).count(word) % 2 == 1:
                            cor_list = cor_list + word
                            prev_tag = tag
                            continue
                            
                        else:
                            cor_list = cor_list + ' ' + word
                            prev_tag = tag
                            continue

                    if prev_tag == 'NNG' and tag == 'MAG' and len(prev_word) == 1:
                        cor_list = cor_list + word + ' '
                        prev_tag = tag
                        continue
                        
                    
                        
                    if tag.startswith('J') or tag.startswith('E') or tag.startswith('XS'):
                        # 단어의 형태가 조사, 어미, 접미사일 경우 문장의 최종 단어를 불러와 자소분해를 해준 후 자소분해 된 현재 단어와 자소결합을 해줌
                        # 이다, ㄴ다, ㄴ

                        # 문장의 최종 단어를 불러와서 자소분해를 해준다.
                        s_0 = split_syllables(cor_list.split()[-1]).strip()
                        # 현재 단어를 자소분해 해준다.
                        s_1 = split_syllables (word).strip()
                        # s_0과 s_1을 합해 자소 결합을 해준다.
                        s = join_jamos (s_0 + s_1)
                        # 문장의 최종 단어를 제외한 후 자소결합 된 현재 단어 (s)를 추가하여 준다.
                        cor_list = ' '.join(cor_list.split()[:-1]) + ' ' + s
                        prev_tag = tag
                        continue
                        
                    if tag in symbs_:
                        # 후행 괄호를 포함한 어미에 붙는 기호들을 처리해 준다.
                        # 선행 괄호는 이미 위에서 처리를 했기 때문에 제외된다.
                        
                        cor_list = cor_list + word
                        prev_tag = tag
                        continue
                        
                    else:
                        cor_list = cor_list + ' ' + word
                        prev_tag = tag
                        continue
            
            cor_text = str(cor_list).replace('[ SEP ]', ' [SEP] ')                
            cor_text = str(cor_text).replace('  ', ' ').strip()
            return cor_text
        else:
            return str(text).strip()
                


    
    def py_sym_checker(self, text):
        def pyspellchecker_detector(text):
            sentence = re.sub(r'[^\w\s]', '', text)
            spell = SpellChecker()
            tokens = sentence.split() # word_tokenize(sentence)
            mis_tokens = []
            for token in spell.unknown(tokens):
                if token.isalpha():
                    mis_tokens.append(token)
            return mis_tokens

        def symspellpy_corrector(mis_tokens):
            try:
                corrected_token = {}
                for token in mis_tokens:
                    terms = sym_spell.lookup_compound(token,
                                                      max_edit_distance=2)
                    if token not in corrected_token.keys():
                        corrected_token[token] = terms[0].term
                return corrected_token

            except UnicodeDecodeError:
                return mis_tokens

        try:
            mis_tokens = []
            for word in pyspellchecker_detector(text):
                mis_tokens.append(word)

            mis_token_rep = symspellpy_corrector(mis_tokens)

            tokens = text.split()
            temp_str = []
            for token in tokens:
                if token in mis_token_rep.keys():
                    temp_str.append(mis_token_rep[token])
                else:
                    temp_str.append(token)
            string = ' '.join(temp_str)
            return string

        except UnicodeDecodeError:
            return text

    #'''
    def spell_checker(self, row):
        text = row['each_sentence']
        language = row['language']
        stopword_placeholders = []
        text_fixed = []
        if language == 'KOREAN':
            # for suggestion in ko_symspell.lookup_compound(text, max_edit_distance=2):
            #       # print(suggestion.term)
            #       text_fixed = suggestion
            # # print(text)
            word_tokens = text.split()# word_tokenize(text)

            for word in word_tokens:
                if len(word) == (1 or 2) and (word in ko_stopwords):
                    placeholder = f'PLACEHOLDER_{len(stopword_placeholders)}'
                    stopword_placeholders.append(word)
                    text_fixed.append(placeholder)
                else:
                    text_fixed.append(word)
            print('first loop done')        
            # text_for_spell_check = ' '.join(text_fixed)

            for word in tqdm(text_fixed):
                if word in ko_stopwords:
                    text_fixed.append(' [STOP] ')
                    continue
                if word.startswith('[') or word.endswith(']'):
                    # [SEP], [EMOJI], kinda
                    text_fixed.append(word)
                    continue

                if word.isalpha():
                    # if the word doesn't contain foreign word
                    # consisted by Korean chars only
                    w_fixed = self.py_sym_checker(word)
                    text_fixed.append(w_fixed)
                    continue

                if word not in ko_word_token.keys():
                    for suggestion in ko_symspell.lookup(word, max_edit_distance=2, verbosity=ko_verbo):
                        # print(suggestion)
                        text_fixed.append(suggestion.term)
                        continue

                else:
                    text_fixed.append(word)
                    
            print('second loop done')

            final_words = []
            for word in text_fixed:
                if word.startswith('PLACEHOLDER_'):
                    index = int(word.split('_')[-1])
                    final_words.append(stopword_placeholders[index])
                else:
                    final_words.append(word)
            final_text = ' '.join(final_words)
            print('third loop done')
            return final_text.strip()
            
        elif language=='ENGLISH':
            text_fixed = self.py_sym_checker(text)
            return text_fixed.strip()

        else:
            return text.strip()



    def lemmatize(self, row):
        text = row['spacing']
        lang = row['language']
        if lang == 'KOREAN':
            words = okt.pos(text)
            text_fixed = []
            for word, pos in words:
                if re.match(r'[^ㄱ-ㅣ가-힣]', word):
                    text_fixed.append(word)
                    continue
                if pos == 'Verb':
                    text_fixed.append(word[:-1] + '다')

                else:
                    text_fixed.append(word)
                
            text_f = ' '.join(text_fixed)
            return text_f

        # elif lang == 'en':
        #     word_tokens = text.split()# word_tokenize(text)
        #     # lem_words = []
        #     lem_words = [eng_lemmatizer.lemmatize(word, pos='v') for word in word_tokens]
        #     text_fixed = ' '.join(lem_words)
        #     # gc.collect()
        #     return text_fixed

        # elif lang == 'ja':
        #     # node = ja_mecab.parseToNode(text)
        #     # lemmas = []
        #     # while node.surface != '':
        #     #     lemma = node.feature.split(',')[6]
        #     #     if lemma != '*':
        #     #         lemmas.append(lemma)
        #     #     else:
        #     #         lemmas.append(node.surface)
        #     #     node = node.next
        #     return text

        # elif lang == 'zh':
        #     tokens = jieba.cut(text, cut_all=False)
        #     # gc.collect()
        #     return ' '.join(tokens)

        else:
            return text
        # gc.collect()
    def kkma_spacer(self, row):
        text = row['each_sentence']
        lang = row['language']
        pattern = r'[^a-zA-Zㄱ-ㅣ가-힣0-9]'
        text = re.sub(pattern, '', text) 
        if lang == 'KOREAN' and text != '':
            text = kkma.sentences(text)
            text = self.process_sentences(text)
            text = reconstruct_text(text)
            text = kkma.sentences(text)
            text = ' '.join(text)
            return text
        else:
            return text
    
    def lang_detector(self, row):
        text = row['text']
        user_input_emotion = row['user_input_emotion']
        
        lang_dict = lingua_detector.detect_language_of(text)
        curr_language = ''
        try:
            curr_language = str(lang_dict).split('.')[1]
        except IndexError:
            curr_language = 'unidentified'

        return curr_language
        # if lingua_detector.detect_language_of(user_input_emotion) == 'CHINESE':
        #     curr_language = 'CHINESE'
        # elif lingua_detector.detect_language_of(user_input_emotion) == 'JAPANESE':
            


    def posneg(self, emotion):
        return get_posneg(emotion)

    def run(self,
        df:pd.DataFrame,
           ) -> pd.DataFrame:
        
        print('language segmentation start')
        df['language'] = df.progress_apply(
            lambda row: self.lang_d_sentencizer(row), axis=1
        )
        print(df[df['text_id']=='m#1696336879902#84cf9609-1314-48bd-ba55-694f8aa23cc5'].iloc[0]['text'])
        print('language segmentation done\n')

        # print('korean typo cleansing start')
        # df['text_cleansed'] = df.progress_apply(
        #     lambda row: self.ko_typo_fixer(row), axis=1
        # )
        # print(df[df['text_id']=='m#1696336879902#84cf9609-1314-48bd-ba55-694f8aa23cc5'].iloc[0]['text_cleansed'])
        # print('korean typo cleansing done\n')
        
        print('emoji cleansing start\n')
        df['text_cleansed'] = df['text'].progress_apply(
            lambda x: self.emoji_text_cleanser(x)
        )
        print(df[df['text_id']=='m#1696336879902#84cf9609-1314-48bd-ba55-694f8aa23cc5'].iloc[0]['text_cleansed'])
        print('emoji cleansing done\n')

        print('korean joiner start')
        df['each_sentence'] = df.progress_apply(
            lambda row: self.initial_joiner(row), axis=1
        )
        print(df[df['text_id']=='m#1696336879902#84cf9609-1314-48bd-ba55-694f8aa23cc5'].iloc[0]['each_sentence'])
        print('korean joiner done\n')

        # print('kor spacing start')
        # df['spacing'] = df.progress_apply(
        #     lambda row: self.spacing(row), axis=1
        # )
        # print(df[df['text_id']=='m#1696336879902#84cf9609-1314-48bd-ba55-694f8aa23cc5'].iloc[0]['spacing'])
        # print('kor spacing done\n')

        print('spacing start')
        df['spacing'] = df.progress_apply(
            lambda row: self.spacing(row), axis=1
        )
        print(df[df['text_id']=='m#1696336879902#84cf9609-1314-48bd-ba55-694f8aa23cc5'].iloc[0]['spacing'])
        print('spacing done\n')

        print('lemmatizing start')
        df['lemmatized'] = df.progress_apply(
            lambda row: self.lemmatize(row), axis=1
        )
        print('lemmatizing done\n')

        # print('spell checker start')
        # df['spell_checked'] = df.progress_apply(
        #     lambda row: self.spell_checker(row), axis=1
        # )
        # print(df[df['text_id']=='m#1696336879902#84cf9609-1314-48bd-ba55-694f8aa23cc5'].iloc[0]['spell_checked'])
        # print('spell checker done\n')

        # df["each_sentence"] = df["each_sentence"].progress_apply(
        #     lambda x: self.text_cleanser(x)
        # )
        # print('text cleaning start')
        # df['each_sentence'] = df['each_sentence'].progress_apply(
        #     lambda x: self.ko_name_marker(x)
        # )

        # print('false separator start')
        # df['each_sentence'] = df.progress_apply(
        #     lambda row: self.false_separator(row), axis=1 #
        # )
        # print(df[df['text_id']=='m#1696336879902#84cf9609-1314-48bd-ba55-694f8aa23cc5'].iloc[0]['each_sentence'])
        # print('false separator done\n')
        # df['each_sentence'] = df.progress_apply(
        #     lambda row: self.ko_spellchecker(row), axis=1 # Spell Checker
        # )

        print('emoji cleansing start')
        df["user_input_emotion"] = df["user_input_emotion"].progress_apply(
            lambda x: self.emoji_col_cleanser(x) # user input emotion column cleansing
        )
        print('emoji cleansing done\n')
        print('posneg segmentation start')
        df['posneg'] = df['default_emotion'].progress_apply(
            lambda x: self.posneg(x)
        )
        print('posneg segmentation done\n')
        return  df

preprocessor = Preprocessor(df=df)

# In[16]:


df = preprocessor.run(df)
df

# In[38]:


df[df['language']=='KOREAN'].iloc[3]['spacing']

# #######################################################################################

# # In[20]:


# # print(df[df['text_id']=='m#1696336879902#84cf9609-1314-48bd-ba55-694f8aa23cc5'].iloc[0]['text'])
# df.iloc[1430]

# # In[ ]:


# # def process_sentences(sentences):
# #     processed_sentences = []
# #     for sentence in sentences:
# #         # 마침표와 그 뒤의 공백을 모두 제거
# #         processed_sentence = re.sub(r'\.\s*', '', sentence)

# #         # 마지막 마침표 유지
# #         if not processed_sentence.endswith('.'):
# #             processed_sentence += '.'
# #         processed_sentences.append(processed_sentence)
# #     return processed_sentences


# # def reconstruct_text(processed_sentences):
# #     # 문장들을 공백으로 연결하여 합침 (각 문장의 끝에는 이미 마침표가 있음)
# #     return ' '.join(processed_sentences)   

# # In[ ]:


# # def preprocessing_pipeline(text):
# #     print("\n원본 텍스트:", text)
# #     text = remove_emoji(text)
# #     print("\n이모지 제거 후:", text)
# #     text = remove_overused_characters(text)
# #     print("\n과도한 문자 제거 후:", text)
# #     text = filter_korean(text)
# #     print("\n한글 필터링 후:", text)
# #     sentences = kkma.sentences(text)
# #     print("\n문장 분리 후:", sentences)
# #     processed_sentences = process_sentences(sentences)
# #     print("\n문장별 처리 후:", processed_sentences)
# #     text = reconstruct_text(processed_sentences)
# #     print("\n문장 재구성 후:", text)
# #     final_sentences = kkma.sentences(text)z
# #     text = ' '.join(final_sentences)
# #     print("\n최종 텍스트:", text)
# #     return text

# # In[ ]:


# # text = '내일은 10월 4일 수요일\
# # 내일부터 난 다이어트를 시작한다.\
# # 아침은 바쁘니 거르고\
# # 점심은 학교에서 나오는 급식으로 떼우고\
# # 저녁은 절대 안 먹는다.\
# # 만약 배고프면 과일 조.금.\
# # 간식도절대안먹는다.'
# # # reconstruct_text(process_sentences(text))

# # preprocessing_pipeline(text)

# # In[14]:


# from tqdm import tqdm
# import re
# import pickle
# import csv

# # In[15]:


# def get_nouns(lang, text):
#     def to_tokenizer(text):
#         word_tag = []
#         for johab in khaiii_api.analyze(text):
#             list_khaiii = [str(m) for m in johab.morphs]
#             for words in list_khaiii:                            
#                 word = words. split('/')[0]
#                 tag = words.split('/')[1]
#                 word_tag.append((word, tag))
#         return word_tag
#     if lang == 'ko':
#         tagged = to_tokenizer(text)
#         nouns = [s for s, t in tagged if t in ['NNG', 'NNP']and len(s)>1]
#     elif lang == 'en':
#         tagged = to_tokenizer(text)
#         nouns = []
#     return nouns

# def tokenize(df):
#     processed_data = []
#     for row_num in tqdm(range(df.shape[0])):
#         lang = df.iloc[row_num]['language']
#         sent = df.iloc[row_num]['spacing']
#         if lang == 'ko':
#             processed_data.append(get_nouns(sent))
#         else:
#             processed_data.append('-')
#     return processed_data

# def save_processed_data(processed_data):
#     with open('tokenized_data_avo.csv', 'w', newline='', encoding='utf-8') as f:
#         writer = csv.writer(f)
#         for data in processed_data:
#             writer.writerow(data)


# t_df = tokenize(df)
# save_processed_data(t_df)
    

# # In[16]:


# t_df

# # In[17]:


# !pip install gensim

# from gensim.models.ldamodel import LdaModel
# from gensim.models.callbacks import CoherenceMetric
# from gensim import corpora
# from gensim.models.callbacks import PerplexityMetric

# import logging
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# # In[18]:


# dict = corpora.Dictionary(t_df)

# # In[ ]:



