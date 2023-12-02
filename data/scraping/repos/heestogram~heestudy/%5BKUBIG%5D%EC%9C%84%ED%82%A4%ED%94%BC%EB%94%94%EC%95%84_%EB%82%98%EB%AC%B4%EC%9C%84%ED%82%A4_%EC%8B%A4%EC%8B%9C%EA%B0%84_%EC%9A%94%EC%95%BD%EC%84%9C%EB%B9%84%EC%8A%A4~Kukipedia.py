

import torch
import wikipediaapi
import pandas as pd
import re
from konlpy.tag import Okt
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import gensim
from gensim.models import LdaModel
import pyLDAvis.gensim
from gensim.models.coherencemodel import CoherenceModel

from transformers import PreTrainedTokenizerFast
from transformers import BartForConditionalGeneration

digit_tokenizer = PreTrainedTokenizerFast.from_pretrained('./digit_tokenizer')
digit_model = BartForConditionalGeneration.from_pretrained('./digit_model')
gogamza_tokenizer = PreTrainedTokenizerFast.from_pretrained('./gogamza_tokenizer')
gogamza_model = BartForConditionalGeneration.from_pretrained('./gogamza_model')
ainize_tokenizer = PreTrainedTokenizerFast.from_pretrained('./ainize_tokenizer')
ainize_model = BartForConditionalGeneration.from_pretrained('./ainize_model')

wiki=wikipediaapi.Wikipedia('ko')


class Kukipedia():

  def __init__(self, target_title):
    self.section_dict = {}
    self.target_title = target_title
    self.model_dict = {'digit': self.digit,
                  'gogamza': self.gogamza,
                  'ainize': self.ainize}


  def sections_dict(self, sections):
    """
    섹션들을 딕셔너리로 모아주는 함수
    """
    for s in sections:
      self.section_dict[s.title] = s.text
      self.sections_dict(s.sections)
    return self.section_dict


  def del_unnecessary_section(self, dictionary):
    """
    섹션 중에는 내용이 없는 것도 있고, 외부링크, 같이보기처럼 쓸모 없는 것도 있다. 
    그러한 섹션은 요약이 무의미하니 삭제하는 함수
    """
    no_contents = []
    reference_etc_list = ['각주', '같이 보기', '외부 링크'] # 요약의 효용이 없는 섹션
    for i in dictionary.keys():
      if not bool(dictionary[i]):
        no_contents.append(i)
      elif i in reference_etc_list:
        no_contents.append(i)
    for j in no_contents:
      del dictionary[j]
    return dictionary


  def make_wordcloud_entire(self, backgroundcolor='white', width=600, height=400):
    """
    문서 전체의 텍스트로 워드클라우드 만드는 함수
    요약문으로 워드클라우드 만드는 함수(make_wordcloud)보다 체감상 성능이 우수
    """
    target_page = wiki.page(self.target_title)
    entire_text = target_page.text
    corpus = self.preprocess_text(entire_text)
    wordcloud = WordCloud(font_path = '/content/drive/MyDrive/sentence/MALGUN.TTF', 
                          background_color = backgroundcolor, 
                          width = width, 
                          height = height).generate(corpus)
    plt.figure(figsize = (15 , 10))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()


  def get_url(self):
    """
    문서의 url을 출력해주는 함수
    """
    target_page = wiki.page(self.target_title)
    print("[{}] 문서의 url주소: {}".format(self.target_title, target_page.fullurl))


  def digit(self, text, num_beams=4, length_penalty=1.0, max_length=512):
    """
    digit82_KoBART 모델로 요약하는 함수
    Args:
      text (str): 요약 대상 원문 텍스트
      num_beams (int): beam search에 사용할 시퀀스 개수
      length_penalty (float): 요약문의 길이에 영향을 주는 가중치
      max_length (int): 요약문 토큰의 최대 길이
    Returns:
      result: 요약 결과물
    """
    raw_input_ids = digit_tokenizer.encode(text) # 토큰화, 정수 인코딩
    raw_input_ids = raw_input_ids[:1022] # KoBART 토큰 길이 제한 맞추기
    input_ids = [digit_tokenizer.bos_token_id] + raw_input_ids + [digit_tokenizer.eos_token_id] # bos, eos 토큰 추가
    summary_ids = digit_model.generate(torch.tensor([input_ids]), 
                                       num_beams=num_beams, 
                                       max_length=max_length, 
                                       length_penalty=length_penalty, 
                                       eos_token_id=1)
    result = digit_tokenizer.decode(summary_ids.squeeze().tolist(), skip_special_tokens=True)
    return result


  def gogamza(self, text, num_beams=4, length_penalty=1.0, max_length=512):
    """
    gogamza_KoBART 모델로 요약하는 함수
    """
    raw_input_ids = gogamza_tokenizer.encode(text) # 토큰화, 정수 인코딩
    raw_input_ids = raw_input_ids[:1022] # KoBART 토큰 길이 제한 맞추기
    input_ids = [gogamza_tokenizer.bos_token_id] + raw_input_ids + [gogamza_tokenizer.eos_token_id] # bos, eos 토큰 추가
    summary_ids = gogamza_model.generate(torch.tensor([input_ids]), 
                                         num_beams=num_beams, 
                                         max_length=max_length, 
                                         length_penalty=length_penalty, 
                                         eos_token_id=1)
    result = gogamza_tokenizer.decode(summary_ids.squeeze().tolist(), skip_special_tokens=True)
    return result


  def ainize(self, text, num_beams=4, length_penalty=1.0, max_length=512):
    """
    ainize_KoBART 모델로 요약하는 함수
    """
    input_ids = ainize_tokenizer.encode(text)
    input_ids = input_ids[1:len(input_ids)-1] # KoBART 토큰 길이 제한 맞추기
    input_ids = [ainize_tokenizer.bos_token_id] + input_ids[:1022] + [ainize_tokenizer.eos_token_id] # 토큰화, 정수 인코딩, torch 자료형으로 변경
    summary_ids = ainize_model.generate(torch.tensor([input_ids]), 
                                        num_beams=num_beams, 
                                        max_length=max_length, 
                                        length_penalty=length_penalty, 
                                        eos_token_id=1)
    result = ainize_tokenizer.decode(summary_ids.squeeze().tolist(), skip_special_tokens=True)
    return result


  def wiki_summary_outline(self, summary_model, num_beams=4, length_penalty=1.0, max_length=512):
    """
    문서의 개요 부분을 요약하는 함수
    """
    target_page = wiki.page(self.target_title)
    if target_page.exists():
      model = self.model_dict[summary_model]
      text = target_page.summary
      result = model(text, num_beams, length_penalty, max_length)

    else:
      result = "[{}](이)라는 이름의 문서는 위키피디아에 존재하지 않습니다".format(self.target_title)

    return result


  def wiki_summary_section(self, summary_model, num_beams=4, length_penalty=1.0, max_length=512):
    """
    원하는 섹션 하나 골라 요약해주는 함수
    """
    model = self.model_dict[summary_model]
    target_page = wiki.page(self.target_title)

    if target_page.exists(): # 문서가 존재할 경우만 요약
      section_collection = self.sections_dict(target_page.sections)
      section_collection = self.del_unnecessary_section(section_collection)

      print(section_collection.keys())
      target_key = input('위의 keys 중 요약을 원하는 섹션을 따옴표 없이 정확히 입력해주세요 : ')
      if target_key in section_collection.keys():
        text = section_collection[target_key]
        result = model(text, num_beams, length_penalty, max_length)
      else:
        result = "[{}](이)라는 이름의 섹션은 [{}] 문서에 존재하지 않습니다.".format(target_key, self.target_title)

    else: # 문서가 존재하지 않으면 아래 문구 출력
      result = "[{}](이)라는 이름의 문서는 위키피디아에 존재하지 않습니다".format(self.target_title)

    return result


  def wiki_summary_combine(self, summary_model, num_beams=4, length_penalty=1.0, 
                           max_length=512, return_dict=False):
    """
    세션을 각각 요약한 것을 하나로 합쳐주는 함수
    """
    model = self.model_dict[summary_model]
    target_page = wiki.page(self.target_title)

    if target_page.exists(): # 문서가 존재할 경우만 요약
      section_collection = self.sections_dict(target_page.sections)
      section_collection = self.del_unnecessary_section(section_collection)
      summary_dict = {}
      for k in section_collection.keys():
        text = section_collection[k]
        summary = model(text, num_beams, length_penalty, max_length)
        summary_dict[k] = summary
      summary_dict_collect = summary_dict

      if return_dict: # return_dict=True인 경우 딕셔너리 형태로 반환
        result = summary_dict_collect

      else: # 개행된 문자열로 출력
        result = ""
        for i, j in enumerate(summary_dict_collect.keys()):
          result += "# section [{}]: [{}] \n    - summary: {} \n\n".format(i, j, 
                                                                           summary_dict_collect[j])

    else: # 문서가 존재하지 않으면 아래 문구 출력
      result = "[{}](이)라는 이름의 문서는 위키피디아에 존재하지 않습니다".format(self.target_title)

    return result


  def preprocess_text(self, text):
    """
    워드클라우드, 토픽모델링을 위해 전처리해주는 함수
    """
    okt = Okt()
    with open('/content/drive/MyDrive/sentence/stopwords.txt',  encoding='cp949') as f:
      list_file = f.readlines()
      stopwords = list_file[0].split(",")

    only_korean = re.sub('[^가-힣]', ' ', text)
    spaced_text = ' '.join(only_korean.split())
    morph_text = okt.morphs(spaced_text, stem=True)
    lenght_two = [x for x in morph_text if len(x)>1]
    nonstopwords = [x for x in lenght_two if x not in stopwords]
    corpus = ' '.join(nonstopwords)
    return corpus

  def make_wordcloud(self, summary_model, num_beams=4, length_penalty=1.0, max_length=512, 
                     backgroundcolor='white', width=600, height=400):
    """
    요약문을 토대로 워드클라우드를 만드는 함수
    """
    summary_dict = self.wiki_summary_combine(summary_model=summary_model,
                                     num_beams=num_beams,
                                     length_penalty=length_penalty,
                                     max_length=max_length,
                                     return_dict=True)
    entire_text = ' '.join(summary_dict.values())
    corpus = self.preprocess_text(entire_text)
    wordcloud = WordCloud(font_path = '/content/drive/MyDrive/sentence/MALGUN.TTF', 
                          background_color = backgroundcolor, 
                          width = width, 
                          height = height).generate(corpus)
    plt.figure(figsize = (15 , 10))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()


  def compute_coherence_values(self, dictionary, corpus, texts, limit=15, start=2, step=3):
    """
    토픽모델링을 하기 앞서 최적의 토픽 개수를 찾아주는 함수
    """
    coherence_values = {}
    for num_topics in range(start, limit, step):
        model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values[num_topics] = coherencemodel.get_coherence()

    return max(coherence_values, key=coherence_values.get)


  def topic_modeling(self, summary_model, num_beams=4, length_penalty=1.0, max_length=512):
      """
      요약문을 토대로 토픽 모델링을 진행하는 함수
      """
      summary_dict = self.wiki_summary_combine(summary_model=summary_model,
                                     num_beams=num_beams,
                                     length_penalty=length_penalty,
                                     max_length=max_length,
                                     return_dict=True)
      preprocessed = []
      for text in summary_dict.values():
        preprocessed.append(self.preprocess_text(text).split())
      dic = gensim.corpora.Dictionary(preprocessed)
      bow_corpus = [dic.doc2bow(doc) for doc in preprocessed]
      NUM_TOPICS = self.compute_coherence_values(dic, bow_corpus, preprocessed)
      lda_model =  gensim.models.LdaModel(bow_corpus, 
                                num_topics = NUM_TOPICS, 
                                id2word = dic,                                    
                                passes = 10)
      pyLDAvis.enable_notebook()
      vis = pyLDAvis.gensim.prepare(lda_model, bow_corpus, dic)
      return vis

  def topic_modeling_entire(self):
    """
    문서 전체를 토대로 토픽 모델링을 진행하는 함수
    """
    target_page = wiki.page(self.target_title)

    if target_page.exists(): # 문서가 존재할 경우만 요약
      section_collection = self.sections_dict(target_page.sections)
      section_collection = self.del_unnecessary_section(section_collection)
      preprocessed = []
      for text in section_collection.values():
        preprocessed.append(self.preprocess_text(text).split())
      dic = gensim.corpora.Dictionary(preprocessed)
      bow_corpus = [dic.doc2bow(doc) for doc in preprocessed]
      NUM_TOPICS = self.compute_coherence_values(dic, bow_corpus, preprocessed)
      lda_model =  gensim.models.LdaModel(bow_corpus, 
                                num_topics = NUM_TOPICS, 
                                id2word = dic,                                    
                                passes = 10)
      pyLDAvis.enable_notebook()
      result = pyLDAvis.gensim.prepare(lda_model, bow_corpus, dic)

    else: # 문서가 존재하지 않으면 아래 문구 출력
      result = "[{}](이)라는 이름의 문서는 위키피디아에 존재하지 않습니다".format(self.target_title)
      
    return result




if __name__ == '__main__': 
  digit_tokenizer = PreTrainedTokenizerFast.from_pretrained('digit82/kobart-summarization')
  digit_tokenizer.save_pretrained('./digit_tokenizer')

  digit_model = BartForConditionalGeneration.from_pretrained('digit82/kobart-summarization')
  digit_model.save_pretrained('./digit_model')

  gogamza_tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-summarization')
  gogamza_tokenizer.save_pretrained('./gogamza_tokenizer')

  gogamza_model = BartForConditionalGeneration.from_pretrained('gogamza/kobart-summarization')
  gogamza_model.save_pretrained('./gogamza_model')

  ainize_tokenizer = PreTrainedTokenizerFast.from_pretrained('ainize/kobart-news')
  ainize_tokenizer.save_pretrained('./ainize_tokenizer')

  ainize_model = BartForConditionalGeneration.from_pretrained('ainize/kobart-news')
  ainize_model.save_pretrained('./ainize_model')