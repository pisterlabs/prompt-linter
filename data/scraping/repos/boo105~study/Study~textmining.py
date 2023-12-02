import pandas as pd
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import matplotlib.pyplot as plt

# 토큰화
def flatten(l):
    flatList = []
    for elem in l:
        if type(elem) == list:
            for e in elem:
                flatList.append(e)
        else:
            flatList.append(elem)
    return flatList
# 단어 빈도분석
def word_frequency():
    f = open("C:\\Users\\user\\Desktop\\MinHo\\Python\\Study\\트럼프취임연설문.txt", 'r')
    # 트럼프취임연설문은 enter가 삽입되지 않은 한 줄 짜리 데이터 이므로 f.readline()으로 저장 할 경우 [내용] 내용 양옆에 대괄호가 쳐져있다.
    # 슬라이싱을 하기 위해 f.readlines()[0]을 사용 하였다.
    lines = f.readlines()[0]
    f.close()
    print(lines[0:100])

    tokenizer = RegexpTokenizer('[\w]+')  # 정규표현식
    stop_words = stopwords.words('english')  # 불용어
    words = lines.lower()  # 소문자로 변환

    tokens = tokenizer.tokenize(words)  # 토큰화
    stopped_tokens = [i for i in list((tokens)) if not i in stop_words]  # 불용어 제거
    stopped_tokens2 = [i for i in stopped_tokens if len(i) > 1]  # 단어수가 2개인것부터 단어 취급해 저장 (글자수가 하나짜리는 단어가 아님)

    # Counter를 사용해 counter를 할 수도 있지만 여기서는 pandas.Series.value_counts() 를 씀
    print(pd.Series(stopped_tokens2).value_counts().head(10))
    """
    f = open("C:\\Users\\user\\Desktop\\MinHo\\Python\\Study\\문재인대통령취임연설문.txt", 'r')
    lines = f.readlines()
    f.close()
    
    # flatten 에서 lines 가 토큰화 안되는 문제? 발생
    word_list = flatten(lines)  # 한국어토큰화를 이용해 토큰화를 해줘도 되는데 여기서는 직접만든 토큰화를 적용시킴
    word_list = pd.Series([x for x in word_list if len(x) > 1])  # 단어수가 하나 이상일떄만 단어로 저장
    print(word_list.value_counts().head(10))
    """

# k-평군 군집화 (분할 군집 분석)
def clustering():
    # 많은 알고리즘이 있지만 주로 분할 군집 분석 과 구조적 군집 분석이 쓰인다.
    # 군집화는 비지도학습이다. , 군집이란 비슷한 특징을 갖는 데이터 집단
    # k-평균 군집 분석 -> 주이진 데이터를 k개의 클러스터로 묶는 알고리즘 (분류의 기준은 거리)
    # 각 클러스터와 거리 차이의 분산을 최소화하는 방식으로 작동한다.
    # 지도학습과 달리 기준이 되는 라벨이 없기 때문에 알고리즘이 데이터를 기준으로 특성(feature) 벡터들의 분산과 거리를 기준으로 카테고리를 자동으로 구성한다.
    # 코사인 유사도는 0~1 사이의 값을 가지므로 코사인 유사도를 이용한 거리는 1-유사도로 정의한다. (텍스트 분석에 대한 군집분석에는 코사인 유사도가 효과적임)
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.cluster import KMeans
    from konlpy.tag import Hannanum
    from sklearn.decomposition import PCA

    hannanum = Hannanum()

    Data = pd.read_csv("C:\\Users\\user\\Desktop\\MinHo\\Python\\Study\\군집분석데이터.csv", engine = "python")
    #print(Data.head())

    # 군집화
    docs = []
    # 첫번째 for문이 끝나면 docs는 이중리스트 형태로 저장이 되있다.
    for i in Data['기사내용']:
        docs.append(hannanum.nouns(i))  # 명사 추출
    # 두번째 for문이 끝나면 docs는 [문장1,문장2,...,문장15] 같이 변형이 되있음
    for i in range(len(docs)):
        docs[i] = ' '.join(docs[i])

    vec = CountVectorizer()
    # fit_transform이란 데이터를 정규화 해줌 fit() 평균빼고 표준편차로 나눠줌 transform() 변환을 적용
    X = vec.fit_transform(docs)     # 추출한 단어들을 이용하여 문서-단어 매트릭스를 생성함

    df = pd.DataFrame(X.toarray(), columns=vec.get_feature_names()) # 배열 형태로 DataFrame화 시킴
    kmeans = KMeans(n_clusters=3).fit(df)   # kmeans 로 군집화 k = 3
    #print(kmeans.labels_)

    # 시각화를 위해 pca기법 사용해 분석 결과를 2차원으로 축소
    pca = PCA(n_components=2)   # n_components 는 주성분을 몇개로 할지 (몇 차원을 할지)
    principalComponents = pca.fit_transform(df)
    principalDf = pd.DataFrame(data=principalComponents, columns=['pricipal component 1', 'principal component 2'])
    principalDf.index = Data['검색어']

    plt.scatter(principalDf.iloc[kmeans.labels_ == 0, 0], principalDf.iloc[kmeans.labels_ == 0, 1], s=10, c='red', label='cluster1')
    plt.scatter(principalDf.iloc[kmeans.labels_ == 1, 0], principalDf.iloc[kmeans.labels_ == 1, 1], s=10, c='blue', label='cluster2')
    plt.scatter(principalDf.iloc[kmeans.labels_ == 2, 0], principalDf.iloc[kmeans.labels_ == 2, 1], s=10, c='green',label='cluster3')

    plt.legend()
    plt.show()
# k-대표값 군집화 (분할 군집 분석)
def clustering2():
    # k-대표값 군집 분석  (k-평균 분석은 이상치에 매우 민감하므로 민감성을 보완한 알고리즘이다.)
    # 분석시 반복 횟수가 많아져 데이터가 많아지면 분석 시간도 대폭 증가함
    from pyclustering.cluster import kmedoids
    import numpy as np
    from konlpy.tag import Hannanum
    from sklearn.feature_extraction.text import CountVectorizer

    hannanum = Hannanum()
    Data = pd.read_csv("C:\\Users\\user\\Desktop\\MinHo\\Python\\Study\\군집분석데이터.csv", engine="python")
    docs = []
    for i in Data['기사내용']:
        docs.append(hannanum.nouns(i))
    for i in range(len(docs)):
        docs[i] = ' '.join(docs[i])
    vec = CountVectorizer()
    X = vec.fit_transform(docs)
    df = pd.DataFrame(X.toarray(), columns=vec.get_feature_names())

    # k-대표값 군집 분석
    kmedoids_instance = kmedoids.kmedoids(df.to_numpy(), initial_index_medoids=np.random.randint(15, size=3))   # 군집화 모델 생성 이때 대표값은 데이터 개수 15개중에 size= 군집화개수 만큼 뽑는다.
    kmedoids_instance.process() # 군집화 반복 실행
    clusters = kmedoids_instance.get_clusters()
    print(clusters)
# 군집화 (구조적 군집 분석)
def clustering3():
    from sklearn.cluster import AgglomerativeClustering
    import scipy.cluster.hierarchy as shc
    from konlpy.tag import Hannanum
    from sklearn.feature_extraction.text import CountVectorizer
    import numpy as np

    hannanum = Hannanum()
    Data = pd.read_csv("C:\\Users\\user\\Desktop\\MinHo\\Python\\Study\\군집분석데이터.csv", engine="python")

    docs = []
    for i in Data['기사내용']:
        docs.append(hannanum.nouns(i))
    for i in range(len(docs)):
        docs[i] = ' '.join(docs[i])

    vec = CountVectorizer()
    X = vec.fit_transform(docs)
    df = pd.DataFrame(X.toarray(), columns=vec.get_feature_names())
    #print(df)
    # 구조적 군집 분석
    cluster = AgglomerativeClustering(n_clusters=3, linkage='ward') # linkage 는 가장 비슷한 클러스트를 측정하는 방법 ward는 기본값, 모든 클러스터 내의 분산을 가장 작게 증가시키는 두 클러스터를 합침, 크기가 비교적 비슷한 클러스터가 만들어짐
    print(cluster.fit_predict(df))
    
    # 시각화
    plt.figure(figsize=(10, 7))
    plt.title("Customer Dendrograms")
    # create dendrogram 구조적으로 어떤식으로 생겼는지 보기위해 dendrogram 생성해 시각화 시키는거임 scatter로도 표현됨
    dend = shc.dendrogram(shc.linkage(df, method='ward'))
    plt.show(dend)
# LDA(토픽 모델링) 초창기 모델임
def lda():
    # 토픽 모델링은 구조화되지 않은 방대한 문헌집단에서 주제를 찾아내기 위한 통계적 추론 알고리즘이다.
    # 맥락과 관련된 단어들을 이용하여 의미를 가진 단어들을 클러스터링하여 주제를 추론한다.
    # 감성 분석과 같은 타 분석 모델과 혼합하여 자주 쓰인다.
    # 표, 단어구름, 단어 네트워크, 연도별 그래프 등 다양한 시각화 기법과 결합했을 때 더 효과적이다.
    # LDA에서는 단어의 교환성만을 가정한다. 교환성 : 단어의 순서는 고려하지않고 단어의 유무만 중요
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer
    import gensim
    from gensim.models import CoherenceModel
    from nltk.tokenize import RegexpTokenizer
    
    # 토큰화 설정
    tokenizer = RegexpTokenizer('[\w]+')    
    stop_words = stopwords.words('english')
    p_stemmer = PorterStemmer()
    
    # 문장들
    doc_a = "Brocolli is good to eat. My brother likes to eat good brocolli, but not my mother."
    doc_b = "My mother spends a lot of time driving my brother around to baseball practice."
    doc_c = "Some health experts suggest that driving may cause increased tension and blood pressure."
    doc_d = "I often feel pressure to perform well at school, but my mother never seems to drive my brother to do better."
    doc_e = "Health professionals say that brocolli is good for your health."
    doc_f = "Big data is a term used to refer to data sets that are too large or complex for traditional data-processing application software to adequately deal with."
    doc_g = "Data with many cases offer greater statistical power, while data with higher complexity may lead to a higher false discovery rate"
    doc_h = "Big data was originally associated with three key concepts: volume, variety, and velocity."
    doc_i = "A 2016 definition states that 'Big data represents the information assets characterized by such a high volume, velocity and variety to require specific technology and analytical methods for its transformation into value'."
    doc_j = "Data must be processed with advanced tools to reveal meaningful information."
    
    # 리스트화
    doc_set = [doc_a, doc_b, doc_c, doc_d, doc_e, doc_f, doc_g, doc_h, doc_i, doc_j]

    texts = []
    
    # 토큰화
    for w in doc_set:
        raw = w.lower() # 소문자 변환
        tokens = tokenizer.tokenize(raw)    # 토큰화
        stopped_tokens = [i for i in tokens if not i in stop_words]     # 불용어 제거
        stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]    # 어간, 표제어 추출
        texts.append(stemmed_tokens)

    # 문서의 단어들을 사전형으로 바꿈
    dictionary = gensim.corpora.Dictionary(texts)

    # 문서-단어 매트릭스 형성
    corpus = [dictionary.doc2bow(text) for text in texts]   # doc2bow 함수는 단어 - 빈도수 형태로 바꿈

    # topic 개수=3으로 지정
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=3, id2word=dictionary)
    # 토픽별로 5개씩 단어 출력
    print(ldamodel.print_topics(num_words=5))  # 단어에 곱해진 숫자는 가중치이다. 해당 단어가 토픽에서 설명하는 비중을 나타냄(빈도)
    print(ldamodel.get_document_topics(corpus)[0])  # 0번째 문장에서의 각 토픽 분포 모든 분포의 확률 합은 1이다.

    # 통계정 방법은 크게 perlexity , topic coherence 가 있다.(토픽개수를 입력하기위한)
    print('\nPerlexity: ', ldamodel.log_perplexity(corpus))  # 모델이 실측값을 얼마나 잘 예측하는지 평가할떄 쓰이는 수치 (작을수록 좋음)
    # 상위 10개(topn=10)의 단어를 이용하여 유사도를 계산
    coherence_model_lda = CoherenceModel(model=ldamodel, texts=texts, dictionary=dictionary, topn=10)   # 유사도 모델 생성하는데 시간이 좀 걸림
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)

    # 토픽개수의 따른 perplexity 시각화
    perplexity_values = []
    for i in range(2, 10):
        ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=i, id2word=dictionary)    # 토픽 개수의 따른 모델 생성
        perplexity_values.append(ldamodel.log_perplexity(corpus))   # 토픽개수별 perplexity 를 저장함

    x = range(2, 10)
    plt.plot(x, perplexity_values)
    plt.xlabel("Number of topics")
    plt.ylabel("Perplexity score")
    plt.show()

    # 토픽개수의 따른 coherence 시각화
    coherence_values = []
    for i in range(2, 10):
        ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=i, id2word=dictionary)
        coherence_model_lda = CoherenceModel(model=ldamodel, texts=texts, dictionary=dictionary, topn=10)   # 토픽개수의 따라 만든 모델에 따른 유사도 모델 생성
        coherence_lda = coherence_model_lda.get_coherence()
        coherence_values.append(coherence_lda)   # 토픽개수별 coherence 를 저장함

    x = range(2, 10)
    plt.plot(x, coherence_values)
    plt.xlabel("Number of topics")
    plt.ylabel("Coherence score")
    plt.show()


if __name__ == "__main__":
    lda()
    #clustering3()
    #word_frequency()

