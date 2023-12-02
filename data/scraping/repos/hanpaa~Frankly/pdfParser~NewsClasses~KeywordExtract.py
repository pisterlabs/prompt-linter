import gensim
import pymysql

from InformationClass.Politician import Politician
from NewsClasses.News import News
from NewsClasses.NewsKeyword import NewsKeyword
from NewsClasses.preprocessor import Preprocessor
from NewsClasses.LDAkey_extractor import LDAKeyExtractor
from NewsClasses.textrank import TextRank
from multiprocessing import freeze_support

from konlpy.tag import Mecab
import pyLDAvis.gensim_models as gensimvis
# import warnings
# warnings.filterwarnings(action='ignore')
from gensim.models import LdaModel

from gensim import corpora
from gensim.models import CoherenceModel


import numpy as np
import kss




class KeywordExtract:

    def keywordExtract(self, date):


        f = open("stop.txt", 'r')
        stopWord = f.read().split(',')
        f.close()
        con, cur = self.dbConnect()
        politicianList = Politician().selectALL(cursor=cur)
        dateList = News().selectNewsDateList(cur, date)
        for politician in politicianList:
            stopWord.append(politician.politicianName)

        for politician in politicianList:
            print(politician.politicianName + " 키워드 추출 시작")
            preprocessor = Preprocessor()


            for date in dateList:

                newsObjectList = News().selectByIDDate(cur,politician.politicianID, \
                                                       startdate=date[0], enddate=date[1])
                if(len(newsObjectList) == 0):
                    continue
                print(date[0] + " to " + date[1])
            # newsObjectList = News().selectByID(cur, politicianList[0].politicianID)

                newsList = []
                newsIDList = []
                for news in newsObjectList:
                    newsIDList.append(news.newsID)
                    newsList.append(news.newsContent)



                id2word,corpus,topicNum,bigramModel = preprocessor.preprocess(newsList,stopWord)
                print("preprocess")
                ldaExtractor = LDAKeyExtractor(topicNum)
                idx_topic,lda_model,_ = ldaExtractor.extract_keyword(corpus,id2word)
                print("lda modeling")
                corp_doc_topic, topic_docs_save = preprocessor.cluster_extract_sentences(lda_model,idx_topic,corpus,newsList,
                                                                                              newsIDList,topicNum,id2word,bigramModel, stopWord)
                print("clustering done")

                textrank = TextRank(corp_doc_topic)
                print("textrank done")
                keywords = textrank.extract_keyword()
                print(len(keywords))
                print("keyword extracted")
                ext_topic_cluster = dict()
                for i in range(1, len(keywords)+1):
                    top_save = dict()
                    for j in range(len(topic_docs_save[i])):
                        top_save[topic_docs_save[i][j][0]] = topic_docs_save[i][j][1]
                    save_key = []
                    for l in range(len(keywords[i-1])):
                        save_key.append(keywords[i-1][l][2])
                    ext_topic_cluster[i] = [list(sorted(save_key, key=lambda x: len(x), reverse=True)), dict(sorted(top_save.items(), key=lambda x: x[1],reverse=True))]
                print(ext_topic_cluster)

                ext_topic_cluster = dict(sorted(ext_topic_cluster.items(), key=lambda x: len(x[1][1]), reverse=True))
                try:
                    for _, k in zip(range(len(keywords)), ext_topic_cluster.keys()):
                        keywords = ""
                        article_num = len(ext_topic_cluster[k][1])

                        for keyword in ext_topic_cluster[k][0]:
                            keywords = keywords + keyword + " "

                        for key, value in ext_topic_cluster[k][1].items():
                            newsKeyword = NewsKeyword()
                            newsKeyword.politicianID = politician.politicianID
                            newsKeyword.newsKeyword = keywords
                            newsKeyword.newsID = key
                            newsKeyword.insert(cur)
                            con.commit()
                except:
                    print(politician.politicianName + " 완료")
                    continue
                print(politician.politicianName + " 완료")



                # #     # topics, keywords 저장
                # for value ,key in zip(range(len(keywords)), ext_topic_cluster.keys()):
                #     newsKeyword =





    def dbConnect(self):
        # dbinfoDir = "E:\work\Frankly\pdfParser\InformationClass/dbinfo.info"
        # dbinfoDir = "D:\code\Frankly\pdfParser\InformationClass/dbinfo.info"
        dbinfoDir = "/home/hanpaa/IdeaProjects/Frankly/pdfParser/dbinfo.info"
        with open(dbinfoDir, encoding="UTF8") as dbInfo:

            IP  = dbInfo.readline().split(" ")[1].replace("\n", "")
            port = dbInfo.readline().split(" ")[1].replace("\n", "")
            userID = dbInfo.readline().split(" ")[1].replace("\n", "")
            password = dbInfo.readline().split(" ")[1].replace("\n", "")
            dbname = dbInfo.readline().split(" ")[1].replace("\n", "")

            connection = pymysql.connect(host= IP, port=int(port), user=userID, passwd=password, db=dbname, charset="utf8")
            cursor = connection.cursor()

            dbInfo.close()
            return connection, cursor

# if __name__ == "__main__":
#
#     freeze_support()
#     key = KeywordExtract()
#
#     key.keywordExtract("2022-10-01")