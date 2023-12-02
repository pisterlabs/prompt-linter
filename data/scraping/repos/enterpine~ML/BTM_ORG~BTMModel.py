import sys
#sys.setdefaultencoding('utf-8')
import numpy as np
import codecs
import random
from BTM_ORG.Biterm import Biterm
import pickle
import jieba

class BtmModel(object):
	"""
		biterm Topic Model
	"""
	def __init__(self, docs,dictionary,topic_num, iter_times, alpha, beta,has_background=False):
		"""
		初始化 模型对象
		:param voca_size: 词典大小
		:param topic_num: 话题数目
		:param iter_times: 迭代次数
		:param save_step:
		:param alpha: hyperparameters of p(z)
		:param beta:  hyperparameters of p(w|z)
		"""
		self.biterms = list()
		# self.voca_size = voca_size
		self.topic_num = topic_num
		self.n_iter = iter_times

		self.alpha = alpha
		self.beta = beta
		self.docs=docs
		self.dictionary=dictionary
		self.nb_z = list()  # int 类型，n(b|z) ,size topic_num+1
		self.nwz = None     # int 类型矩阵， n(w,z), size topic_num * voca_size
		self.pw_b = list()  # double 词的统计概率分布
		self.has_background = has_background

	def build_word_dic(self,sentences):
		"""
			构建词典,并统计 各词的词频（整个语料中）
		:param sentences:
		:return:
		"""
		self.word_dic = self.dictionary   # 词典 索引
		self.word_fre = {}   # word frequently
		word_id = 1

		for sentence in sentences:
			for word in sentence:
				word_index = self.word_dic[word]
				if word_index in self.word_fre:
					self.word_fre[word_index] += 1
				else:
					self.word_fre[word_index] = 1
		self.voca_size = len(self.word_fre.keys())
		sum_val = sum(self.word_fre.values())
		smooth_val = 0.001
		# 归一化，正则化
		for key in self.word_fre:
			self.word_fre[key] = (self.word_fre[key] + smooth_val) / (sum_val + smooth_val * (self.topic_num + 1))
		with codecs.open("PreProcess/word_freq.txt", 'w', encoding='utf8') as fp:
			for key in self.word_fre:
				fp.write("{}\t{}\n".format(key,self.word_fre[key]))
		with open("PreProcess/word_dic.txt", 'w') as fp:
			for key in self.word_dic:
				# print "{}\t{}".format(str(key), self.word_dic[key])
				fp.write(str(key)+"\t"+str(self.word_dic[key])+"\n")

	def build_wordId(self,sentences):
		"""
		将文本 中word 映射到 word_id 并将结果存储到文件
		:param sentences: 切词后的文档
		:return: 当回 文档的 [wid,...,wid]列表
		"""
		with codecs.open("PreProcess/word_id.txt",'w',encoding='utf8') as fp:
			for sentence in sentences:
				doc = []
				# print sentence
				for word in sentence:
					doc.append(self.word_dic[word])
				wid_list = [str(wid) for wid in doc]
				# print wid_list
				fp.write(' '.join(wid_list)+"\n")

	def build_Biterms(self, sentence):
		"""
		获取 document 的 biterms
		:param sentence: word id list sentence 是切词后的每一词的ID 的列表
		:return: biterm list
		"""
		win = 15 # 设置窗口大小
		biterms = []
		# with codecs.open("word_id.txt", 'r', encoding="utf8") as fp:
		# 	sentence = []
		# sentence =
		for i in range(len(sentence)-1):
			for j in range(i+1, min(i+win+1, len(sentence))):
				biterms.append(Biterm(int(sentence[i]),int(sentence[j])))
		return biterms

	def loadwordId(self,file='PreProcess/word_id.txt'):
		"""
		获取语料的词 ID
		:param file:
		:return:
		"""
		sentences_wordId = []
		with open(file, 'r') as fp:
			[sentences_wordId.append(line.strip().split(" ")) for line in fp]
		return sentences_wordId

	def staticBitermFrequence(self):
		"""
		统计 biterms 的频率
		:param sentences: 使用word id 表示的 sentence 列表
		:return: 返回corpus 中各个 biterm （wid,wid）: frequence 的频率
		"""
		sentences = []
		with codecs.open("PreProcess/word_id.txt", 'r', encoding="utf8") as fp:
			sentences = [ line.strip().split(" ") for line in fp]
		self.biterms = []
		for sentence in sentences:
			bits = self.build_Biterms(sentence)
			self.biterms.extend(bits)
		with open("PreProcess/biterm_freq.txt", 'w') as fp:
			for key in self.biterms:
				fp.write(str(key.get_word())+" "+str(key.get_word(2))+"\n")

	def model_init(self):
		"""
		模型初始化
		:return:
		"""
		# 初始化 话题 biterm 队列和word -topic 矩阵
		self.nb_z = [0]*(self.topic_num+1)
		self.nwz = np.zeros((self.topic_num,self.voca_size))

		for bit in self.biterms:
			k = random.randint(0, self.topic_num-1)
			self.assign_biterm_topic(bit, k)

	def assign_biterm_topic(self, bit, topic_id):
		"""
		为 biterm 赋予 topic ，并更新 相关nwz 及 nb_z 数据
		:param bit:
		:param topic_id:
		:return:
		"""
		w1 = int(bit.get_word())-1
		w2 = int(bit.get_word(2))-1
		bit.setTopic(topic_id)
		self.nb_z[topic_id] += 1
		self.nwz[int(topic_id)][w1] = self.nwz[int(topic_id)][w1] + 1
		self.nwz[int(topic_id)][w2] = self.nwz[int(topic_id)][w2] + 1

	def runModel(self,res_dir="./output/"):
		"""
		运行构建模型
		:param doc_pt: 数据源文件路径
		:param res_dir: 结果存储文件路径
		:return:
		"""
		sentences = self.docs
		self.build_word_dic(sentences)
		self.build_wordId(sentences)
		self.staticBitermFrequence()
		self.model_init()

		print ("Begin iteration")
		out_dir = res_dir + "k" + str(self.topic_num)+'.'
		for iter in range(self.n_iter):
			#print("\r当前迭代{}，总迭代{}".format(iter,self.n_iter))
			for bit in self.biterms:
				self.updateBiterm(bit)

	def updateBiterm(self, bit):
		self.reset_biterm(bit)

		pz = [0]*self.topic_num
		self.compute_pz_b(bit, pz)

		#
		topic_id = self.mult_sample(pz)
		self.assign_biterm_topic(bit, topic_id)

	def compute_pz_b(self, bit, pz):
		"""
		更新 话题的概率分布
		:param bit:
		:param pz:
		:return:
		"""
		w1 = bit.get_word()-1
		w2 = bit.get_word(2)-1
		for k in range(self.topic_num):
			if self.has_background and k == 0:
				pw1k = self.pw_b[w1]
				pw2k = self.pw_b[w2]
			else:
				pw1k = (self.nwz[k][w1] + self.beta)/ (2*self.nb_z[k] + self.voca_size*self.beta)
				pw2k = (self.nwz[k][w2] + self.beta) / (2 * self.nb_z[k] + 1 + self.voca_size * self.beta)
			pk = (self.nb_z[k] + self.alpha) / (len(self.biterms) + self.topic_num * self.alpha)
			pz[k] = pk * pw1k * pw2k

	def mult_sample(self, pz):
		"""
		sample from mult pz
		:param pz:
		:return:
		"""
		for i in range(1,self.topic_num):
			pz[i] += pz[i-1]

		u = random.random()
		k = None
		for k in range(0,self.topic_num):
			if pz[k] >= u * pz[self.topic_num-1]:
				break
		if k == self.topic_num:
			k -= 1
		return k

	def show(self, top_num=10):
		print ("BTM topic model \t",)
		print ("topic number {}, voca word size : {}".format(self.topic_num, self.voca_size))
		word_id_dic = {}
		for key in self.word_dic:
			word_id_dic[self.word_dic[key]] = key

		for topic in range(self.topic_num):
			print ("\nTopic: #{}".format(topic),)
			print ("Topic top word \n",)
			print(self.nwz)
			b = list(zip(self.nwz[int(topic)],range(self.voca_size)))

			b.sort(key=lambda x: x[0], reverse=True)
			print(b)
			for index in range(top_num):
				print (word_id_dic[b[index][1]+1], b[index][0],)
			print

	def SentenceProcess(self,sentence):
		"""
		文本预处理
		:param sentence: 输入文本
		:return:
		"""
		# 去停用词等过滤处理
		words = jieba.cut(sentence)
		words_id = []
		# 将文本转换为 word ID
		print (words)
		for w in words:
			if w in list(self.word_dic.keys()):
				words_id.append(self.word_dic[w])
		return self.build_Biterms(words_id)

	def sentence_topic(self, sentence, topic_num=1, min_pro=0.01):
		"""
		计算 sentence 最可能的话题属性,基于原始的LDA 方法
		:param sentence: sentence
		:param topic_num: 返回 可能话题数目 最多返回
		:param min_pro: 话题概率最小阈值，只有概率大于该值，才是有效话题，否则不返回
		:return: 返回可能的话题列表，及话题概率
		"""
		words_id = self.SentenceProcess(sentence)
		topic_pro = [0.0]*self.topic_num
		sentence_word_dic = [0]*self.voca_size
		weigth = 1.0/len(words_id)
		for word_id in words_id:
			sentence_word_dic[word_id] = weigth
		for i in range(self.topic_num):
			topic_pro[i] = sum(map(lambda x, y: x*y, self.nwz[i], sentence_word_dic))
		sum_pro = sum(topic_pro)
		topic_pro = map(lambda x: x/sum_pro, topic_pro)
		# print topic_pro
		min_result = list(zip(topic_pro, range(self.topic_num)))
		min_result.sort(key=lambda x: x[0], reverse=True)
		result = []
		for re in min_result:
			if re[0] > min_pro:
				result.append(re)

		return result[:topic_num]

	def infer_sentence_topic(self, sentence, topic_num=1, min_pro=0):
		"""
		BTM topic model to infer a document or sentence 's topic
		基于 biterm s 计算问题
		:param sentence: sentence
		:param topic_num: 返回 可能话题数目 最多返回
		:param min_pro: 话题概率最小阈值，只有概率大于该值，才是有效话题，否则不返回
		:return: 返回可能的话题列表，及话题概率
		"""
		sentence_biterms = self.SentenceProcess(sentence)

		topic_pro = [0]*self.topic_num
		# 短文本分析中，p (b|d) = nd_b/doc(nd_b)  doc(nd_b) 表示 计算的query 的所有biterm的计数
		# 因此，在short text 的p(b|d) 计算为1／biterm的数量
		bit_size = len(sentence_biterms)
		for bit in sentence_biterms:
			# cal p(z|d) = p(z|b)*p(b|d)
			# cal p(z|b)
			pz = [0]*self.topic_num
			self.compute_pz_b(bit, pz)
			pz_sum = sum(pz)
			pz = map(lambda pzk: pzk/pz_sum, pz)

			for x, y in list(zip(range(self.topic_num), pz)):
				topic_pro[x] += y/bit_size

		min_result = list(zip(topic_pro, range(self.topic_num)))
		min_result.sort(key=lambda x: x[0], reverse=True)
		result = []
		for re in min_result:
			if re[0] > min_pro:
				result.append(re)
		return result[:topic_num]

	def infer_sentence_topic_2(self, sentence):
		"""
		BTM topic model to infer a document or sentence 's topic
		基于 biterm s 计算问题
		:param sentence: sentence
		:param topic_num: 返回 可能话题数目 最多返回
		:return: 返回可能的话题列表，及话题概率
		"""
		sentence_biterms = self.SentenceProcess(sentence)

		topic_pro = [0]*self.topic_num
		# 短文本分析中，p (b|d) = nd_b/doc(nd_b)  doc(nd_b) 表示 计算的query 的所有biterm的计数
		# 因此，在short text 的p(b|d) 计算为1／biterm的数量
		bit_size = len(sentence_biterms)
		for bit in sentence_biterms:
			# cal p(z|d) = p(z|b)*p(b|d)
			# cal p(z|b)
			pz = [0]*self.topic_num
			self.compute_pz_b(bit, pz)
			pz_sum = sum(pz)
			pz = map(lambda pzk: pzk/pz_sum, pz)

			for x, y in list(zip(range(self.topic_num), pz)):
				topic_pro[x] += y/bit_size

		min_result = list(topic_pro)
		#min_result.sort(key=lambda x: x[0], reverse=True)
		result = []
		for re in min_result:
			result.append(re)
		return result

	def reset_biterm(self, bit):
		k = bit.getTopic()
		w1 = int(bit.get_word())-1
		w2 = int(bit.get_word(2))-1

		self.nb_z[k] -= 1
		self.nwz[k][w1] -= 1
		self.nwz[k][w2] -= 1
		min_val = -(10**(-7))
		# if self.nb_z[k] > min_val and self.nwz[k][w1] > min_val and
		bit.resetTopic()

	def get_topics(self):
		rtn=[]
		for doc in self.docs:
		 	rtn.append(self.infer_sentence_topic_2(sentence= ''.join(doc)))
		return rtn


def save(model,file="Model/BitModel_5.model"):
	with codecs.open(file,'wb') as fp:
		pickle.dump(model, fp)

def load(file="Model/BitModel_5.model"):
	with codecs.open(file, 'rb') as fp:
		model = pickle.load(fp)
	return model

def main():

	jieba.suggest_freq('沙瑞金', True)
	jieba.suggest_freq('易学习', True)
	jieba.suggest_freq('王大路', True)
	jieba.suggest_freq('京州', True)
	jieba.suggest_freq('桓温', True)
	import pandas as pd
	df = pd.read_csv('./btm_text_corpus.txt', header=None, sep=',').astype(str)
	stpwrdpath = "stop_words.txt"
	stpwrd_dic = open(stpwrdpath, encoding='GBK')
	stpwrd_content = stpwrd_dic.read()
	stpwrdlst = stpwrd_content.splitlines()
	segment = []
	for index, row in df.iterrows():
		content = row[0]
		if content != 'nan':
			words = jieba.cut(content)
			splitedStr = ''
			rowcut = []
			for word in words:
				if word not in stpwrdlst:
					splitedStr += word + ' '
					rowcut.append(word)
			segment.append(rowcut)
	docs = segment  # 赋值给docs

	from gensim.corpora.dictionary import Dictionary
	dictionary = Dictionary(docs)  # 生成字典
	BTMdic = {}
	for i in dictionary:
		BTMdic[dictionary[i]] = i+1

	BitM = BtmModel(docs=docs,dictionary=BTMdic,topic_num=3, iter_times=50, alpha=0.1, beta=0.01, has_background=False)
	BitM.runModel()	#save(BitM)#BitM = load()

	BitM.show()
	#print(BitM.get_topics())
	#print(BitM.infer_sentence_topic(sentence='沙瑞金欧阳菁',topic_num=3))
	from gensim.corpora.dictionary import Dictionary
	from gensim.models.coherencemodel import CoherenceModel
	coherence_model_lda = CoherenceModel(model=BitM, texts=docs, dictionary=dictionary, coherence='c_npmi')
	coherence_lda = coherence_model_lda.get_coherence()
	print('\nCoherence Score: ', coherence_lda)

	# def compute_coherence_values(dictionary, texts, start, limit, step):
	# 	coherence_values = []
	# 	model_list = []
	# 	for num_topics in range(start, limit, step):
	# 		model = BtmModel(docs=docs, dictionary=BTMdic, topic_num=num_topics, iter_times=50, alpha=0.1, beta=0.01,
	# 						has_background=False)
	# 		model.runModel()
	# 		model_list.append(model)
	# 		coherencemodel = CoherenceModel(model=model, \
	# 										texts=texts, \
	# 										dictionary=dictionary, \
	# 										coherence='c_uci')
	# 		coherence_values.append(coherencemodel.get_coherence())
	# 	return model_list, coherence_values
    #
	# limit = 8;
	# start = 2;
	# step = 1;  # K的最大值，起始值，步长
	# model_list, coherence_values = compute_coherence_values(dictionary=dictionary, texts=docs,
	# 														start=start, limit=limit, step=step)
	# # Show graph
	# import matplotlib.pyplot as plt
	# x = range(start, limit, step)
	# plt.plot(x, coherence_values)
	# plt.xlabel("Num Topics")
	# plt.ylabel("Coherence score")
	# plt.legend(("coherence_values"), loc='best')
	# plt.show()

if __name__ == "__main__":
	main()