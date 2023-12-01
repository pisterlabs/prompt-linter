import codecs
import pickle
import jieba
import pandas as pd
from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from BTM_ORG_REMASTER.BTMModel import BtmModel

def save(model,file="Model/BitModel_5.model"):
	with codecs.open(file,'wb') as fp:
		pickle.dump(model, fp)

def load(file="Model/BitModel_5.model"):
	with codecs.open(file, 'rb') as fp:
		model = pickle.load(fp)
	return model

def main():
	df = pd.read_csv('../data/buwenminglvke.csv', header=None, sep=',', encoding='GBK').astype(str)
	# 从文件导入停用词表
	stpwrdpath = "stop_words.txt"
	stpwrd_dic = open(stpwrdpath, encoding='GBK')
	stpwrd_content = stpwrd_dic.read()
	stpwrdlst = stpwrd_content.splitlines()

	# 处理输入数据
	segment = []
	for index, row in df.iterrows():
		content = row[7]
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


	dictionary = Dictionary(docs)  # 生成字典
	BTMdic = {}
	for i in dictionary:
		BTMdic[dictionary[i]] = i+1

	BitM = BtmModel(docs=docs,dictionary=BTMdic,topic_num=3, iter_times=50, alpha=0.1, beta=0.01, has_background=False)
	BitM.runModel()	#save(BitM)#BitM = load()
	BitM.show()
	print(BitM.get_topics())

	coherence_model_lda = CoherenceModel(model=BitM, texts=docs, dictionary=dictionary, coherence='c_npmi')
	coherence_lda = coherence_model_lda.get_coherence()
	print('\nCoherence Score: ', coherence_lda)

	def compute_coherence_values(dictionary, texts, start, limit, step):
		coherence_values = []
		model_list = []
		for num_topics in range(start, limit, step):
			model = BtmModel(docs=docs, dictionary=BTMdic, topic_num=num_topics, iter_times=50, alpha=0.1, beta=0.01,
							has_background=False)
			model.runModel()
			model_list.append(model)
			coherencemodel = CoherenceModel(model=model, \
											texts=texts, \
											dictionary=dictionary, \
											coherence='u_mass')
			coherence_values.append(coherencemodel.get_coherence())
		return model_list, coherence_values

	limit = 8;
	start = 2;
	step = 1;  # K的最大值，起始值，步长
	model_list, coherence_values = compute_coherence_values(dictionary=dictionary, texts=docs,
															start=start, limit=limit, step=step)
	# Show graph
	import matplotlib.pyplot as plt
	x = range(start, limit, step)
	plt.plot(x, coherence_values)
	plt.xlabel("Num Topics")
	plt.ylabel("Coherence score")
	plt.legend(("coherence_values"), loc='best')
	plt.show()

if __name__ == "__main__":
	main()