# -*- coding: utf-8 -*-

"""
	Created by:		Shaheen Syed
	Date:			August 2018	
	
	The evaluation phase includes a careful analysis and inspection of the latent variables from the various created LDA models. Since LDA is an unsupervised machine learning technique, 
	extra care should be given during this post-analysis phase; in contrast to, for example, supervised methods where typically a labeled gold-standard dataset exist. 

	Measures such as predictive likelihood on held-out data have been proposed to evaluate the quality of generated topics. However, such measures correlate negatively with human 
	interpretability, making topics with high predictive likelihood less coherent from a human perspective. High-quality or coherent latent topics are of particular importance when 
	they are used to browse document collections or understand the trends and development within a particular research field. As a result, researchers have proposed topic coherence measures, 
	which are a quantitative approach to automatically uncover the coherence of topics. Topics are considered to be coherent if all or most of the words (e.g., a topic's top-N words) are 
	related. Topic coherence measures aim to find measures that correlate highly with human topic evaluation, such as topic ranking data obtained by, for example, word and topic intrusion 
	tests. Human topic ranking data are often considered the gold standard and, consequently, a measure that correlates well is a good indicator for topic interpretability. 

	Exploring the topics by a human evaluator is considered the best approach. However, since this involves inspecting all the different models, this approach might not be feasible. 
	Topic coherence measures can quantitatively calculate a proxy for topic quality, and per our analysis, topics with high coherence were considered interpretable by domain experts. 
	Combing coherence measures with a manual inspection is thus a good approach to find the LDA model that result in meaningful and interpretable topics. In short, three questions 
	should be answered satisfactory: 

		- Are topics meaningful, interpretable, coherent and useful?
		- Are topics within documents meaningful, appropriate and useful?  
		- Do the topics facilitate a better understanding of the underlying corpus?
	
	The evaluation phase can also result in topics that are very similar (i.e., identical topics), topics that should ideally be merged or split (i.e., chained or mixed topics), topics 
	that are un-interpretable (i.e. nonsensical), or topics that contain unimportant, too specific, or too general words. In those cases, it would be wise to revisit the pre-processing 
	phase and repeat the analysis.


	For reference articles see:
	Syed, S., Borit, M., & Spruit, M. (2018). Narrow lenses for capturing the complexity of fisheries: A topic analysis of fisheries science from 1990 to 2016. Fish and Fisheries, 19(4), 643–661. http://doi.org/10.1111/faf.12280
	Syed, S., & Spruit, M. (2017). Full-Text or Abstract? Examining Topic Coherence Scores Using Latent Dirichlet Allocation. In 2017 IEEE International Conference on Data Science and Advanced Analytics (DSAA) (pp. 165–174). Tokyo, Japan: IEEE. http://doi.org/10.1109/DSAA.2017.61
	Syed, S., & Spruit, M. (2018a). Exploring Symmetrical and Asymmetrical Dirichlet Priors for Latent Dirichlet Allocation. International Journal of Semantic Computing, 12(3), 399–423. http://doi.org/10.1142/S1793351X18400184
	Syed, S., & Spruit, M. (2018b). Selecting Priors for Latent Dirichlet Allocation. In 2018 IEEE 12th International Conference on Semantic Computing (ICSC) (pp. 194–202). Laguna Hills, CA, USA: IEEE. http://doi.org/10.1109/ICSC.2018.00035
	Syed, S., & Weber, C. T. (2018). Using Machine Learning to Uncover Latent Research Topics in Fishery Models. Reviews in Fisheries Science & Aquaculture, 26(3), 319–336. http://doi.org/10.1080/23308249.2017.1416331

"""

# packages and modules
import logging, sys, re
from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from database import MongoDatabase
from helper_functions import *



class Evaluation():

	def __init__(self):

		logging.info('Initialized {}'.format(self.__class__.__name__))

		# instantiate database
		self.db = MongoDatabase()


	def calculate_coherence(self, file_folder = os.path.join('files', 'lda'), models_folder = os.path.join('files', 'models')):

		"""
			Calculate the CV coherence score for each of the created LDA models

			Parameters
			----------
			file_folder: os.path
				location of the dictionary and corpus for gensim
			models_folder: os.path
				location where the lda model is saved
		"""

		logging.info('Start {}'.format(sys._getframe().f_code.co_name))

		# read dictionary and corpus
		dictionary, corpus = get_dic_corpus(file_folder)

		# load bag of words features of each document from the database
		texts = [x['tokens'] for x in self.db.read_collection('publications_raw')]

		# get path location for models
		M = [x for x in read_directory(models_folder) if x.endswith('lda.model')]

		# read processed models from database
		processed_models = ['{}-{}-{}-{}-{}'.format(x['k'], x['dir_prior'], x['random_state'], x['num_pass'], x['iteration']) for x in self.db.read_collection('coherence')]
		
		# calculate coherence score for each model
		for i, m in enumerate(M):

			logging.info('Calculating coherence score: {}/{}'.format(i+1, len(M)))

			print m

			# number of topics
			k = m.split(os.sep)[2]
			# different dirichlet priors
			dir_prior = m.split(os.sep)[3]
			# random initiatilizations
			random_state = m.split(os.sep)[4]
			# passes over the corpus
			num_pass = m.split(os.sep)[5]
			# max iteration for convergence
			iteration = m.split(os.sep)[6]

			logging.info('k: {}, dir_prior: {}, random_state: {}, num_pass: {}, iteration: {}'.format(k, dir_prior, random_state, num_pass, iteration))

			# check if coherence score already obtained
			if '{}-{}-{}-{}-{}'.format(k, dir_prior, random_state, num_pass, iteration) not in processed_models: 
				
				# load LDA model
				model = models.LdaModel.load(m)

				# get coherence c_v score
				coherence_c_v = CoherenceModel(model = model, texts = texts, dictionary = dictionary, coherence='c_v')
				
				# get coherence score
				score = coherence_c_v.get_coherence()
				
				# logging output
				logging.info('coherence score: {}'.format(score))

				# save score to database
				doc = {	'k' : k, 'dir_prior' : dir_prior, 'random_state' : random_state, 'num_pass' : num_pass, 'iteration' : iteration, 'coherence_score' : score}
				self.db.insert_one_to_collection('coherence', doc)

			else:
				logging.info('coherence score already calculated, skipping ...')
				continue


	def plot_coherence(self, min_k = 2, max_k = 20, save_location = os.path.join('files', 'plots'), plot_save_name = 'coherence_scores_heatmap.pdf'):

		"""
			Read coherence scores from database and create heatmap to plot scores

			Parameters
			-----------
			min_k: int 
				owest number of topics created when creating LDA models. Here 2
			max_k: int
				highest number of topics created when creating LDA models. Here 20
			save_location: os.path
				location where to save the plot
			plot_save_name: string
				name for the plot
		"""

		logging.info('Start {}'.format(sys._getframe().f_code.co_name))

		# make sure plot save location exists
		create_directory(save_location)

		# read documents from database that contain coherence scores
		D = list(self.db.read_collection(collection = 'coherence'))

		# convert data from document into a list
		data = [[int(x['k']), x['dir_prior'],x['random_state'], x['num_pass'], x['iteration'], x['coherence_score']] for x in D]

		# create empty dataframe where we can store our scores
		df = pd.DataFrame()

		# loop trough values of k parameter and find relevant scores for each grid search combination
		for k in range(min_k, max_k + 1):

			# create dataframe to temporarily store values
			df_temp = pd.DataFrame(index = [k])

			# loop trough the data to obtain only the scores for a specific k value
			for row in sorted(data):
				if row[0] == k:
					df_temp['{}-{}-{}-{}'.format(row[1],row[2],row[3],row[4])] = pd.Series(row[5], index=[k])
			
			# append temporarary dataframe of only 1 k value to the full dataframe 
			df = df.append(df_temp)
		
		# transpose the dataframe
		df = df.transpose()
		
		# plot the heatmap
		ax = sns.heatmap(df, cmap = "Blues", annot = True, vmin = 0.500, vmax = 0.530, square = True, annot_kws = {"size": 11},
							fmt = '.3f', linewidths = .5, cbar_kws = {'label': 'coherence score'})

		# adjust the figure somewhat
		ax.xaxis.tick_top()
		plt.yticks(rotation=0)
		plt.xticks(rotation=0, ha = 'left') 
		fig = ax.get_figure()
		fig.set_size_inches(19, 6)

		# save figure
		fig.savefig(os.path.join(save_location, plot_save_name), bbox_inches='tight')



	def output_lda_topics(self, K = 9, dir_prior = 'auto', random_state = 42, num_pass = 15, iteration = 200, top_n_words = 10, models_folder = os.path.join('files', 'models'), 
						save_folder = os.path.join('files', 'tables')):

		"""
			Create table with LDA topic words and probabilities
			Creates a table of topic words and probabilties + topics in a list format
			
			Values for K, dir_prior, random_state, num_pass and iteratrion will become visible when plotting the coherence score. Use the model that 
			achieved the highest coherence score and plug in the correct values. The values will create the correct file location of the LDA model
			for example : files/models/2/auto/42/5/200/lda.model

			Parameters
			-----------
			k: int
				number of topics that resulted in the best decomposition of the underlying corpora
			dir_prior: string
				dirichlet priors 'auto', 'symmetric', 'asymmetric'
			random_state: int
				seed value for random initialization
			num_pass: int
				number of passes over the full corpus
			iteration: int
				max iterations for convergence
			top_n_words: int
				only print out the top N high probability words
			models_folder: os.path
				location of created LDA models
			save_folder: os.path
				location to store the tables

		"""

		logging.info('Start {}'.format(sys._getframe().f_code.co_name))

		# load LDA model according to parameters
		model = load_lda_model(os.path.join(models_folder, str(K), dir_prior, str(random_state), str(num_pass), str(iteration)))
		
		# define empty lists so we can fill them with words		
		topic_table, topic_list = [], []

		# loop trough all the topics found within K
		for k in range(K):

			# create topic header, e.g. (1) TOPIC X
			topic_table.append(['{}'.format(get_topic_label(k, labels_available = False).upper())])
			# add column for word and probability
			topic_table.append(["word", "prob."])


			list_string = ""
			topic_string = ""
			topic_string_list = []

			# get topic distribution for topic k and return only top-N words 
			scores = model.print_topic(k, top_n_words).split("+")
			
			# loop trough each word and probability
			for score in scores:

				# extract score and trimm spaces
				score = score.strip()

				# split on *
				split_scores = score.split('*')

				# get percentage
				percentage = split_scores[0]
				# get word
				word = split_scores[1].strip('"')

				# add word and percentage to table
				topic_table.append([word.upper(), "" + percentage.replace("0.", ".")])
				
				# add word to list table
				list_string += word + ", "

			# add empty line for the table
			topic_table.append([""])
			# add topic words to list
			topic_list.append([str(k+1), list_string.rstrip(", ")])

		# save to CSV
		save_csv(topic_list, 'topic-list', folder = save_folder)
		save_csv(topic_table, 'topic-table', folder = save_folder)

