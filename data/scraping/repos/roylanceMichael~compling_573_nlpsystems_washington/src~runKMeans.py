__author__ = 'mroylance'
"""
  Python Source Code for ling573 Deliverable 3: Ordering Summarizer
  Author: Thomas Marsh
  Team: Thomas Marsh, Brandon Gaylor, Michael Roylance
  Date: 5/16/2015

  This code does the following:
  1. opens doc files
  2. extracts data from doc files
  3. summarizes doc files and outputs summaries for each topic
  4. compares summary to baseline using ROUGE and outputs results

"""

import argparse
import os
import re

import extract
import extract.topicReader
import extract.documentRepository
import coherence.scorer
import model.idf
import model.doc_model as doc_model
import npclustering.npClustering

from evaluate.rougeEvaluator import RougeEvaluator
from npclustering.npClustering import NpClustering
from summarization.initialSummarizer import InitialSummarizer
from evaluate.evaluationCompare import EvaluationCompare

# get parser args and set up global variables
parser = argparse.ArgumentParser(description='Basic Document Summarizer.')
parser.add_argument('--doc-input-path', help='Path to data files', dest='docInputPath')
parser.add_argument('--doc-input-path2', help='Path to secondary data files', dest='docInputPath2')
parser.add_argument('--topic-xml', help='Path to topic xml file', dest='topicXml')
parser.add_argument('--output-path', help='Path to our output', dest='outputPath')
parser.add_argument('--rouge-path', help='Path to rouge', dest='rougePath')
parser.add_argument('--data-type', help='one of: \"devtest\", \"training\", or \"evaltest\"', nargs='?', default="devtest", dest='dataType')
parser.add_argument('--gold-standard-summary-path', help='Path to gold standard summaries',
					dest='goldStandardSummaryPath')
args = parser.parse_args()

##############################################################
# global variables
##############################################################
summaryOutputPath = args.outputPath
evaluationOutputPath = "../results"
modelSummaryCachePath = "../cache/modelSummaryCache"
documentCachePath = "../cache/documentCache"
idfCachePath = "../cache/idfCache"
meadCacheDir = "../cache/meadCache"
rougeCacheDir = "../cache/rougeCache"

rouge = RougeEvaluator(args.rougePath, args.goldStandardSummaryPath, summaryOutputPath, modelSummaryCachePath, rougeCacheDir)
idf = model.idf.Idf(idfCachePath)


##############################################################
# send the data to the model generator
##############################################################
def getModel(docData):
	model = doc_model.Doc_Model(docData)
	model.updateWithCoref()
	model.scoreWithCoherence()

	return model


##############################################################
# summarize
##############################################################
def summarize(docModels):
	initialSummarizer = InitialSummarizer(docModels, idf, False, False, False, False, True)
	return initialSummarizer.getBestSentences(w_tfidf=0.0, w_sd=0.0, w_sl=0.0, w_topic=0.0, w_cosign=0.0, w_np=0.0,
		pullfactor=0.0, initialwindow=1, initialbonus=1, topicsize=0, parameters=None)


##############################################################
# evaluate our summary with rouge
##############################################################
def evaluate():
	return rouge.evaluate()


##############################################################
# print out models
##############################################################
def printSummary(docModels):
	for docModel in docModels:
		for paragraph in docModel.paragraphs:
			print str(paragraph)


##############################################################
# helper function for printing out buffers to files
##############################################################
def writeBufferToFile(path, buffer):
	outFile = open(path, 'w')
	outFile.write(buffer)
	outFile.close()

##############################################################
# Script Starts Here
###############################################################

# get training xml file
# go through each topic
topics = []
for topic in extract.topicReader.Topic.factoryMultiple(args.topicXml):
	topics.append(topic)

documentRepository = extract.documentRepository.DocumentRepository(args.docInputPath, args.docInputPath2, args.dataType, topics)

# load the cached docs
documentRepository.readFileIdDictionaryFromFileCache(documentCachePath)

# cache the model summaries
rouge.cacheModelSummaries(topics)


# load and cache the docs if they are not loaded.  just get them if they are.
for topic in topics:
	transformedTopicId = topic.docsetAId[:-3] + '-A'
	print "caching topicId: " + transformedTopicId
	# let's get all the documents associated with this topic

	# get the doc objects, and build doc models from them
	for foundDocument in documentRepository.getDocumentsByTopic(topic.id):
		# print "caching document: " + foundDocument.docNo
		pass

# recache documents for later
documentRepository.writefileIdDictionaryToFileCache(documentCachePath)

for topic in topics:
	transformedTopicId = topic.docsetAId[:-3] + '-A'
	print "processing topicId: " + transformedTopicId
	# let's get all the documents associated with this topic
	models = list()
	# get the doc objects, and build doc models from them
	for foundDocument in documentRepository.getDocumentsByTopic(topic.id):
		print "processing docNo: " + foundDocument.docNo
		convertedModel = getModel(foundDocument)
		models.append(convertedModel)

	# make a summary of the topic cluster
	print topic.category + " : " + topic.title + " : building summary for " + str(len(models)) + " models"
	allPoints = []

	# for point in npclustering.kmeans.buildPointForEachSentence(models):
	# 	allPoints.append(point)
	#
	# initialPoints = npclustering.kmeans.getInitialKPoints(allPoints, 5)
	# clusters = npclustering.kmeans.performKMeans(initialPoints, allPoints)

	# instance = NpClustering(models, topic.title)
	#
	# maxWords = 100
	# totalWords = 0
	# topSentences = []
	# for sentenceTuple in instance.buildSentenceDistances():
	# 	if totalWords > maxWords:
	# 		break
	#
	# 	cleansedSentence = re.sub("\s+", " ", sentenceTuple[0].simple)
	# 	topSentences.append(sentenceTuple[0].simple)
	# 	totalWords += len(cleansedSentence.split(" "))

	summary = topic.title
	# for sentence in topSentences:
	# 	summary = summary + sentence + "\n"

	for docModel in models:
		summary += str(docModel.headline).strip() + "\n"

	# selectedSentences = {}
	# we receive a tuple back, currently
	# for cluster in clusters[0]:
	# 	selectedSentences[cluster.number] = []
	# 	for pointKey in cluster.points:
	# 		for topSentence in topSentences:
	# 			if topSentence == cluster.points[pointKey].sentence.simple:
	# 				selectedSentences[cluster.number].append(topSentence)
	# 				break
	#
	# for clusterNumber in selectedSentences:
	# 	for sentence in selectedSentences[clusterNumber]:
	# 		summary += sentence + "\n"

	if summary is not None:
		summaryFileName = summaryOutputPath + "/" + topic.id
		summaryFile = open(summaryFileName, 'w')
		summaryFile.write(summary)
		summaryFile.close()

	print summary
	print "----------"

print "running the rouge evaluator"
evaluationResults = evaluate()
evaluation = evaluationResults[0]
writeBufferToFile(os.path.join(evaluationOutputPath, "D4.results"), evaluation)
writeBufferToFile(os.path.join(evaluationOutputPath, "D4.results_reordered"), evaluation)
# call the evaluation comparison routine.
# note:  this will only print the summaries you have on your machine.
# 		 i.e. you should have run the meadSummaryGenerator.py first
# 		 (though defaults are checked into git)
comparator = EvaluationCompare(evaluationOutputPath, meadCacheDir, rouge)
comparison = comparator.getComparison()
print "\n" + comparison
writeBufferToFile(os.path.join(evaluationOutputPath, "results_compare.txt"), comparison)