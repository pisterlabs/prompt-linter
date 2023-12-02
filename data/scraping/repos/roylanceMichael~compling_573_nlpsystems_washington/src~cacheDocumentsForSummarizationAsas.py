__author__ = 'thomas'
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
import extract
import extract.topicReader
import extract.documentRepository2
import extract
import re
import attensity.semantic_server
import model.doc_model
import extract.topicReader
import extract.documentRepository
import extractionclustering.docModel
import extractionclustering.paragraph
import coherence.scorer
import coreference.rules
import os
import cPickle as pickle

# get parser args and set up global variables
parser = argparse.ArgumentParser(description='Basic Document Summarizer.')
parser.add_argument('--doc-input-path', help='Path to data files', dest='docInputPath')
parser.add_argument('--topic-xml', help='Path to topic xml file', dest='topicXml')
parser.add_argument('--data-type', help='one of: \"devtest\", \"training\", or \"evaltest\"', dest='dataType')

args = parser.parse_args()

##############################################################
# global variables
##############################################################
documentCachePath = "../cache/documentCache"




##############################################################
# Script Starts Here
###############################################################

# get training xml file
# go through each topic
topics = []
for topic in extract.topicReader.Topic.factoryMultiple(args.topicXml):
	topics.append(topic)

documentRepository = extract.documentRepository2.DocumentRepository2(args.docInputPath, args.dataType, topics)

# load the cached docs
documentRepository.readFileIdDictionaryFromFileCache(documentCachePath)

ss = attensity.semantic_server.SemanticServer("http://192.168.1.7:8888")
configUrl = ss.configurations().config_url(1)

docModelCache = {}
# load and cache the docs if they are not loaded.  just get them if they are.
for topic in topics:
	transformedTopicId = topic.docsetAId[:-3] + '-A'
	print "caching topicId: " + transformedTopicId
	# let's get all the documents associated with this topic

	# get the doc objects, and build doc models from them
	for foundDocument in documentRepository.getDocumentsByTopic(topic.id):
		documentRepository.writefileIdDictionaryToFileCache(documentCachePath)
		initialModel = model.doc_model.Doc_Model(foundDocument)
		docNo = initialModel.docNo
		coreference.rules.updateDocumentWithCoreferences(initialModel)
		coherence.scorer.determineDoc(initialModel)

		docModel = extractionclustering.docModel.DocModel()
		for paragraph in initialModel.paragraphs:
			newParagraph = extractionclustering.paragraph.Paragraph()
			docModel.paragraphs.append(newParagraph)
			# cleansedParagraph = re.sub("\s+", " ", str(paragraph))
			newParagraph.text = re.sub(r'[^\x00-\x7F]', ' ', str(paragraph))

			parse = ss.parse()
			actualText = newParagraph.text
			print actualText
			parse.process(actualText, configUrl)
			ext = attensity.extractions.Extractions.from_protobuf(parse.result)

			for extraction in ext.extractions():
				if extraction.type == attensity.ExtractionMessage_pb2.Extraction.KEYWORD_RESULTS:
					roots = {}
					i = 0
					for item in extraction.keyword_results.root:
						roots[i] = {"root": item.root, "word": item.word, "pos": item.pos}
						i += 1
					i = 0
					for item in extraction.keyword_results.location:
						roots[i]["sentence"] = item.sentence
						i += 1

					for key in roots:
						if "sentence" not in roots[key]:
							continue

						try:
							sentenceId = int(roots[key]["sentence"])
							root = str(roots[key]["root"])
							word = str(roots[key]["word"])
							pos = list(roots[key]["pos"])
							cacheTuple = (sentenceId, root, word, pos)
							newParagraph.extractionKeywordResults.append(cacheTuple)
						except Exception:
							print "error happened"
				if extraction.type == attensity.ExtractionMessage_pb2.Extraction.FACT_RELATION:
					newParagraph.extractionFactRelations.append((extraction.fact_relation.fact_one, extraction.fact_relation.fact_two, extraction.fact_relation.text))
				if extraction.type == attensity.ExtractionMessage_pb2.Extraction.TEXT_SENTENCE:
					print "|||" + newParagraph.text[extraction.text_sentence.offset:extraction.text_sentence.offset + extraction.text_sentence.length] + "|||"
					newParagraph.extractionSentences.append((extraction.text_sentence.text_sentence_ID, extraction.text_sentence.offset, extraction.text_sentence.length))
				if extraction.type == attensity.ExtractionMessage_pb2.Extraction.ENTITY:
					mid = ""
					if len(extraction.entity.search_info) > 0:
						mid = extraction.entity.search_info[0].machine_ID
					newParagraph.extractionEntities.append((extraction.entity.sentence_id, extraction.entity.display_text, extraction.entity.sem_tags, extraction.entity.domain_role, mid))
				if extraction.type == attensity.ExtractionMessage_pb2.Extraction.TRIPLE:
					newParagraph.extractionTriples.append((extraction.triple.sentence_ID, extraction.triple.t1.value, extraction.triple.t1.sem_tags, extraction.triple.t2.value, extraction.triple.t2.sem_tags, extraction.triple.t3.value, extraction.triple.t3.sem_tags))
					# print extraction
				if extraction.type == attensity.ExtractionMessage_pb2.Extraction.FACT:
					# print extraction
					newParagraph.extractionFacts.append((extraction.fact.sentence_ID, extraction.fact.element.text, extraction.fact.mode.text))
				if extraction.type == attensity.ExtractionMessage_pb2.Extraction.TEXT_PHRASE:
					# print extraction
					newParagraph.extractionTextPhrases.append((extraction.text_phrase.sentence_ID, extraction.text_phrase.head, extraction.text_phrase.root))

		docModelCache[docNo] = docModel

	# cache
	pickleFileName = os.path.join("../cache/asasCacheEval", topic.id)
	pickleFile = open(pickleFileName, 'wb')
	pickle.dump(docModelCache, pickleFile, pickle.HIGHEST_PROTOCOL)
	docModelCache = {}

# recache documents for later
# documentRepository.writefileIdDictionaryToFileCache(documentCachePath)
print "done caching documents"
