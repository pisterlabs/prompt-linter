__author__ = 'mroylance'

import os
import re
import sys
import extract
import attensity.semantic_server
import model.doc_model
import extract.topicReader
import extract.documentRepository
import extractionclustering.topicSummary
import extractionclustering.docModel
import extractionclustering.paragraph
import extractionclustering.sentence
import coherence.scorer
import coreference.rules
import pickle

cachePath = "../cache/docModelCacheOld"
summaryOutputPath = "../outputs"
reorderedSummaryOutputPath = summaryOutputPath + "_reordered"
evaluationOutputPath = "../results"
modelSummaryCachePath = "../cache/modelSummaryCache"
documentCachePath = "../cache/documentCache"
idfCachePath = "../cache/idfCache"
meadCacheDir = "../cache/meadCache"
rougeCacheDir = "../cache/rougeCache"

ss = attensity.semantic_server.SemanticServer("http://192.168.1.7:8888")
configUrl = ss.configurations().config_url(1)

directories = ["/opt/dropbox/14-15/573/Data/models/devtest","/opt/dropbox/14-15/573/Data/models/training/2009","/opt/dropbox/14-15/573/Data/mydata"]

fileNames = []
for directory in directories:
	files = os.listdir(directory)
	for file in files:
		fileToPickle = os.path.join(directory, file)
		text = ""
		with open(fileToPickle, "r") as myFile:
			text = unicode(myFile.read(), errors='ignore')

		rawDocument = extract.document.Document()
		rawDocument.paragraphs.append(text)
		initialModel = model.doc_model.Doc_Model(rawDocument)
		coreference.rules.updateDocumentWithCoreferences(initialModel)
		coherence.scorer.determineDoc(initialModel)

		docModel = extractionclustering.docModel.DocModel()
		for paragraph in initialModel.paragraphs:
			newParagraph = extractionclustering.paragraph.Paragraph()
			docModel.paragraphs.append(newParagraph)
			# cleansedParagraph = re.sub("\s+", " ", str(paragraph))
			newParagraph.text = str(paragraph)

			parse = ss.parse()
			parse.process(str(paragraph), configUrl)
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

		# cache
		pickleFileName = os.path.join("../cache/asasGoldCache", file)
		pickleFile = open(pickleFileName, 'wb')
		pickle.dump(docModel, pickleFile, pickle.HIGHEST_PROTOCOL)
		print "pickled " + pickleFileName
