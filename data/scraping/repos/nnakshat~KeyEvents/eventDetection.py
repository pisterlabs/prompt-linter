"""
This script is used to identify all the events within a context window of size "k" around the point-of-interest/peak; 
"""
from generalUtils import readJson, writeJson, readPickle, writePickle, createFolder
import argparse
from sentence_transformers import SentenceTransformer, util
import spacy
import openai
import os
from dateutil.rrule import rrule, MONTHLY, WEEKLY, DAILY
from datetime import datetime, timedelta
from tqdm import tqdm
import numpy as np
import time
from copy import deepcopy
from gensim.models import TfidfModel
from gensim import corpora
import matplotlib.pyplot as plt
import torch
import itertools
import umap
import hdbscan
from sklearn.preprocessing import normalize


spacy.prefer_gpu()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
openAIKeyStr = ""
openai.api_key = openAIKeyStr
tokenizerMod = None

def chatGptRequest(promptList, model="gpt-3.5-turbo", temperature=0.2, numTrials=5):
    answer = ""
    response = None
    for trialNum in range(numTrials):
        try:
            response = openai.ChatCompletion.create(
            model=model,
            messages=promptList,
            temperature=temperature
            )
            if response.get("choices", -1) != -1:
                if len(response['choices']) > 0:
                    answer = response.get("choices", -1)[0]['message']['content'].strip()
                    break
        except openai.error.RateLimitError:
            print("exception from openAI...sleeping code for two minutes!")
            time.sleep(122)
            continue
        except:
            print("random exception from openAI...sleeping code for two minutes!")
            time.sleep(122)
            continue
    
    return answer, response


def tokenizeArticle(article):
    global tokenizerMod
    tokenizedArticle = None
    tokenizedArticle = tokenizerMod(article).sents

    return [str(s) for s in tokenizedArticle]


def processIssueRelData(data, nelaData):
    newDict = {}
    fastAccessDict = {}
    for issueName, docSet in data.items():
        newDict[issueName] = set()
        fastAccessDict[issueName] = {}
        scoreList = np.array([d[1] for d in docSet])
        positionList = np.array([d[0] for d in docSet])

        threshold = np.floor(np.median(scoreList))
        indiciesOfInterest = np.where(scoreList > threshold)[0]
        
        for index in indiciesOfInterest:
            docPosition = positionList[index]
            matchScore = scoreList[index]
            fastAccessDict[issueName][nelaData[docPosition]['id']] = docPosition
            newDict[issueName].add((docPosition, nelaData[docPosition]['id']))
    
    print("Analyze issue-specific data...")
    print({k: len(v) for k, v in newDict.items()})

    uniqueDocs = set()
    for issueName, docSet in newDict.items():
        uniqueDocs = uniqueDocs.union({d[0] for d in docSet})

    print("Number of Unique Documents: {docCount}".format(docCount=len(uniqueDocs)))

    return newDict, fastAccessDict


def getArticleIdeology(article, leftSourceList, rightSourceList, noBiasSourceList, conspiracySourceList):
    retVal = None
    if article.get('source', "n/a") in leftSourceList:
        retVal = "left"

    elif article.get('source', "n/a") in rightSourceList:
        retVal = "right"

    if article.get('source', "n/a") in noBiasSourceList:
        retVal = "center"

    if article.get('source', "n/a") in conspiracySourceList:
        retVal = "conspiracy_pseudoscience"
    
    return retVal


def getPeakData(issueName, analysisInterval, outputDir, nelaData, fastAccessIssueData, sourceDict, contextWindow=None):
    issueFolder = "{interval}_day_{issueName}".format(interval=analysisInterval, issueName=issueName)
    eventInfoPath = os.path.join(*[outputDir, issueFolder, "eventInfo.json"])
    issueDateBucketPath = os.path.join(*[outputDir, issueFolder, "dateBucket.json"])
    if not os.path.isfile(eventInfoPath):
        print("Wrong path for event file corresponding to issue-{issueName}".format(issueName=issueName))
        return None
    issueDateBucket = readJson(issueDateBucketPath)
    rawDateBucketInfo = [(counter, issueDateBucket[k][0], issueDateBucket[k][1], k) for counter, k in enumerate(issueDateBucket.keys())]
    eventInfo = readJson(eventInfoPath)
    ## check coverage and select events with good coverage;
    peakDocIds = {}
    peakDocs = {}

    for eventId, eventDict in eventInfo.items():
        eventSpecificArticleIds = set()
        peakDocs[eventId] = {"left": [], "right": [], "center": [], "conspiracy_pseudoscience": []}
        ## NOTE: consider only the peak-related articles; this will be changed when only peak-related items are considered in eventInfo.json
        peakStartCounter = eventDict["eventStartCounter"] + 1
        peakEndCounter = eventDict["eventStartCounter"] + eventDict['eventSpanDays'] - 2
        if contextWindow is not None:
            loopStartCounter = peakStartCounter - contextWindow
            loopEndCounter = peakEndCounter + contextWindow + 1
        else:
            loopStartCounter = peakStartCounter
            loopEndCounter  = peakEndCounter + 1

        for i in range(loopStartCounter, loopEndCounter):
        # for i in range(eventDict["eventStartCounter"], eventDict["eventStartCounter"] + eventDict['eventSpanDays']):
            for docId in rawDateBucketInfo[i][2]:
                lenS = len(eventSpecificArticleIds)
                eventSpecificArticleIds.add(docId)
                if lenS != len(eventSpecificArticleIds):
                    article = nelaData[fastAccessIssueData[docId]]
                    if article['id'] != docId:
                        print("Fatal ERROR!")
                        return
                    artIdeology = getArticleIdeology(article, sourceDict['left'], sourceDict['right'], sourceDict['center'],
                                       sourceDict['conspiracy'])
                    article['ideology'] = artIdeology if artIdeology is not None else ""
                    if article['ideology'] != "":
                        peakDocs[eventId][article['ideology']].append(article)

        peakDocIds[eventId] = list(eventSpecificArticleIds)
    
    return peakDocs, peakDocIds, eventInfo

def generateClusterLabel(testMessage):
    systemMessage = "You need to provide a title and a sentence long description for the news event based on news article snippets shown below. The title and description should not be too specific to the articles shown below but rather, they need to focus on the main event."
    userMessage1 = "News Article1: CNN, Biden Official Spark Outrage After Suggesting Climate Change Could Have Caused Condo Collapse\nEnergy Secretary Jennifer Granholm suggested on CNN Tuesday that the deadly collapse last week of a high-rise condominium building in Surfside , Florida , might have been caused by 'climate change' and forest mismanagement.\nYou knew the insane Democrats would find a way to cash in on the tragedy in South Florida which left at least 11 dead and 150 unaccounted for.\n\nNews Article2: Joe Biden:Families Who Lost Loved Ones in Condo Collapse Concerned About Global Warming\nPresident Joe Biden said Thursday the families who lost their loved ones in the Surfside condo collapse in Miami told him about their concerns about global warming."
    assistantMessage1 = "News Event Title: Condo Collapse in Florida.\nNews Event Description: This is about a deadly building collapse in Surfside, Miami, Florida."

    userMessage2 = "News Article1: Pacific Northwest heat wave may have killed hundreds\nHundreds may have died from the record-breaking heat wave that struck Oregon , Washington and the Canadian province of British Columbia this past week as temperatures hit all-time highs in typically moderate cities .\nAt least 63 people died from a heat wave in Oregon, according to media reports that cited state health officials .\n\nNews Article2: Over 100 deaths may be tied to historic Northwest heat wave\nSALEM , Ore.( AP ) —The grim toll of the historic heat wave in the Pacific Northwest became more apparent as authorities in Canada , Washington state and Oregon said Wednesday that they were investigating more than 100 deaths likely caused by scorching temperatures that shattered all-time records .\nOregon health officials said more than 60 deaths have been tied to the heat , with the state's largest county , Multnomah , blaming the weather for 45 deaths since the heat wave began Friday."
    assistantMessage2 = "News Event Title: Heat Wave in Pacific Northwest\nNews Event Description: This is about a horrific heat wave in Pacific Northwest which resulted in the deaths of several people."

    promptList = [{"role": "system", "content": systemMessage},
                {"role": "user", "content": userMessage1},
                {"role": "assistant", "content": assistantMessage1},
                {"role": "user", "content": userMessage2},
                {"role": "assistant", "content": assistantMessage2},
                {"role": "user", "content": testMessage}]
    response, rawResponse = chatGptRequest(promptList, model="gpt-3.5-turbo")

    return response, rawResponse


def checkLabelEquivalence(label1, label2):
    label1 = label1.replace("Title", "Title1")
    label1 = label1.replace("Description", "Description1")
    label2 = label2.replace("Title", "Title2")
    label2 = label2.replace("Description", "Description2")
    systemMessage = "You need to tell if the following two news event descriptions belong to the same news event. You need to say yes or no and nothing more."
    userMessage1 = "News Event Title1: U.S. Braces for Wildfire Disaster\nNews Event Description1: This is about U.S. President Joe Biden recognizing the need to prepare for a record number of forest fires due to drought and high temperatures, and pledging to pay federal firefighters more.\n\nNews Event Title2: Biden's Infrastructure Plan to Combat Climate Change.\nNews Event Description2: President Biden used extreme weather conditions on the west coast to promote his infrastructure plan as a solution to climate change."
    assistantMessage1 = "No"

    userMessage2 = "News Event Title1: Biden Signs Aggressive Executive Orders on Climate Change.\nNews Event Description1: President Biden has signed sweeping executive orders that will force the US government to plan for, respond to, and combat the urgent threat of climate change, creating new offices and interagency groups to prioritize job creation, conservation, and environmental justice, while stopping new fossil fuel leases on public lands and boosting renewable energy development.\n\nNews Event Title2: Biden Halts New Fracking and Oil Leases on Federal Land.\nNews Event Description2: President Biden signed an executive order to suspend new leases for fracking and oil drilling on federal lands in an effort to reduce the nation's contribution to climate change."
    assistantMessage2 = "Yes"
    testMessage = "{newsEvent1}\n\n{newsEvent2}".format(newsEvent1=label1, newsEvent2=label2)
    promptList = [{"role": "system", "content": systemMessage},
                    {"role": "user", "content": userMessage1},
                    {"role": "assistant", "content": assistantMessage1},
                    {"role": "user", "content": userMessage2},
                    {"role": "assistant", "content": assistantMessage2},
                    {"role": "user", "content": testMessage}]
    response, rawResponse = chatGptRequest(promptList, model="gpt-3.5-turbo")

    return response, rawResponse

def l2_normalize(vectors):
    if vectors.ndim == 2:
        return normalize(vectors)
    else:
        return normalize(vectors.reshape(1, -1))[0]


def pruneEventCount(repDocs, eventVectors):
    representativeDocs = {}
    repDocsKeys = sorted(repDocs, key=lambda k: len(repDocs[k]['docs']), reverse=True)
    pruningFlag = False
    prunedEventVectors = np.empty((0, eventVectors.shape[1]), dtype=np.float64)
    newKey = 0
    for e, key in enumerate(repDocsKeys):
        if e < 8:
            representativeDocs[newKey] = repDocs[key]
            prunedEventVectors = np.append(prunedEventVectors, eventVectors[key].reshape(1, -1), axis=0)
        else:
            # eventVectors = np.delete(eventVectors, e, 0)
            pruningFlag = True
        newKey += 1
    return representativeDocs, prunedEventVectors, pruningFlag


def searchClusterParameters(embeddings, hdbParamdist):
    umapArgs = {'n_neighbors': 15,
                        'n_components': 5,
                        'metric': 'cosine'}
            
    bestUmapModel = None
    bestHDBModel = None

    bestScore = 0
    bestParameters = {}
    
    for minClusterSize in hdbParamdist["min_cluster_size"]:
        umapArgs['n_neighbors'] = minClusterSize
        umapModel = umap.UMAP(**umapArgs, random_state=11).fit(embeddings)
        for minSamples in hdbParamdist["min_samples"]:
            for clusterSelectionMethod in hdbParamdist["cluster_selection_method"]:
                for metric in hdbParamdist["metric"]:
                    hdb = hdbscan.HDBSCAN(min_cluster_size=minClusterSize,min_samples=minSamples,
                                        cluster_selection_method=clusterSelectionMethod, metric=metric, 
                                        gen_min_span_tree=True).fit(umapModel.embedding_)
                    # DBCV score
                    score = hdb.relative_validity_
                    # if we got a better DBCV, store it, the model and their parameters
                    if score > bestScore:
                        bestScore = score
                        bestParameters = {'min_cluster_size': minClusterSize, 
                                'min_samples':  minSamples, 'cluster_selection_method': clusterSelectionMethod,
                                'metric': metric}
                        bestUmapModel = umapModel
                        bestHDBModel = hdb

    return bestUmapModel, bestHDBModel, bestScore, bestParameters

def extractClusterLabel(label):
    labelNewsFormat = "{title}\n{description}"
    temp = label.split("News Event Description:")
    if len(temp) > 1:
        title = ""
        description = ""
        tempTitle = temp[0].split("Title:")
        if len(tempTitle) > 1:
            title = tempTitle[1].strip()
        description = temp[1].strip()

        labelNewsFormat = labelNewsFormat.format(title=title, description=description)

    return labelNewsFormat


def mergeAndReIndex(cIndexPair, clusterOutput, repDocs, eventVectors, model, eventData, normalizedEmbeddings, eventRelatedData):
    newRepDocs = {}
    newClusterOutput = {}

    newClusterKey = 0
    newEventVectors = np.empty((0, eventVectors.shape[1]), dtype=np.float64)
        
    mergeKey = len(clusterOutput.keys())

    ## create a new-key to store updated details: these details must also reflect eventVector centroids
    repDocs[mergeKey] = {}
    repDocs[mergeKey]['docs'] = repDocs[cIndexPair[0]]['docs'] + repDocs[cIndexPair[1]]['docs']
    repDocs[mergeKey]['docIds'] = repDocs[cIndexPair[0]]['docIds'] + repDocs[cIndexPair[1]]['docIds']
    repDocs[mergeKey]['docPositions'] = repDocs[cIndexPair[0]]['docPositions'] + repDocs[cIndexPair[1]]['docPositions']
    repDocs[mergeKey]['cosSim'] = repDocs[cIndexPair[0]]['cosSim'] + repDocs[cIndexPair[1]]['cosSim']
    
    ## update centroid: take mean of two cluster-centroids along with generated summaries; 
    label1, label2 = extractClusterLabel(clusterOutput[cIndexPair[0]]['label']), extractClusterLabel(clusterOutput[cIndexPair[1]]['label'])
    encodedLabels = model.encode([label1, label2], convert_to_tensor=True, device="cuda:0")
    encodedLabels = l2_normalize(encodedLabels.cpu().numpy())
    eventVectors = np.append(eventVectors, np.vstack([eventVectors[cIndexPair[0]], 
                                                      eventVectors[cIndexPair[1]],
                                                      encodedLabels[0],
                                                      encodedLabels[1]]).mean(axis=0).reshape(1, -1), axis=0)
    
    ## recompute distances and update repDocs
    repDocs[mergeKey] = getRepresentativeDocs(eventVectors, eventData, mergeKey, 
                          normalizedEmbeddings[repDocs[mergeKey]['docPositions']], 
                          repDocs[mergeKey]['docPositions'], eventRelatedData)

    ## generate updatedLabel
    newsArticle2 = ""
    for d in range(1, len(repDocs[mergeKey]['docs'])):
        if repDocs[mergeKey]['docs'][d] != repDocs[mergeKey]['docs'][0]:
            newsArticle2 = repDocs[mergeKey]['docs'][d]
            break
    testMessage = "News Article1: {newsArticle1}\n\nNews Article2: {newsArticle2}".format(newsArticle1=repDocs[mergeKey]['docs'][0], 
                                                                                    newsArticle2=newsArticle2)
    response, _ = generateClusterLabel(testMessage)
    finalLabel = "{updatedLabel}###{label1}###{label2}".format(updatedLabel=response, 
                                                               label1=clusterOutput[cIndexPair[0]]["label"], 
                                                               label2=clusterOutput[cIndexPair[1]]["label"])
    clusterOutput[mergeKey] = {"label": finalLabel, "documentClusterIndices": repDocs[mergeKey]['docPositions'], "documentIds": repDocs[mergeKey]['docIds']}
    
    ## then re-index by arranging the clusters based on their decreasing size:
    oldNewIndexMapper = {}
    repDocsKeys = sorted(repDocs, key=lambda k: len(repDocs[k]['docs']), reverse=True)
    for key in repDocsKeys:
        if key in cIndexPair:
            continue
        
        oldNewIndexMapper[key] = newClusterKey

        newRepDocs[newClusterKey] = repDocs[key]
        newClusterOutput[newClusterKey] = clusterOutput[key]
        newEventVectors = np.append(newEventVectors, eventVectors[key].reshape(1, -1), axis=0)
        newClusterKey += 1
    

    return newRepDocs, newClusterOutput, newEventVectors, oldNewIndexMapper


def getRepresentativeDocs(eventVectors, eventData, label, docEmbeddings, labelPos, eventRelatedData):
    cosScores = util.cos_sim(eventVectors[label].reshape(1, -1), docEmbeddings)
    topKDocIndices = torch.argsort(cosScores, descending=True).reshape(-1, 1)
    cosScores = cosScores[0]
    docs = {'docs': [], 'docIds': [], 'docPositions': [], 'cosSim': []}
    for index, i in enumerate(topKDocIndices):
        # if len(docs['docs']) > 0 and docs['docs'][-1] == eventData[labelPos[i.item()]]:
        #     continue
        docs['docs'].append(eventData[labelPos[i.item()]])
        docs['docIds'].append(eventRelatedData[labelPos[i.item()]]['id'])
        docs['docPositions'].append(int(labelPos[i.item()]))
        docs['cosSim'].append(cosScores[i.item()].item())
    return docs


def coreClusteringMethod(eventData, eventRelatedData, model):
    encodedDocs = model.encode(eventData, show_progress_bar=True, convert_to_tensor=True, device="cuda:0")
    numDataPoints = encodedDocs.shape[0]
    normalizedEmbeddings = l2_normalize(encodedDocs.cpu())
    hdbParamdist = {'min_samples': [2, 3, 5, 10],
                'min_cluster_size': [15, int(np.ceil(.05*numDataPoints)), int(np.ceil(.1*numDataPoints)), int(np.ceil(.2*numDataPoints)),
                                        int(np.ceil(.25*numDataPoints))],  
                'cluster_selection_method' : ['eom'],
                'metric' : ['euclidean'] }
    umapModel, hdbModel, bestScore, bestParameters = searchClusterParameters(normalizedEmbeddings, hdbParamdist)
    ## form event-vectors;
    uniqueLabels = set(hdbModel.labels_)
    if -1 in uniqueLabels:
        uniqueLabels.remove(-1)
    uniqueLabels = sorted(list(uniqueLabels))
    # uniqueLabels = [label for label in uniqueLabels if len(np.where(hdbModel.labels_ == label)[0]) > 3]
    eventVectors = l2_normalize(np.vstack([normalizedEmbeddings[np.where(hdbModel.labels_ == label)[0]].mean(axis=0) for label in uniqueLabels]))

    ## get most representative documents;
    ### get top-3 representative documents for each cluster -- based on cosine-similarity;
    repDocs = {}
    coverage = 0.0
    for label in uniqueLabels:
        labelPos = np.where(hdbModel.labels_ == label)[0]
        coverage = coverage + len(labelPos)
        docEmbed = normalizedEmbeddings[labelPos]
        repDocs[label] = getRepresentativeDocs(eventVectors, eventData, label, docEmbed, labelPos, eventRelatedData)

    coverage = coverage / len(hdbModel.labels_)

    return umapModel, hdbModel, bestScore, bestParameters, eventVectors, repDocs, coverage, normalizedEmbeddings


def checkForInCoherency(model, clusterSummary=None, docEmbed=None):
    """Rationale: Top-3 documents other than the two documents used to generate summary should all match the cluster description.
        If not, these clusters are rejected. 
    """
    incoherent = False  # not incoherent

    clusterSummary = extractClusterLabel(clusterSummary)
    summaryEmbed = model.encode(clusterSummary, convert_to_tensor=True, device="cuda:0")
    summaryEmbed = l2_normalize(summaryEmbed.cpu().numpy())
    
    cosScores = util.cos_sim(summaryEmbed, docEmbed.astype(np.float32))
    
    for val in cosScores[0]:
        if val.item() < 0.6:
            incoherent = True
            print("Incoherent Cluster!")
            break

    return incoherent

def reIndexClusterOut(clusterOutput, repDocs, eventVectors, ignoreList):
    newClusterOutput = {}
    newEventVectors = np.empty((0, eventVectors.shape[1]), dtype=np.float64)
    newRepDocs = {}
    
    repDocsKeys = sorted(repDocs, key=lambda k: len(repDocs[k]['docs']), reverse=True)

    newKey = 0
    for e, key in enumerate(repDocsKeys):
        if key not in ignoreList:
            newRepDocs[newKey] = repDocs[key]
            newEventVectors = np.append(newEventVectors, eventVectors[key].reshape(1, -1), axis=0)
            newClusterOutput[newKey] = clusterOutput[key]
            newKey += 1

    return newClusterOutput, newRepDocs, newEventVectors


def clusterEventData(eventId, analysisInterval, issueName, eventData, eventRelatedData, model, outputDir):
    """
    Cluster the documents with a very high-cosine similarity score to get most representative documents for a set of events;
    """
    clusterMergeCount = 0
    clusterRemovalCount = 0
    issueFolder = "{interval}_day_{issueName}".format(interval=analysisInterval, issueName=issueName)
    clusterSavePathDir = [outputDir, issueFolder, "newsEventDetection", "mergedCluster", "peak" + str(eventId)]
    if not os.path.isdir(os.path.join(*clusterSavePathDir)):
        success = createFolder(os.path.join(*clusterSavePathDir))
        if success == -1:
            print("Output Folder Creation Error!")
            return
    clusterSaveTensorPath = os.path.join(*(clusterSavePathDir + ["encodedEventVectors.npy"]))

    clusterSavePath = os.path.join(*(clusterSavePathDir + ["clusterInformation.json"]))

    if os.path.isfile(clusterSavePath):
        savedOutput = readJson(clusterSavePath)
        with open(clusterSaveTensorPath, 'rb') as f:
            encodedDocs = np.load(f)
        f.close()
        return savedOutput, encodedDocs, {"tensorSavePath": clusterSaveTensorPath, "clusterInfoSavePath": clusterSavePath}

    print("Encoding documents and clustering based on HDBSCAN algorithm...")
    # ## try HDBSCAN here for clustering data 
    ## actual 'docs' within repDocs are arranged in descending order of cosine similarity with the cluster centroid;
    umapModel, hdbModel, bestScore, bestParameters, eventVectors, repDocs, coverage, normalizedEmbeddings = coreClusteringMethod(eventData, eventRelatedData, model)

    if len(repDocs) < 1:
        return None, None, None
    
    ## arrange clusters in decreasing order of cluster sizes;
    repDocs, eventVectors, pruneFlag = pruneEventCount(repDocs, eventVectors)

    if pruneFlag:
        print("Issue Name: {issueName}\tEvent-Id {eventId} --> Potential event-count {eventCount} --> DBVC Score (before pruning) {dbvcScore}".format(issueName=issueName, 
                                                                                                      eventId=eventId,
                                                                                                      eventCount=len(repDocs),
                                                                                                      dbvcScore=hdbModel.relative_validity_))
    else:
        print("Issue Name: {issueName}\tEvent-Id {eventId} --> Potential event-count {eventCount} --> DBVC Score {dbvcScore}".format(issueName=issueName, 
                                                                                                        eventId=eventId,
                                                                                                        eventCount=len(repDocs),
                                                                                                        dbvcScore=hdbModel.relative_validity_))

    clusterOutput = {}
    incoherentClusterIndices = []
    for cIndex, docDict in repDocs.items():
        removeFlag = False
        newsArticle2 = ""
        for d in range(1, len(docDict['docs'])):
            if docDict['docs'][d] != docDict['docs'][0]:
                newsArticle2 = docDict['docs'][d]
                break
        testMessage = "News Article1: {newsArticle1}\n\nNews Article2: {newsArticle2}".format(newsArticle1=docDict['docs'][0], 
                                                                                        newsArticle2=newsArticle2)
        response, _ = generateClusterLabel(testMessage)
        removeFlag = checkForInCoherency(model, clusterSummary=response, docEmbed=normalizedEmbeddings[docDict['docPositions']][:5])
        if removeFlag:
            incoherentClusterIndices.append(cIndex)
        clusterOutput[cIndex] = {"label": response, "documentClusterIndices": docDict['docPositions'], "documentIds": docDict['docIds']}


    ## re-index before starting to merge clusterOutputs;
    if len(incoherentClusterIndices) > 0:
        clusterOutput, repDocs, eventVectors = reIndexClusterOut(clusterOutput, repDocs, eventVectors, incoherentClusterIndices)
        clusterRemovalCount = len(incoherentClusterIndices)
    
    mergeIterationCounter = 0
    while mergeIterationCounter < 2: ## run for 2 iterations;
        mergeFlag = False
        mergeIndexMapper = []
        
        clusterIndexCombinations = list(itertools.combinations(list(clusterOutput.keys()), 2))
        ignoreVertexSet = set()

        for cIndexPair in clusterIndexCombinations:
            if cIndexPair[0] in ignoreVertexSet or cIndexPair[1] in ignoreVertexSet:
                continue
            
            cIndI, cIndJ = cIndexPair[0], cIndexPair[1]
            for indexMapper in mergeIndexMapper:
                cIndI = indexMapper[cIndI]
                cIndJ = indexMapper[cIndJ]

            label1, label2 = clusterOutput[cIndI]['label'], clusterOutput[cIndJ]['label']
            ## check if these are the same; 
            response, _ = checkLabelEquivalence(label1.split("###")[0], label2.split("###")[0])
            ## merge
            if "yes" in response.lower():
                clusterMergeCount += 1
                mergeFlag = True
                # print("Merging the following news-events:\n{label1}\n\n{label2}\n".format(label1=label1, label2=label2))
                repDocs, clusterOutput, eventVectors, oldNewMapper = mergeAndReIndex((cIndI, cIndJ), clusterOutput, repDocs, 
                                                                                eventVectors, model, eventData, normalizedEmbeddings, eventRelatedData)
                mergeIndexMapper.append(oldNewMapper)
                ignoreVertexSet.add(cIndexPair[0])
                ignoreVertexSet.add(cIndexPair[1])
        
        ## if in case of no merges -- break out of the loop;
        if not mergeFlag:
            break
        mergeIterationCounter += 1

    toSave = {eventId: {"event_related_data": eventRelatedData, "event_data_content": eventData,
                "clusterOutput": clusterOutput, "representative_docs": repDocs,
                "clusterMergeCount": clusterMergeCount, "clusterRemovalCount": clusterRemovalCount}}
    
    
    with open(clusterSaveTensorPath, 'wb') as f:
        np.save(f, eventVectors)
    f.close()

    writeJson(toSave, clusterSavePath, indent=1)

    return toSave, eventVectors, {"tensorSavePath": clusterSaveTensorPath, "clusterInfoSavePath": clusterSavePath}


def clusterAndRefineEventCandidates(args):
    global openAIKeyStr
    openAIKeyStr = args.openai_api_key
    outputDir = args.output_dir
    analysisInterval = args.analysis_interval
    issueList = args.issues_of_interest
    contextWindowAroundPeak = args.peak_context_window
    model = SentenceTransformer(args.similarity_model)

    issueSpecificDataPath = os.path.join(*[outputDir, "issueSpecificData.json"])
    print("Reading Issue-Specific Data (for all issues)...")
    issueSpecificData = readJson(issueSpecificDataPath)
    print("Reading NELA-data2021 from relevant sources...")
    nelaData = readJson(os.path.join(*[outputDir, "ent_augment_data.json"]))
    _, issueDataFastAccess = processIssueRelData(issueSpecificData, nelaData)

    relevantSources = readJson(args.relevant_sources)
    relevantSourceList = [sourceDict['source_name'] for sourceDict in relevantSources.values()]
    leftSourceList = [sourceDict['source_name'] for sourceDict in relevantSources.values() if sourceDict['source_bias_rating'] == "left"]
    rightSourceList = [sourceDict['source_name'] for sourceDict in relevantSources.values() if sourceDict['source_bias_rating'] == "right"]
    noBiasSourceList = [sourceDict['source_name'] for sourceDict in relevantSources.values() if sourceDict['source_bias_rating'] == "center"]
    conspiracySourceList = [sourceDict['source_name'] for sourceDict in relevantSources.values() if sourceDict['source_bias_rating'] == "conspiracy_pseudoscience"]
    
    sourceDict = {"left": leftSourceList, "right": rightSourceList, "center": noBiasSourceList, "conspiracy": conspiracySourceList}

    totalDocCount = {}
    totalUniqueDocCount = {}
    for issueName in issueList:
        coveredDocCount = 0
        totalDocCount[issueName] = 0.0
        totalUniqueDocCount[issueName] = 0.0
        ## get all news-articles related to an event corresponding to the issue
        peakDocs, peakDocIds, eventInfo = getPeakData(issueName, analysisInterval, outputDir, nelaData, 
                                                            issueDataFastAccess[issueName], sourceDict)
        
        eventDocs, eventDocIds, _ = getPeakData(issueName, analysisInterval, outputDir, nelaData, 
                                                            issueDataFastAccess[issueName], sourceDict, contextWindow=contextWindowAroundPeak)
        
        eventCoverage = {}
        totalMergeCount = 0
        totalClusterRemovalCount = 0
        for eventId, groupedArticles in eventDocs.items():
            peakRelatedData = peakDocs[eventId]['left'] + peakDocs[eventId]['right'] + peakDocs[eventId]['center'] + peakDocs[eventId]['conspiracy_pseudoscience']
            eventRelatedData = eventDocs[eventId]['left'] + eventDocs[eventId]['right'] + eventDocs[eventId]['center'] + eventDocs[eventId]['conspiracy_pseudoscience']
            print("Peak Document Count for {issueName} - event number {eventId} is: {dCount}".format(issueName=issueName, eventId=eventId, dCount=len(peakRelatedData)))
            print("Document Count around peak for {issueName} - event number {eventId} is: {dCount}".format(issueName=issueName, eventId=eventId, dCount=len(eventRelatedData)))
            
            totalDocCount[issueName] = totalDocCount.get(issueName, 0.0) + len(eventRelatedData)
            totalUniqueDocCount[issueName] = totalUniqueDocCount.get(issueName, 0.0) + len(eventDocIds[eventId])
            
            print("Tokenizing Article...")
            eventData = [x['title'] + "\n" + x['content'] for x in eventRelatedData]

            peakData = [x['title'] + "\n" + x['content'] for x in peakRelatedData]
            tempPeakData = ["".join(tokenizeArticle(x)).split("\n") for x in peakData]
            peakData = ["\n".join(x[:4]) for x in tempPeakData]
            # use gpt-3.5-turbo to generate summary/label for these clusters;
            clusterOutput, clusterCentroidEmbed, clusterSavePathDict = clusterEventData(eventId, analysisInterval, issueName, peakData, peakRelatedData, model, outputDir)
            if clusterOutput is None or len(clusterOutput[eventId]['clusterOutput']) == 0:
                continue
            ## query gpt for identifying entailment between all possible cluster summary/label pairs
            ### for each cluster output -- 
                ### merge if the clusters are part of the same event ;
            ## consider embeddings -- with cluster title - description ; and the news-articles as well;
            ## identify most semantically similar documents that are above a certain lesser threshold and mark them as visited;
            tempEventData = ["".join(tokenizeArticle(x)).split("\n") for x in eventData]
            eventData = ["\n".join(x[:4]) for x in tempEventData]
            
            docEmbeddings = model.encode(eventData, show_progress_bar=True, convert_to_tensor=True, device="cuda:0")
            docEmbeddings = l2_normalize(docEmbeddings.cpu().numpy())

            ## try doing entailment here -- through similarity module
            ## embed cluster summaries:
            clusterLabelDoc = [""] * len(clusterOutput[eventId]['clusterOutput'])
            for cIndex, clusterDict in clusterOutput[eventId]['clusterOutput'].items():
                ## basic cleaning:
                tempVar = clusterDict['label'].replace("###News Event Title:", "")
                tempVar = tempVar.replace("News Event Title:", "")
                tempVar = tempVar.replace("News Event Description:", "").strip()

                clusterLabelDoc[int(cIndex)] = tempVar
            
            clusterLabelEmbed = model.encode(clusterLabelDoc, show_progress_bar=True, convert_to_tensor=True, device="cuda:0")
            clusterLabelEmbed = l2_normalize(clusterLabelEmbed.cpu().numpy())
            
            cosScores = util.cos_sim(clusterLabelEmbed, docEmbeddings)
            cosScoresEventClass = torch.argmax(cosScores, dim=0)

            ## associate documents to these centroids
            finalClusterOutput = {k: {'label': clusterOutput[eventId]['clusterOutput'][k]['label'], 
                                    'documentIds': [], 
                                    'articles': []} for k in clusterOutput[eventId]['clusterOutput'].keys()}
            finalClusterOutputKeyInstance =  list(finalClusterOutput.keys())[0]
            mappedDocCount = 0
            for docIndex, cId in enumerate(cosScoresEventClass):
                if cosScores[cId][docIndex].item() > 0.69:
                    mappedDocCount += 1
                    if isinstance(finalClusterOutputKeyInstance, str):
                        finalClusterOutput[str(cId.item())]['articles'].append(eventRelatedData[docIndex])
                        finalClusterOutput[str(cId.item())]['documentIds'].append(eventRelatedData[docIndex]['id'])
                    else:
                        finalClusterOutput[cId.item()]['articles'].append(eventRelatedData[docIndex])
                        finalClusterOutput[cId.item()]['documentIds'].append(eventRelatedData[docIndex]['id'])
            
            coveredDocCount = coveredDocCount + mappedDocCount
            print("Issue Name: {issueName}\tEvent-Id {eventId} --> Coverage {coverage} --> \
                  clusterMergeCount --> {mc} --> clusterRemovalCount --> {cr}".format(issueName=issueName, 
                                                                                        eventId=eventId,
                                                                                        coverage=mappedDocCount/len(eventData),
                                                                                        mc=clusterOutput[eventId]['clusterMergeCount'],
                                                                                        cr=clusterOutput[eventId]['clusterRemovalCount']))
            totalMergeCount = totalMergeCount + clusterOutput[eventId]['clusterMergeCount']
            totalClusterRemovalCount = totalClusterRemovalCount + clusterOutput[eventId]['clusterRemovalCount']

            eventCoverage[eventId] = mappedDocCount/len(eventData)
            issueFolder = "{interval}_day_{issueName}".format(interval=analysisInterval, issueName=issueName)
            retainedClusterOutput = {}
            ## retain clusters with more than 10 documents --> removed for now.
            for cIndex, clusterDict in finalClusterOutput.items():
                clusterSavePathDir = [outputDir, issueFolder, "newsEventDetection", "newsEvents", "peak" + str(eventId)]
                success = createFolder(os.path.join(*clusterSavePathDir))
                savePath = os.path.join(*(clusterSavePathDir + ["event{cIndex}.json".format(cIndex=cIndex)]))
                retainedClusterOutput.update({cIndex: clusterDict})

                writeJson({cIndex: clusterDict}, path=savePath, indent=1)

            if len(retainedClusterOutput) > 0:
                clusterSavePathDir = [outputDir, issueFolder, "newsEventDetection", "newsEvents", "peak" + str(eventId)]
                savePath = os.path.join(*(clusterSavePathDir + ["peakEventsAll.json".format(cIndex=cIndex)]))
                writeJson(retainedClusterOutput, path=savePath, indent=1)

        print("Issue --> {issueName}\tAverage. Coverage Per Event --> {avgCoverage}\t --> Total Doc Count {totalDocCount}\t --> Total Unique Doc Count {totalUniqueCount}\t Covered Doc Count {coveredDocCount}\t Total Merge Count {mc}\t Total Cluster Removal Count {cr}".format(
            issueName=issueName, avgCoverage=sum(eventCoverage.values())/len(eventCoverage),
            totalDocCount=totalDocCount[issueName], totalUniqueCount=totalUniqueDocCount[issueName], coveredDocCount=coveredDocCount,
            mc=totalMergeCount, cr=totalClusterRemovalCount))





