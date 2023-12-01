from tulip import *
from math import *
from random import *
import os
import sys

#for specific imports as numpy
#sys.path.append("/usr/lib/python2.6/dist-packages/PIL")
#sys.path.append("/usr/lib/python2.6/dist-packages")
#sys.path.append("C:\Users\melancon\Documents\Dropbox\OTMedia\lighterPython")
#sys.path.append("/home/brenoust/Dropbox/OTMedia/lighterPython")

import os

dir = "C:\Users\melancon\Documents\Dropbox\OTMedia\lighterPython"
if not os.path.isdir(dir) :
	dir = "/work/svn/renoust/lighterPython"

if not os.path.isdir(dir) :
	dir = "/home/brenoust/Dropbox/OTMedia/lighterPython"

if not os.path.isdir(dir) :
	print "cannot find the required library folder"

import datetime
import numpy as np

from coherenceComputationLgt import *

'''
Computes the coherence metric from a specifically formatted graph and
also offers the possibility to synchronize selections with a subgraph and its dual
from selecting types in the dual graph.

The graph must be multilayer with each layer of the edge formatted as "nameSimilarity"
where name is the name of the layer (in our case, a descriptor).
The list of all names must contained in a text file specified in RCMN.loadWordList().

After patching over patches, the code has been poorly redesigned to apply to all 
subgraphs available in the current graph selection.

'''

'''
'''
class clusterAnalysisLgt():
	'''
	Initializes some parameters, and starts the analysis
	'''
	def __init__(self, _graph, _descriptorList):
	
		# the label to detect the layer of a descriptor in links
		self.linkId = "Similarity"
		# the full list of possible descriptors
		self.descriptorList = _descriptorList
		# the original graph
		self.graph = _graph


		# the number of relevant descriptors
		self.nbTypes = 0
		# the list of descriptors
		self.typeList = []
		# the raw matrix of co-occurences
		self.rawMatrix = [[]]
		# the probability matrix of co-occurences
		self.cMatrix = [[]]
		

		self.isComplex = False
		
		# data for cluster stats
		# the date when the first and last document of the cluster appears
		self.dateBegin = 999999999.0
		self.dateEnd = 0.0
		self.globalCoherence = 0.0
		self.globalCosine = 0.0
		self.nbDocuments = 0.0
		self.nbDocLinks = 0.0
		self.nbTypeLinks = 0.0
		self.documentsDensity = 0.0
		self.descriptorsDensity = 0.0
		self.typeToIntrications = {}
		self.name = ""
		self.gravityCenter = [0,0,0]
		
		

		# a map from descriptors to their number of occurences in links
		self.typeToOccurence = {}

		# the graph of descriptors (description graph)
		self.typeGraph = tlp.newGraph()
		self.tName = self.typeGraph.getStringProperty("typeName")

		# desciption graph parameters
		self.nbTypeComponents = 1
		self.typeIsConnected = True
		# fills the typeList
		self.getCurrentDescriptors()
		if len(_descriptorList)>1:
			self.typeList = _descriptorList
		
		# initializations
		self.nbTypes = len(self.typeList)
		
		
		#/!\ what do we do if nbTypes < 2 ?
		if self.nbTypes > 1:

			self.analyse()
			self.globalStats()
			self.visualize()				
		else:
			self.nbTypeComponents = 0
			#print "the cluster has less than 2 types"
		

	'''
	gets the current relevant descriptors by looking for each descriptor in the list 
	whether they exist as properties having at least an edge value>0 in the considered graph
	'''
	def getCurrentDescriptors(self):
		
		'''		
		for t in self.descriptorList:
			if self.graph.existProperty(t+self.linkId):
				sim = self.graph.getDoubleProperty(t+self.linkId)
				if sim.getEdgeMax(self.graph) > 0.0:
					self.typeList.append(t)	
		'''
		
		self.typeToOcc = {}
		self.cotypeToOcc = {}
		foundTypes = set()

		descP = self.graph.getStringProperty("descripteurs")

		for e in self.graph.getEdges():
			dList = descP[e].split(";")
			foundTypes.update(dList)
			
			for i in range(len(dList)):
				d1 = dList[i]
				if d1 not in self.typeToOcc:
					self.typeToOcc[d1] = 0.0
				self.typeToOcc[d1] += 1.0

				for j in range(i+1, len(dList)):
					d2 = dList[j]
					key = frozenset([d1, d2])
					if key not in self.cotypeToOcc:
						self.cotypeToOcc[key] = 0.0
					self.cotypeToOcc[key] += 1.0
				
		self.typeList = list(foundTypes)
		print self.typeList

		#on voudra peut etre inclure la liste des types dans la structure du cluster?
		#utiliser des stringVector?



	'''
	builds the raw matrix (considering an edge value of a descriptor is either 0 or 1)
	it also fills by the way the map of types to their occurences and helps initializing
	the probability matrix
	'''		
	def buildRawMatrix(self):

		for i in range(self.nbTypes):
			typeI = self.typeList[i] 
			nI = self.typeToOcc[typeI]
			self.typeToOccurence[typeI] = nI
			self.rawMatrix[i][i] = nI
			self.cMatrix[i][i] = nI

			for j in range(i+1,self.nbTypes):
				typeJ = self.typeList[j]
				key = frozenset([typeI, typeJ])
				nIJ = 0.0
				if key in self.cotypeToOcc:
					nIJ = self.cotypeToOcc[key]

				self.rawMatrix[i][j] = nIJ
				self.rawMatrix[j][i] = nIJ
				self.cMatrix[i][j] = nIJ
				self.cMatrix[j][i] = nIJ
		
		'''
		for i in range(self.nbTypes):

			typeI = self.typeList[i] 
			simI = self.graph.getDoubleProperty(typeI+self.linkId)
			edgesI = tlp.IteratorEdge			

			if simI.getEdgeDefaultValue() == 0.0:
				edgesI = simI.getNonDefaultValuatedEdges(self.graph)
			else:
				edgesI = simI.getDefaultValuatedEdges(self.graph)

			edgeListI = [e for e in edgesI]
			nI = float(len(edgeListI))
			self.typeToOccurence[typeI] = nI
			self.rawMatrix[i][i] = nI
			self.cMatrix[i][i] = nI
			
			for j in range(i+1,self.nbTypes):
				typeJ = self.typeList[j]
				simJ = self.graph.getDoubleProperty(typeJ+self.linkId)
				edgesJ = tlp.IteratorEdge			

				if simJ.getEdgeDefaultValue() == 0.0:
					edgesJ = simJ.getNonDefaultValuatedEdges(self.graph)
				else:
					edgesJ = simJ.getDefaultValuatedEdges(self.graph)

				edgeListJ = [e for e in edgesJ]
				inter = set(edgeListI) & set(edgeListJ)
				
				nIJ = float(len(inter))
				self.rawMatrix[i][j] = nIJ
				self.rawMatrix[j][i] = nIJ
				self.cMatrix[i][j] = nIJ
				self.cMatrix[j][i] = nIJ
				
		#print self.typeToOccurence
		'''


	'''
	builds the probability matrix of co-occurences of descriptors on links
	rMatrix: the raw matrix to build the probability matrix with
	cMatrix: the probability matrix
	nbTypes: the number of involved types
	nEdges: the number of edges for normalization	
	'''
	def buildCMatrix(self, rMatrix, cMatrix, nbTypes, nEdges):

		nbDoc = self.graph.numberOfNodes()
		for i in range(nbTypes):
			cMatrix[i][i] /= nEdges
			#self.anotherCMatrix[i][i] /= nbDoc*(nbDoc - 1)/2
		
		for i in range(nbTypes):
			for j in range(i+1, nbTypes):
				cMatrix[i][j] /= rMatrix[j][j] 
				cMatrix[j][i] /= rMatrix[i][i]
				#self.anotherCMatrix[i][j] = cMatrix[i][j]
				#self.anotherCMatrix[j][i] = cMatrix[j][i]
				
		
		#for i in range(nbTypes):
		#	for j in range(nbTypes):
		#		self.anotherCMatrix[i][j] = self.cMatrix[j][i]
		
		#for i in range(nbTypes):
		#	for j in range(nbTypes):
		#		self.cMatrix[i][j] = self.anotherCMatrix[i][j] 

	'''
	builds the graph of descriptors and fills some properties such as descriptors names 
	and labels, and occurences
	gets the number of components of the description graph
	'''
	def buildTypeGraph(self):

		self.typeToNode = {}
		
		tLabel = self.typeGraph.getStringProperty("viewLabel")
		tOccurence = self.typeGraph.getDoubleProperty("occurence")

		for i in range(self.nbTypes):
			nI = self.typeGraph.addNode()
			typeI = self.typeList[i]
			self.typeToNode[typeI] = nI
			self.tName.setNodeValue(nI, typeI)
			tLabel.setNodeValue(nI, typeI)
			tOccurence.setNodeValue(nI, self.typeToOccurence[typeI])			
			
		for i in range(self.nbTypes):
			typeI = self.typeList[i]
			nI = self.typeToNode[typeI]
			for j in range(i+1, self.nbTypes):
				typeJ = self.typeList[j]
				nJ = self.typeToNode[typeJ]
				occ = self.rawMatrix[i][j]
				if occ > 0.0:
					e = self.typeGraph.addEdge(nI, nJ)
					tOccurence.setEdgeValue(e,occ)

		#/!\we can also get the list of lists of nodes, or just get the number of nodes
		self.typeIsConnected = tlp.ConnectedTest.isConnected(self.typeGraph)
		


	'''
	builds matrices and types for each description connected component
	'''
	def cutComponents(self):
		self.components = tlp.ConnectedTest.computeConnectedComponents(self.typeGraph)
		self.nbTypeComponents = len(self.components)
		'''
		print "########################"
		print self.typeList
		print self.cMatrix
		'''
		self.indexLists = []

		for c in self.components:
			indexList = []
			subRawMatrix = [[0.0] *len(c) for x in range(len(c))]
			subCMatrix = [[0.0] * len(c) for x in range(len(c))]		
			typeMap = []

			# size of the component we will consider (at least 4 descriptors)
			# but "at least 2" might also interest us
			if len(c) > 1: 
				occ = [0.0]
				if len(c) < 4:
					#gets the number of links related to each type					
					for n in c:
						nName = self.tName.getNodeValue(n)
						occ.append(self.typeToOccurence[nName])
				
				# at least 4 descriptors or 3 links in the documents
				if len(c) > 3 or max(occ) > 2:
				

					for n in c:
						nName = self.tName.getNodeValue(n)
						indexList.append(self.typeList.index(nName))
						typeMap.append(nName)
			
					for i in range(len(indexList)):
						subRawMatrix[i][i] = self.rawMatrix[indexList[i]][indexList[i]]
						subCMatrix[i][i] = self.rawMatrix[indexList[i]][indexList[i]]
						for j in range(i+1,len(indexList)):
							subRawMatrix[i][j] = self.rawMatrix[indexList[i]][indexList[j]]
							subRawMatrix[j][i] = self.rawMatrix[indexList[j]][indexList[i]]
							subCMatrix[i][j] = self.rawMatrix[indexList[i]][indexList[j]]
							subCMatrix[j][i] = self.rawMatrix[indexList[j]][indexList[i]]
				
					self.sRMatrices.append(subRawMatrix)
					self.sCMatrices.append(subCMatrix)
					self.typeMaps.append(typeMap)
					self.indexLists.append(indexList)
				
			
				
			# else: /!\ what do we do else with the "useless" components?
			# delete the links in the documens graph?

		for t in range(len(self.sRMatrices)):
			print self.typeMaps[t]
			print self.indexLists[t]
			print self.sRMatrices[t]
			
		print "########################"	

	def computeNbMultiEdges(self):
		return self.graph.numberOfEdges()
		edges = []
		nbSimP = self.graph.getDoubleProperty("nbSimilarities")
		nbSim = [nbSimP.getEdgeValue(e) for e in self.graph.getEdges()]
		s = sum(nbSim)
		if s == 0:
			print "WARNING: need to set the property nbSimilarities"
		return s
		
		#for e in self.graph.getEdges():
		#	nbSim = 
		#	edges.append(nbSim.getEdgeValue(e))

	def analyse(self):

		self.cMatrix = [[0.0]*self.nbTypes for x in range(self.nbTypes)]
		self.rawMatrix = [[0.0]*self.nbTypes for x in range(self.nbTypes)]
		

		# creates the raw matrix
		self.buildRawMatrix()
		
		#/!\ detect diagonal blocks (laplacian, Dulmage-Mendelsohn decomposition, build the type graph)
		# creates the descriptors graph
		self.buildTypeGraph()

		#/!\controls the number connected components
		# if there are more than one connected component
		if not self.typeIsConnected:
			# build arrays of types
			# build arrays of rawMatrices and cMatrices
			self.components = []
			self.sRMatrices = []
			self.sCMatrices = []
			self.typeMaps = []

			self.cutComponents()

			if len(self.sRMatrices) > 1:

				coherences = []
				
				nbEdges = self.graph.numberOfEdges()#self.computeNbMultiEdges()
				
				# number of edges involved?
				for c in range(len(self.sRMatrices)):
					self.buildCMatrix(self.sRMatrices[c], self.sCMatrices[c], len(self.typeMaps[c]), nbEdges)
					coherences.append(coherenceComputationLgt(self.typeMaps[c], self.sRMatrices[c], self.sCMatrices[c]))
				
				for c in coherences :
					self.typeToIntrications.update(c.intricationValues)
					print c.rawMatrix
					print c.cMatrix
					print c.coherenceMetric
					print c.intricationValues
					print c.cosineDistance
				
				nTypesToComp = {x.nbTypes:x for x in coherences}
				maxNTypes = max(nTypesToComp.keys())
				self.globalCosine = nTypesToComp[maxNTypes].cosineDistance
				self.globalCoherence = nTypesToComp[maxNTypes].coherenceMetric
				
				#self.globalCoherence = sum([x.coherenceMetric*x.nbTypes/self.nbTypes for x in coherences])
				#self.globalCosine = sum([x.cosineDistance*x.nbTypes/self.nbTypes for x in coherences])
				
				
				#self.anotherGlobalCosine = 0.0
				if True in [x.complexResult for x in coherences]:
					self.isComplex = True

				#/!\treat small types, delete small components?		
			elif len(self.sRMatrices) == 1:
				self.typeIsConnected = True
				self.rawMatrix = self.sRMatrices[0]
				self.cMatrix = self.sCMatrices[0]
				self.nbTypes = len(self.typeMaps[0])
				self.typeList = self.typeMaps[0]
		
		# creates the probability matrix if there is only 1 connected component		
		if self.typeIsConnected:
			#self.buildCMatrix(self.rawMatrix, self.cMatrix, self.nbTypes, self.computeNbMultiEdges())
			self.buildCMatrix(self.rawMatrix, self.cMatrix, self.nbTypes, self.graph.numberOfEdges())
			coherence = coherenceComputationLgt(self.typeList, self.rawMatrix, self.cMatrix)
			self.globalCoherence = coherence.coherenceMetric
			self.globalCosine = coherence.cosineDistance
			self.typeToIntrications = coherence.intricationValues
			self.isComplex = coherence.complexResult
			#coherence = coherenceComputationLgt(self.typeList, self.rawMatrix, self.anotherCMatrix)
			#self.anotherGlobalCoherence = coherence.coherenceMetric
			#self.anotherGlobalCosine = coherence.cosineDistance
			#self.anotherTypeToIntrications = coherence.intricationValues
			#self.anotherIsComplex = coherence.complexResult


	def globalStats(self):
		self.nbDocuments = self.graph.numberOfNodes()
		self.nbDocLinks = self.graph.numberOfEdges()
		self.nbTypeLinks = self.typeGraph.numberOfEdges()
		if self.nbTypes > 1:
			self.descriptorsDensity = self.nbTypeLinks/(self.nbTypes * (self.nbTypes-1.0))
		else :
			self.descriptorsDensity = 0.0

		if self.nbDocuments > 1:
			self.documentsDensity = sum(self.typeToOccurence.values())
			self.documentsDensity /= self.nbTypes*self.nbDocuments*(self.nbDocuments-1.0)
		else:
			self.documentsDensity = 0.0
	
		intr = self.typeGraph.getDoubleProperty("intrication")
		complexP = self.typeGraph.getBooleanProperty("complex")
			
		for t in self.typeList:
			n = self.typeToNode[t]
			if t in self.typeToIntrications:
				if(np.iscomplex(self.typeToIntrications[t])):
					complexP.setNodeValue(n, True)
				intr.setNodeValue(n, self.typeToIntrications[t])
			else:
				intr.setNodeValue(n, 0.0)
	
		# gets the min and max dates from the documents
		# gets the gravity center
		dateP = self.graph.getIntegerProperty("date")
		viewLayout = self.graph.getLayoutProperty("viewLayout")
		for n in self.graph.getNodes():
			date = dateP.getNodeValue(n)
			if self.dateBegin > date:
				self.dateBegin = date
			if self.dateEnd < date:
				self.dateEnd = date

			pos = viewLayout.getNodeValue(n)
			self.gravityCenter[0] += pos[0]
			self.gravityCenter[1] += pos[1]
			self.gravityCenter[2] += pos[3]
		self.gravityCenter = [x/self.nbDocuments for x in self.gravityCenter]


		
		

	def visualize(self):
		#here we should sets the size and colors in the document graph
		a = 1
		



		

def main(graph) : 

	descG = graph.getSuperGraph().addSubGraph()
	descG.setName("Descriptors")
	return

	#location = "/work/data/wordlist_0911_1111"
	#wList = ""
	
	#with open(location) as f:
	#   for line in f:
	#	oTypeList = line.split()

	c = clusterAnalysisLgt(graph, [])
	print c.rawMatrix
	print c.cMatrix
	
	print c.typeToIntrications
	print c.globalCoherence
	print c.globalCosine
	
	descG = graph.getSuperGraph().addSubGraph()
	descG.setName("Descriptors")
	print "descG created"
	
	tlp.copyToGraph(descG, c.typeGraph)

	return
	
	#oTypeList = ['a', 'b', 'c', 'd', 'e', 'f']
	#oTypeList = ['_0','_1','_2','_3','_4','_5']
	
	cToDesc = {}
	cToDocs = {}
	cToNode = {}
	
	for sg in graph.getSubGraphs():
		id = sg.getId()
		for ssg in sg.getSubGraphs():
			if ssg.getName() == "Descriptors":
				cToDesc[id] = ssg
			if ssg.getName() == "Documents":
				cToDocs[id] = ssg
				
	clusterGraph = graph
	for sg in graph.getSuperGraph().getSubGraphs():
		if sg.getName() == "clusters":
			clusterGraph = sg
			
	if clusterGraph == graph:
		print "no cluster graph"
		return
			
	gID = clusterGraph.getDoubleProperty("clusterGraphID")
	for n in clusterGraph.getNodes():
		cToNode[int(gID.getNodeValue(n))] = n
	
	if set(cToDesc.keys()) == set(cToDocs.keys()):
		print "desc 2 docs clean"
	if set(cToDesc.keys()) == set(cToNode.keys()):
		print "desc 2 node clean"
	else:
		print set(cToDesc.keys()) ^ set(cToNode.keys())
		print len(set(cToDesc.keys()))
		print len(set(cToNode.keys()))
	
	cosineP = clusterGraph.getDoubleProperty("clusterCosine")
	for i in cToDocs:
		cNode = cToNode[i]
		docGraph = cToDocs[i]
		descGraph = cToDesc[i]
		
		cAnalysis = clusterAnalysisLgt(docGraph, [])
				
		cosineP.setNodeValue(cNode, abs(cAnalysis.globalCosine))
		
		typeName = descGraph.getStringProperty("typeName")
		intrP = descGraph.getDoubleProperty("intrication")
		#print len(set(cAnalysis.typeToIntrications.keys()))
		#print len(set([typeName.getNodeValue(n) for n in descGraph.getNodes()]))
		for n in descGraph.getNodes():
			t = typeName.getNodeValue(n)
			if t in cAnalysis.typeToIntrications.keys():
				intrP.setNodeValue(n, abs(cAnalysis.typeToIntrications[t]))
			else:
				intrP.setNodeValue(n, 0.0)


	return
	c = clusterAnalysisLgt(g, [])
	print c.rawMatrix
	print c.cMatrix
	
	print c.typeToIntrications
	print c.globalCoherence
	print c.globalCosine
	
	#g = graph.addSubGraph()
	#tlp.copyToGraph(g, c.typeGraph)
	print "opposite"
	print c.anotherCMatrix
	print c.anotherTypeToIntrications
	print c.anotherGlobalCoherence
	print c.anotherGlobalCosine
	


