import os,re
import time
import operator
import dendropy
from Bio import SeqIO
from Bio.Blast import NCBIXML
from Bio import AlignIO
from Bio import Align
from Bio.Align import MultipleSeqAlignment
from Bio import Phylo
from Bio import Entrez
Entrez.email = 'jgrant@smith.edu'
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import Phylo
##############################################################



##############################################################
			
class Gene:
	def __init__(self,OG,Pipeline):
		self.PathtoFiles = Pipeline.PathtoFiles
		self.PathtoTemp = Pipeline.PathtoTemp
		self.PathtoOutput = Pipeline.PathtoOutput
		self.PathtoOGFiles = Pipeline.PathtoOGFiles
		self.OG = OG.strip() #OGnumber that identifies the gene
		self.PathtoContamFiles = self.PathtoOutput + '/ContaminationFiles/'
		self.seqs = []
		self.taxa = [] #all taxa from new data and orthomcl that are represented in the gene
		#self.alignment = "" # guidance output
		#self.tree = "" #raxml single gene tree
		#self.OGsequences = [] # all sequences from OrthomCL for this OG
		self.seqCodes = {} #dictionary holding numeric seq codes (keys) and original sequence names (values)
		#self.fastaFile = "" #the fasta file of all sequences in the gene.  
		#x = open(self.PathtoOutput + '/igp_to_remove.txt_' + self.OG,'w')
		#x.close()
		self.getOGsequences() #UNCOMMENT LATER
		self.paralogDict = {}
		self.MClist = {}
		self.sequenceDict = {}
		self.seqtoDelete = []
		self.shortCode = {}
		self.backDict = {}

##############################################################

#############################################################
	def clearmem(self):
		self.seqs = []
		self.taxa = []
		self.seqCodes = {}

	
	def setTaxa(self,tax):
		self.taxa.append(tax)
		
	def getOGsequences(self): 
		#print self.OG
		inSeqs = SeqIO.parse(open(self.PathtoOGFiles + '/' + self.OG.strip(),'r'),'fasta')
		for seq in inSeqs:
			self.seqs.append(seq)
	
	
	def getAllSeqs(self,restart):
		if restart == 'no':
			
			for tax in self.taxa:				
				self.seqs.append(tax.OGxSeqHashtoKeep[(self.OG,tax.code)])
			outfile = open(self.PathtoOutput + '/' + self.OG + '_all.fas','a')
			for seq in self.seqs:
				if type(seq) == list:
					for sequence in seq:
						outfile.write('>' + sequence.id + '\n' + str(sequence.seq) + '\n')
				else:
					outfile.write('>' + seq.id + '\n' + str(seq.seq) + '\n')
		
		elif restart == 'add':
			os.system('cat ' + self.PathtoOGFiles + '/' + self.OG + ' ' +  self.PathtoOutput + '/fasta2keep/' + self.OG + '* > ' + self.PathtoOutput + '/' + self.OG + '_all.fas')
			
		elif restart == 'nt':
			os.system('cat ' +   self.PathtoOutput + '/fasta2keep/' + self.OG + '*' + '_ntfastatokeep.fas > ' + self.PathtoOutput + '/' + self.OG + '_all.fas')  #for yonas pipeline, combine the nucleotide seqs only
		
		else:
			os.system('cat ' + self.PathtoOGFiles + '/' + self.OG + ' ' +  self.PathtoOutput + '/fasta2keep/' + self.OG + '* > ' + self.PathtoOutput + '/' + self.OG + '_all.fas')

		outfile.close()

#####################################################################
#renames seqs so information doesn't get lost in truncation in mafft 
#and later, raxml
#####################################################################
	def getSeqCodes(self):
		os.system('mkdir ' + self.PathtoTemp + '/mafft')
		#print self.seqs
		i = 0
		os.system('cp ' + self.PathtoOutput + '/' + self.OG + '_all.fas ' + self.PathtoTemp + '/mafft/' + self.OG + '_all.fas')
		infile = open(self.PathtoTemp + '/mafft/' + self.OG + '_all.fas','r')
		outfile = open(self.PathtoTemp + '/mafft/'  + self.OG + 'formafft.fas','a')
		codeout = open(self.PathtoOutput + '/' + self.OG + '_seqcodes.txt','a')
		for line in infile:
			if line[0] == '>':
				seqid = line[1:].strip()
				
				newseqid = '>' + str(i) + '_' +seqid.split('_')[1] + '_' +  seqid.split('_')[2]
				i = i + 1
				#print seqid, newseqid
				outfile.write(newseqid + '\n')
				codeout.write(seqid + ':' + newseqid[1:].strip() + '\n')
				self.seqCodes[seqid] = newseqid[1:].strip()
				
			else:
				line = re.sub('J','X',line)
				line = re.sub('O','X',line)
				line = re.sub('Z','X',line)
				outfile.write(line)
		os.system('rm ' + self.PathtoTemp + '/mafft/'  + self.OG + 'temp.fas')	
		infile.close()
		outfile.close()
		codeout.close()

#####################################################################
# *Gene1* In-group paralog removal methods
#####################################################################
	def runmafft(self,iterantion_num):
		#name = self.OG + '_all.fas' #gets the gene's all seqs file for maffting
		os.system('rm ' + self.PathtoOutput + '/igp_to_remove.txt_' + self.OG) #make sure no overlap from previous genes
		os.system('mkdir ' + self.PathtoOutput + '/MAFFT')
		os.system('mafft --auto --treeout --anysymbol ' + self.PathtoTemp + '/mafft/' + self.OG + 'formafft.fas  > ' + self.PathtoOutput + '/MAFFT/' + self.OG + 'mafft')
		treefile = self.PathtoTemp + '/mafft/'  + self.OG + 'formafft.fas.tree'

		
		if self.checkid(iterantion_num):
			print 'ok'
			return
		else:
			self.remove_ingroup_paralogs(treefile,iterantion_num)
			
	#####################################################################


	def checkid(self,iterantion_num):
		if iterantion_num > 0:

			infile1 = open(self.PathtoTemp + '/mafft/' + self.OG + 'formafft.fas'  , 'r')
			infile2 = open(self.PathtoTemp + '/mafft'  + str(iterantion_num) + '/' + self.OG + 'formafft.fas', 'r')
			inFile1 = infile1.readlines()
			inFile2 = infile2.readlines()
			infile1.close()
			infile2.close()
			if str(inFile1) == str(inFile2):
				return True
			else: 
				return False
		else: 
			return False	
	#####################################################################


	def remove_ingroup_paralogs(self,treefile,iterantion_num):
		iterantion_num = iterantion_num + 1
		igp = []
		seq = {}
		#print 'treefile: ' + treefile
		tree = Phylo.read(treefile,'newick')
		for clade in tree.get_terminals():
			self.get_monophyletic(tree,clade,0,clade,'0',clade)
	
	
		self.delete_igp_from_alignment(treefile)

	
		os.system('mkdir ' + self.PathtoTemp + '/mafft'  + str(iterantion_num))
		os.system('cp ' + self.PathtoTemp + '/mafft/'  + self.OG + 'formafft.fas '  + self.PathtoTemp + '/mafft'  + str(iterantion_num))		
		os.system('mv ' + self.PathtoTemp + '/mafft/outformafft.fas ' + self.PathtoTemp + '/mafft/' + self.OG + 'formafft.fas')

		self.runmafft(iterantion_num)
		
		
	#####################################################################


	def get_monophyletic(self,tree,clade,recursion_depth,child,length,sequence):
		#print 'get monophyletic'
		#############################################################
		#checks to see if the clade is monophyletic for taxon and, if it is, 
		#recursively checks the parent clade until it finds a clade that is not monophyletic
		#############################################################
		uiList = [] #list of unique identitfiers in the clade
		seqList = [] #list of sequences in the clade
		parent = clade
		igp = [('','clade')]
		try:
			for seq in clade.get_terminals(): #for each terminal sequence in the clade
			
				ui = str(seq).strip('_').split('_')[-2] + '_' + str(seq).strip('_').split('_')[-1] #ui is the taxon identifier
			
				#print seq, ui
			
				uiList.append(ui) #add the unique identifier to the ui list
				seqList.append(str(seq)) #add the sequence to the seq list
		except:
			print 'no clade: ' + str(clade) + ' ' + str(sequence)
		if len(uiList) >= 1:
			if self.all_same(uiList): #if the taxa are all the same in the clade, check to see if parent clade is also monophyletic
				length = len(uiList)
				child = clade			
				parent = self.get_parent(tree,clade)			
				if parent != 'None':			
					recursion_depth = recursion_depth + 1				
					self.get_monophyletic(tree,parent,recursion_depth,child,length,sequence)
				else:
					print 'no parent: ' + str(sequence)
				
			else:
				if recursion_depth > 1:	
					for clade in child.get_terminals():					
						if str((child,str(clade))) not in igp:
							igp.append((str(clade), child))

					self.removeigp(tree,igp)
	#####################################################################
				
	def removeigp(self,tree,igp):
		#print 'removeIGP'
		igp.pop(0)
	
		distance_dict = {}
		for double in igp:
			for element in (tree.find_elements(name = double[0])):
				distance_dict[element] = tree.distance(element)
	
		sorted_distance_dict = sorted(distance_dict.iteritems(), key=operator.itemgetter(1))
		#print sorted_distance_dict
		for element in sorted_distance_dict:
			if element != sorted_distance_dict[0]:			
				self.igptoDelete(element[0])

	#####################################################################

	def delete_igp_from_alignment(self,rooted_treefile):	

		try:
			infile = open(self.PathtoOutput + '/igp_to_remove.txt_' + self.OG,'r')
		except:
			print 'can not open file "igp_to_remove.txt"'
			return
		
		
		try:
			infile2 = SeqIO.parse(self.PathtoTemp + '/mafft/'  + self.OG + 'formafft.fas'  ,'fasta')
		except:
			print 'can not open alignment file'
			return
		outfile = open('outformafft.fas','a')
		inFile = infile.readlines() #seqs to remove

		inFile = set(inFile)
		inFile = list(inFile)
		#print inFile
		for seq in infile2:
			if not re.search(seq.id, str(inFile)):
				print 'OK!'
				outfile.write('>' + seq.id + '\n' + str(seq.seq) + '\n')
		os.system('mv outformafft.fas ' + self.PathtoTemp + '/mafft/')
		infile.close()
		infile2.close()
		outfile.close()
	#####################################################################
	def removeigp(self,tree,igp):
		igp.pop(0)
	
		distance_dict = {}
		for double in igp:
			for element in (tree.find_elements(name = double[0])):
				distance_dict[element] = tree.distance(element)
	
		sorted_distance_dict = sorted(distance_dict.iteritems(), key=operator.itemgetter(1))
		for element in sorted_distance_dict:
			if element != sorted_distance_dict[0]:			
				self.igptoDelete(element[0])

	#####################################################################

	def igptoDelete(self,sequence):
		#############################################################
		#takes sequence name from other scripts and writes a list
		#of files to be deleted
		#############################################################
		out = open(self.PathtoOutput + '/igp_to_remove.txt_' + self.OG,'a')
		out.write(str(sequence) + '\n')
		out.close

	#####################################################################
	def all_same(self,uiList):
	#############################################################
	#returns True if all items are the same in the list
	#############################################################
		return all(x == uiList[0] for x in uiList)	
		
		
	def get_parent(self,tree, child_clade):
		#############################################################
		#############################################################
		node_path = tree.get_path(child_clade)
		#print node_path
		#assert node_path[-2]
		try:
			return node_path[-2]				
		except:
			#print str(node_path), str(child_clade)
			return None 
			
			

#####################################################################
# *Gene2* Guidance stuff  GuidanceRef: Penn O, Privman E, Ashkenazy H, Landan G, Graur D, Pupko T: GUIDANCE: a web server for assessing alignment confidence scores. Nucleic Acids Res 2010, 38:W23-W28.
#####################################################################
	def fixGuidFile(self):
		inortho = open(self.PathtoOutput + '/Guidance/'+ self.OG + 'forGuidance.fas','r').readlines()
		outortho = open(self.PathtoOutput + '/Guidance/'+ self.OG + 'forGuidance.fas2','a')
		for line in inortho:
			 newline = re.sub('U','X',line)
			 outortho.write(newline)
		outortho.close()	 
		os.system('mv  ' + self.PathtoOutput + '/Guidance/'+ self.OG + 'forGuidance.fas2 ' + self.PathtoOutput + '/Guidance/'+ self.OG + 'forGuidance.fas')
	def run_guidancent(self):
		try:
			ortho = self.PathtoOutput + '/Guidance/'+ self.OG + 'forGuidance.fas '
			os.system('mkdir ' + self.PathtoOutput + '/Guidance')		
			os.system('mv ' + self.PathtoTemp + '/mafft/' + self.OG + 'formafft.fas ' + self.PathtoOutput + '/Guidance/'+ self.OG + 'forGuidance.fas ')
			#os.system('cp ' + self.PathtoOutput + '/' + self.OG + '_all.fas ' + self.PathtoOutput + '/Guidance/'+ self.OG + 'forGuidance.fas ')
			#*Gene5* to skip in-group paralog removal, remove # from line above and add # to line above that

			try:
				os.system('mkdir  ' + self.PathtoOutput + '/Guidance/' + self.OG + '10.5.4_OUTDIR/')
				os.system('cp ' + self.PathtoOutput + '/Guidance/' + ortho + ' ' +  self.PathtoOutput + '/Guidance/' + ortho + '10.5.4_OUTDIR/Seqs.Orig.fas')
				self.fixGuidFile()
				Guidance_Command = 'perl guidance.v1.3.1_Penn2010/www/Guidance/guidance10.5.4.pl --seqFile ' + ortho + ' --bootstraps 10 --msaProgram MAFFT --seqType nuc --outDir '  + self.PathtoOutput + '/Guidance/' + self.OG + '10.5.4_OUTDIR'
				os.system(Guidance_Command)
			except:
				self.fixGuidFile()
				self.run_guidance()
			try:
				probGuid = open(self.PathtoOutput + '/Guidance/' + self.OG + '10.5.4_OUTDIR/ENDS_OK','r') #Guidance finished.
				probGuid.close()
				self.rerunGuidancent()
			except:
				return False
		except:
				return False
	def rerunGuidancent(self):
		try:
			ortho = self.OG + '10.5.4_OUTDIR'
			removedSeqList = []
			seqRemovedFile =  self.PathtoOutput + '/Guidance/'  + ortho + '/Seqs.Orig.fas.FIXED.Removed_Seq'
			filetoReanalyze =  self.PathtoOutput + '/Guidance/'  + ortho + '/Seqs.Orig.fas.FIXED.Without_low_SP_Seq.With_Names'
			try:
				seqRemoved = open(seqRemovedFile,'r')
			except:
				return
			for line in seqRemoved:
				if line[0] == '>':
					removedSeqList.append(line)
			if removedSeqList == []:
				out = open('NoSeqsRemoved','w')
			else:
				Guidance_Command = 'perl guidance.v1.3.1_Penn2010/www/Guidance/guidance10.5.4.pl --seqFile ' + filetoReanalyze + ' --bootstraps 10 --msaProgram MAFFT --seqType nuc --outDir ' +  self.PathtoOutput + '/Guidance/'  + ortho + '10.5.4_reanalyzed'
				os.system(Guidance_Command)
		except:
			print 'problem rerunning'	
	def run_guidance(self):
		ortho = self.PathtoOutput + '/Guidance/'+ self.OG + 'forGuidance.fas '
		os.system('mkdir ' + self.PathtoOutput + '/Guidance')		
		os.system('mv ' + self.PathtoTemp + '/mafft/' + self.OG + 'formafft.fas ' + self.PathtoOutput + '/Guidance/'+ self.OG + 'forGuidance.fas ')

		

		try:
			os.system('mkdir  ' + self.PathtoOutput + '/Guidance/' + self.OG + '10.5.4_OUTDIR/')
			os.system('cp ' + self.PathtoOutput + '/Guidance/' + ortho + ' ' +  self.PathtoOutput + '/Guidance/' + ortho + '10.5.4_OUTDIR/Seqs.Orig.fas')
			self.fixGuidFile()
			Guidance_Command = 'perl guidance.v1.3.1_Penn2010/www/Guidance/guidance10.5.4.pl --seqFile ' + ortho + ' --bootstraps 10 --msaProgram MAFFT --seqType aa --outDir '  + self.PathtoOutput + '/Guidance/' + self.OG + '10.5.4_OUTDIR'
			os.system(Guidance_Command)
		except:
			self.fixGuidFile()
			self.run_guidance()
		try:
			probGuid = open(self.PathtoOutput + '/Guidance/' + self.OG + '10.5.4_OUTDIR/ENDS_OK','r') #Guidance finished.
		
			self.rerunGuidance()
		except:
			print 'problem!'
		
	def rerunGuidance(self):
		ortho = self.OG + '10.5.4_OUTDIR'
		removedSeqList = []
		seqRemovedFile =  self.PathtoOutput + '/Guidance/'  + ortho + '/Seqs.Orig.fas.FIXED.Removed_Seq'
		filetoReanalyze =  self.PathtoOutput + '/Guidance/'  + ortho + '/Seqs.Orig.fas.FIXED.Without_low_SP_Seq.With_Names'
		try:
			seqRemoved = open(seqRemovedFile,'r')
		except:
			return
		for line in seqRemoved:
			if line[0] == '>':
				removedSeqList.append(line)
		if removedSeqList == []:
			out = open('NoSeqsRemoved','w')
		else:
			Guidance_Command = 'perl guidance.v1.3.1_Penn2010/www/Guidance/guidance10.5.4.pl --seqFile ' + filetoReanalyze + ' --bootstraps 10 --msaProgram MAFFT --seqType aa --outDir ' +  self.PathtoOutput + '/Guidance/'  + ortho + '10.5.4_reanalyzed'
			os.system(Guidance_Command)	
#####################################################################
#from guidance output to raxml
#####################################################################	
	def move_guid(self):
		reanalyzedlist = []
		dir = self.PathtoOutput + '/Guidance'	
		os.system('mkdir ' + dir + '/GuidanceOutput')
		try:
			if os.path.isdir(dir + '/' + self.OG + '10.5.4_OUTDIR10.5.4_reanalyzed'):
				os.system('cp ' + dir + '/' + self.OG + '10.5.4_OUTDIR10.5.4_reanalyzed/MSA.MAFFT.Without_low_SP_Col.With_Names ' + dir + '/GuidanceOutput/' + self.OG  + '_GuidanceOut')	
			else:		
				os.system('cp ' + dir + '/' + self.OG + '10.5.4_OUTDIR/MSA.MAFFT.Without_low_SP_Col.With_Names ' + dir + '/GuidanceOutput/' + self.OG  + '_GuidanceOut')			

		except:
			print 'guidance problem'
			
	def mask(self):
		filetype = 'fasta'		
		percent = 50.0

		dir = self.PathtoOutput + '/Guidance/GuidanceOutput'
		os.system('mkdir ' +self.PathtoOutput + '/ForRaxML')
		f = self.OG  + '_GuidanceOut'
		self.maskalignment(f, percent,filetype)
		if self.makephy(f) == False:
			return False
		
				
		
	def maskalignment(self,arg, percent,filetype):
	
		name = arg[0:10]
		maskFileName =  self.PathtoOutput + '/Guidance/GuidanceOutput/' + name + '_masked_' + str(percent) + '.fas'
		outFile = open(maskFileName,'w')
		alignment = AlignIO.read(self.PathtoOutput + '/Guidance/GuidanceOutput/' + arg, filetype)
		trimAlign = MultipleSeqAlignment([])
		numRows = len(alignment)
		x = float(percent) * float(numRows) / 100.0
		numGap = numRows - int(x)
		numCol = alignment.get_alignment_length()
	
		#print "Total number of rows: %i" % numRows
		#print "Number of gapped sequences allowed at a given site: %i" % numGap
		#print "Total number of columns: %i" % numCol
		my_array = {}
		colToKeep=[]
		for i in range(numCol):
			#print i
			lineName = "line_" + str(i)
			my_array[lineName] = alignment[:,i]
			if my_array[lineName].count('-') > numGap:
				print "get rid of column %i" % i
			else:
				colToKeep.append(i)
		
		for record in alignment:
			newseq = ""
			for i in colToKeep:
				newseq= newseq + (record[i])
				
			newRecord = SeqRecord(Seq(newseq), id=record.id)
			trimAlign.append(newRecord)
			outFile.write('>' + record.id + '\n' + newseq + '\n')
		
		#print "Total number of columns remaining: %i" % trimAlign.get_alignment_length()
	
	def checkmask(self,f):
		allgood = []
		inseqfile = open(self.PathtoOutput + '/Guidance/GuidanceOutput/' + f,'r')
		for line in inseqfile:
			if re.search('[A-Za-z]',line):
				allgood.append('good')
			else:
				allgood.append('bad')
		if all(x == 'good' for x in allgood):
			return 'good'
		if all(x == 'bad' for x in allgood):
			return 'bad'
		else:
			return 'mixed'
	
	
	
	def makephy(self,f):
		badseq = []
		#print 'checking ' + f
		if self.checkmask(f) == 'good':
			name = f[0:10]
			maskFileName =  self.PathtoOutput + '/Guidance/GuidanceOutput/' + self.OG  + '_masked_50.0.fas'
	
			inFile = []
			infile = open(maskFileName,'r')
			for line in infile:
				#print line
				try:
					inFile.append(line.strip())
				except:
					print line
			name = self.OG + '.phy'
			outfile = open(self.PathtoOutput + '/ForRaxML/' + name ,'a')
			seqCount = 0
	
			for line in inFile:
				if line[0] == '>':
					seqID = line.strip()[1:] + '         '
					seqCount = seqCount + 1
				else:
					if all(char == '-' for char in line):
						seqCount = seqCount - 1
						badseq.append(seqID)
					charCount = len(line.strip())
			outfile.write(str(seqCount) + ' ' + str(charCount) + '\n')

			for line in inFile:
				if line[0] == '>':
			
					seqID = line.strip()[1:] + '         '
					seqID2 = seqID[0:10] + ' '
					if seqID not in badseq:
						outfile.write('\n' + seqID2)

				else:	
					if seqID not in badseq:
						outfile.write(line.strip())
		
		else:
			print 'no charcters remain in ' + f + ' afer masking.'
			return False
#####################################################################
#call raxml, and rename best tree and best alignment for GTST analyses
#####################################################################

	def callraxml(self):
		os.system('mkdir ' + self.PathtoOutput + '/BestRaxTrees')
		file = self.PathtoOutput + '/ForRaxML/' + self.OG + '.phy'
		raxCL = ('raxml -T 4 -s ' + file + ' -n ' + self.OG + '_outrax.tree -f d -m PROTGAMMALG') #quick best tree
		#print raxCL
		os.system(raxCL)	
		os.system('mv RAxML_bestTree.' + self.OG + '_outrax.tree ' +  self.PathtoOutput + '/BestRaxTrees/' + self.OG + '_outrax.tree')
		os.system('rm RAx*')
	
	def renameTree(self):
		#self.fixseqcode()
		#self.checkNames()
		x = self.makehash()
		self.rename(x)


		
		
	def makehash(self):
		OG = self.OG.split('_')[1] #JUST THE NUMBER
		infile = open(self.PathtoOutput + '/' + self.OG + '_seqcodes.txt','r')
		myHash = {}
		for line in infile:
			#try: 
			#OG=line.split(':')[0].strip()
			#print OG
			num=line.split(':')[1].strip()[0:9]
			name=line.split(':')[0].strip()

			myHash[num] = name
			#except:
			#	continue
		
		return myHash
	
	
	def rename(self,hash):
		os.system('mkdir ' +  self.PathtoOutput + '/RenamedAlignments/')
		os.system('mkdir ' +  self.PathtoOutput + '/BestRaxTreesRenamed/')
		matchob = []
		
		infile = open(self.PathtoOutput + '/Guidance/GuidanceOutput/' + self.OG + '_masked_50.0.fas','r')
		file_out=open(self.PathtoOutput + '/Guidance/GuidanceOutput/' + self.OG + '_renamed','a')

		infiletree = open(self.PathtoOutput + '/BestRaxTrees/' + self.OG + '_outrax.tree','r').readlines()
		
		tree = Phylo.read(infiletree,'newick')
		treefile_out=open(self.PathtoOutput + '/BestRaxTreesRenamed/' + self.OG + '_outrax.tree' + 'renamed','a')

		for line in infile:	


			code = line.split()[0].strip()
			name = self.OG.split('_')[1]
			
			if re.search('_',line):
				num = code[1:]
				num2 = re.sub('fu_X','fu_U',num)[0:9]
				sub = hash[num2]
				line = re.sub(code, '>' + sub, line)
			
			file_out.write(line)
		file_out.close
		os.system('mv ' +  self.PathtoOutput + '/Guidance/GuidanceOutput/' + self.OG + '_renamed ' + self.PathtoOutput + '/RenamedAlignments/' + self.OG + '_renamed')
		for line in infiletree:
			matchob = re.findall('\d+_[A-Za-z]+_[\dA-Za-z]+:',line)
			#print line
			if matchob != []:
		
				print matchob
				for code in matchob:
					print code
					num = re.sub('fu_X','fu_U',code)[0:9]
					try:
						sub = hash[num] + ':'
					except:
						sub = 'problem!!'
				#print sub

					line = re.sub(code, sub, line)
			#print newline

			treefile_out.write(line)
		treefile_out.close


##############################################################
# *Gene3* contamination removal
##############################################################
	def getsistaxlist(self,f,node, taxon, tree, sistax,seenDict,closest_neighbors):	
		out = open(self.PathtoContamFiles + 'sisterLists','a')
		taxname = re.sub(' ','_',str(taxon))
		taxid = taxname.split('_')[0] + '_' + taxname.split('_')[1] + '_' + taxname.split('_')[2]
		par = node.parent_node
	
		sistax[node.taxon] = []
		#get nearest neighbor
		try:
			x = par.leaf_nodes()
		except:
			return
		
		for sister in par.leaf_nodes():
		
			if sister != node:
				sistax[taxon].append(sister)
				for sister in sistax[taxon]:
					#print str(sister.taxon).split()[0] + '_' +  str(sister.taxon).split()[1] + '_' +  str(sister.taxon).split()[2]
					if sister.taxon != None and sister.taxon != taxon and (str(sister.taxon).split()[0] + '_' +  str(sister.taxon).split()[1] + '_' +  str(sister.taxon).split()[2]) != taxid:
						#print str(sister.taxon).split()[0] + '_' +  str(sister.taxon).split()[1] + '_' +  str(sister.taxon).split()[2]

						MC = str(sister.taxon).split()[0]
						mc = str(sister.taxon).split()[1]
						taxid = str(sister.taxon).split()[2]
						tax = MC + '_' + mc + '_' + taxid
						closest_neighbors = closest_neighbors + str(sister.taxon) + ','

					
		print closest_neighbors
		if closest_neighbors != '':
			try:
				x = seenDict[(f + ':' + taxname)] #has it been printed before?
			except: #if not, write it out (if it has, it has already gone up a level)
				seenDict[(f + ':' + taxname)] = 'yes'	
			
			
		#if nearest neighbor clade has few taxa, go until 4 or more	
		if len(sistax[taxon]) < 4:
			sistax = self.getsistaxlist(f,node.parent_node, taxon, tree, sistax,seenDict,closest_neighbors)	
	

		else:
			line = ""
			for sister in sistax[taxon]:
				if sister.taxon != None and sister.taxon != taxon:
					MC = str(sister.taxon).split()[0]
					mc = str(sister.taxon).split()[1]
					taxid = str(sister.taxon).split()[2]
					tax = MC + '_' + mc + '_' + taxid
					line = line + (tax + ',')				
			out.write(f + ':' + taxname + ':')
			out.write(line.strip(',') + '\n')			
		out.close()
		return sistax	

	def get_sister_tax(self):
		try:
			os.system('mkdir ' + self.PathtoOutput + '/ContaminationFiles')
		except:
			pass
		f = self.PathtoOutput + '/BestRaxTreesRenamed/' + self.OG + '_outrax.treerenamed'
		self.fix_trees(f)
		tree = dendropy.Tree.get_from_path(f,'newick')
		taxdict = {}
		sistax = {}
		for node in tree.leaf_nodes(): 
			#print str(node.taxon)
			print str(node.taxon).split()[0] + '_' +  str(node.taxon).split()[1] + '_' +  str(node.taxon).split()[2] 
		
			if str(node.taxon) != 'problem!!':
				print node.taxon
				sistax = self.getsistaxlist(f,node, node.taxon, tree, sistax, {},'')
	
	
	def fix_trees(self,f):
		y = open(f,'r')
		x = y.readlines()
		y.close()
		for line in x:
			newline = re.sub('>','',line)
			newline = re.sub('/','',newline)
			newnewline = re.sub('|','',newline)
	
			outfile = open( f,'w')
			outfile.write(newnewline)
			outfile.close()	
	def parseContam_Taxon(tax,contam,otherlist):
		os.system('mkdir contaminationReports')
		outfile = open('contaminationReports/contaminationReport_' + tax + '_' + contam,'w')
		try:
			MC,mc == contam.split('_')
		except:
			MC = contam
			mc = None
			
			
		infile = open('nearestNeighbors.txt','r')
		for line  in infile:
			try:
				tree,seq,closest = line.split(':')

				if seq.split('_')[0] + '_' + seq.split('_')[1] + '_' + seq.split('_')[2] == tax:
					
					for clade in closest.split(','):
						cladetax = clade.split('_')[0] + '_' +  clade.split('_')[1]  + '_' +  clade.split('_')[2]

						if mc != None:
							if clade.split('_')[0] == MC and clade.split('_')[1] == mc:
								outfile.write(tree + ':' + clade)		
						elif clade.split('_')[0] == MC:
							outfile.write(tree + ':' + clade)	
							

								
			except:
				print 'problem with ' + line		
			
	def contamination(self):
		self.parseContam()
		#self.collate()
		#self.parse()
		self.removecontam()
		

	def parseContam(self):  
		EuMC = ['Am','EE','Ex','Op','Pl','Sr']
		outfile = open(self.PathtoContamFiles + 'checkContam.txt','w')
		filetaxdict = {}
		infile = open(self.PathtoContamFiles + 'sisterLists','r')
		for line in infile:
			file = 'OG5_' + line.split('_')[1].strip()

			testseq = line.split(':')[1]
			filetaxdict[(file,testseq)] = line.split(':')[2].strip().split(',') #a list of the sister seqs

		for double in filetaxdict.keys():
			testtax = double[1].split('_')[0] # just MC
			flag = 0
			if testtax in EuMC: #test taxon is a euk
				for tax in filetaxdict[double]: #is it alone in a sea of prok?
			
					if tax.split('_')[0] in EuMC: #any euk seqs in sister tax?
						flag = 1
				
				if flag == 0:	
					outfile.write(str(double) + str(filetaxdict[double]) + '\n')
				#else:
				#	outfile.write(str(double) + 'no contam.' + str(filetaxdict[double]) + ' \n')
			elif testtax == 'Ba': #test taxon is a bacterium
				for tax in filetaxdict[double]: #is it alone in a sea of euk?
			
					if tax.split('_')[0] == 'Ba': #any other ba seqs in sister tax?
						flag = 1
				
				if flag == 0:	
					outfile.write(str(double) + str(filetaxdict[double]) + '\n')
				#else:
				#	outfile.write(str(double) + 'no contam. ' + str(filetaxdict[double]) + '\n')
			elif testtax == 'Ar': #test taxon is a archaeaon
				for tax in filetaxdict[double]: #is it alone in a sea of euk?
			
					if tax.split('_')[0] == 'Ar': #any other ar  seqs in sister tax?
						flag = 1
				
				if flag == 0:	
					outfile.write(str(double) + str(filetaxdict[double]) + '\n')
				#else:
				#	outfile.write(str(double) + 'no contam. ' + str(filetaxdict[double]) + '\n')
		outfile.close()	
		

	
	
	def removecontam(self):
		removeDict = {}
		infile = open(self.PathtoContamFiles + 'checkContam.txt','r').readlines()
		for line in infile:
			gene = line.split("'")[1]
			taxa = line.split("'")[3]
			try:
				removeDict[gene].append(taxa)
			except:
				removeDict[gene] = []
				removeDict[gene].append(taxa)
			

		f = self.PathtoOutput + '/RenamedAlignments/' + self.OG + '_renamed'
		gene = self.OG
		print gene
		if gene in removeDict.keys():
			infile2 = open(f,'r')
		
			inseq = SeqIO.parse(infile2,'fasta')
			for seq in inseq:
				if seq.id not in removeDict[gene]:
					outfile = open(f + '.contrem','a')
					outfile.write('>' + seq.id + '\n' + str(seq.seq) + '\n')
					outfile.close()
				else:
					outfile = open(f + '_seqsRemoved','a')
					outfile.write(seq.id + '\n')


		os.system('rm ' + self.PathtoContamFiles + '/*')
	

##############################################################
# *Gene4* paralog removal
##############################################################

	def getseqsfromCodeFile(self):
		for mc in ['Op','Pl','EE','Ex','Sr','Ba','Ar','Am']:
			self.MClist[mc]  = []
		infile = open(self.PathtoOutput + '/' + self.OG + '_seqcodes.txt','r')
		for line in infile:
			mc = line.split('_')[0]
			ui = line.split('_')[0] + '_' + line.split('_')[1] + '_' +  line.split('_')[2]
			seqCode = line.split(':')[-1].split('_')[0] #just use number
			fullseq = line.split(':')[0].strip()

			self.MClist[mc].append(seqCode)
			
			self.sequenceDict[seqCode] = (mc,ui,fullseq)
			self.backDict[fullseq] = seqCode			
			self.paralogDict[ui] = [] #instantiate the dictionary as a dict of lists

			if 'U' in ui: #fix the Umey problem
				ui = re.sub('U','X',ui)
				self.paralogDict[ui] = []
		infile.close()		
	
	def removeParalogs(self):
		self.getseqsfromCodeFile()		
		self.uilist = []
		self.tree_in = Phylo.read(self.PathtoOutput + '/BestRaxTrees/' + self.OG + '_outrax.tree','newick')
		try:
			self.alignment = open(self.PathtoOutput + '/RenamedAlignments/' + self.OG + '_renamed.contrem','r')
		except:
			self.alignment = open(self.PathtoOutput + '/RenamedAlignments/' + self.OG + '_renamed','r')
		for seq in self.tree_in.get_terminals():
			print self.OG			
			try:
				ui = self.sequenceDict[str(seq).split('_')[0]][1] #ui is MC_mc_code
				self.paralogDict[ui].append(str(seq)) # so len is # of paralogs per taxon
				if ui not in self.uilist:
					self.uilist.append(ui)
			except:
				print 'problem with ' + self.OG
				
	
		for ui in self.uilist:	
			print 'self.paralogDict[ui] ' + str(self.paralogDict[ui])
			if len(self.paralogDict[ui]) > 1:
				print ui
				self.pickParalog(ui)
		print 'seq to delete ' + str(self.seqtoDelete)
		self.deleteSeqsFromAlignment()

		self.alignment.close()
	
	def deleteSeqsFromAlignment(self):  #self.shortCode[seqCode] = (MC,ui,fullseq)
		nametoDelete = []
		os.system('mkdir ' + self.PathtoOutput + '/AlignmentsforConcatenation')
	
		infile2 = SeqIO.parse(self.alignment ,'fasta')
		
		outfile = open(self.PathtoOutput + '/AlignmentsforConcatenation/' + self.OG + '_paralog_purged.fas','a')
	
		print self.seqtoDelete
		for code in self.seqtoDelete:
			nametoDelete.append(self.sequenceDict[code][2])
		outfile2 = open(self.PathtoOutput + '/' + self.OG + '_sequencesKept.txt','a')
		for seq in infile2:
			print seq.id
			if seq.id  not in nametoDelete: #if not in too delete, keep.
				newname = seq.id.split('_')[0] + '_' + seq.id.split('_')[1] + '_' + seq.id.split('_')[2]
				outfile.write('>' + newname + '\n' + str(seq.seq) + '\n')
				
				outfile2.write(newname + ':' + seq.id + '\n')
		outfile2.close()
	
		outfile.close
	def pickParalog(self, ui): #ui = unique identifier of taxa that have paralogs
		for sequence in self.paralogDict[ui]:
			clade = self.get_clade(sequence)
			try:
				seq_clade= self.get_parent(self.tree_in,clade)
				if seq_clade == None:
					seq_clade = clade
			except:
				seq_clade = clade
			
			if seq_clade != None:
				self.is_monophyletic(seq_clade,0,"None",0,sequence,[],[])
			else:
				print sequence
				self.seqtoDelete.append(sequence[0:9])
				self.seqtoDelete.append(sequence[0:10 ])
			self.paralogDict[ui] = []
		self.sort_clade_sizes()
	
	def sort_clade_sizes(self):
	#############################################################	
	#for each taxon, makes a list of seq/num pairs sorted py num
	#and passes it to get_seq_to_keep which will take the best seq
	#############################################################	
	
		inFile = open(self.PathtoTemp + '/clade_size.txt','r')
		infile = inFile.readlines()
		inFile.close()
		#infile.append('XX_Xxxx_XXXXX:0\n') #end of file
		seqDict = {}
		seqnumDict = {}
		
		for line in infile:
			
			seq = line.split(':')[0]
			num = line.split(':')[1]
			seqDict[seq] = int(num)


		sorted_seqDict = sorted(seqDict.iteritems(), key=operator.itemgetter(1), reverse = True)	
		xx = open(self.PathtoTemp + '/clade_size.txt','w') #clear file
		xx.close()	
		self.get_seq_to_keep(sorted_seqDict) #call next method, not written yet

	def get_seq_to_keep(self,sorted_seqDict):
	#############################################################
	#checks to see if there is one sequence in a larger monophyletic clade
	#if not, makes a list of seqs and takes the one with the smallest 
	#branch length
	#############################################################
		check_dist_list=[]

		if len(sorted_seqDict) > 1:			# sorted_seqDict, list of (seqs,sizeofclade) sorted by size of clade
			if float(sorted_seqDict[0][1]) > float(sorted_seqDict[1][1]): #if the first is bigger than the next then it is the biggest
				#print 'KEEP '+ str(sorted_seqDict[0][1])		
				sorted_seqDict.pop(0) #keep the first, get rid of the rest
				for double in sorted_seqDict:  
					self.seqtoDelete.append(double[0])  # put all the rest in seqtoDelete
			else: #More than seq in clade with equal numbers
				for double in sorted_seqDict:
					#print double
					if float(double[1]) == float(sorted_seqDict[0][1]):
						check_dist_list.append(double) #get list of seqs to check for distance
					else:
						self.seqtoDelete.append(double[0])
				#print 'check distance list = ' + str(check_dist_list)		
				self.check_distance(check_dist_list)


	def check_distance(self,check_dist_list):
		distance_dict = {}
		for double in check_dist_list:
			for element in (self.tree_in.find_elements(name = double[0])):
				distance_dict[element] = self.tree_in.distance(element)
	
		sorted_distance_dict = sorted(distance_dict.iteritems(), key=operator.itemgetter(1))
		
		#print 'XXDistanceXX ' + str(sorted_distance_dict)
		sorted_distance_dict.pop(0) #keep the shortest, remove others
		for element in sorted_distance_dict:
				self.seqtoDelete.append(str(element[0]).split('_')[0]) #want just the number, or SeqCode


	def is_monophyletic(self,clade,recursion_depth,child,length,sequence,mcCladeList,seqCladeList):
		#############################################################
		#checks to see if the clade is monophyletic for major clade and, if it is, 
		#recursively checks the parent clade until it finds a clade that is not monophyletic
		#############################################################
		out = open(self.PathtoTemp + '/clade_size.txt','a')
		try:
			for seq in clade.get_terminals(): #for each terminal sequence in the clade
		
				mc = self.sequenceDict[str(seq)][0] #ui is the major clade identifier i.e Op, Pl, etc
			
				if str(seq) not in seqCladeList:
					seqCladeList.append(str(seq)) #add the sequence to the seq list
					mcCladeList.append(mc) #add the unique identifier to the ui list
		
			parent = clade


			if len(mcCladeList) >= 1:
				if self.all_same(mcCladeList): #if the taxa are all the same in the clade, check to see if parent clade is also monophyletic
					length = len(mcCladeList)
					child = clade			
					parent = self.get_parent(self.tree_in,clade)			
					if parent != 'None':			
						recursion_depth = recursion_depth + 1				
						self.is_monophyletic(parent,recursion_depth,child,length,sequence,mcCladeList,seqCladeList)
					else:
						print 'no parent: ' + str(sequence)#'''and...what'''
				
				else:
					
					if recursion_depth < 1:
						out.write(sequence.strip() + ':' + str(1) + '\n')

					else:	
						out.write(sequence.strip() + ':' + str(length) + '\n')			


					#print sequence, str(length), mcCladeList,seqCladeList
		except:
			out.write(sequence.strip() + ':' + str(0) + '\n')
		out.close()
			
			
			
	def all_same(self,uiList):
	#############################################################
	#returns True if all items are the same in the list
	#############################################################
		return all(x == uiList[0] for x in uiList)	
	
	
	def get_clade(self,sequence):
 		cladelist = [] #leaf.names.
		for clade in self.tree_in.get_terminals():		
			if str(clade.name).strip() == str(sequence).strip():
				cladelist.append(clade)
		try:
			#print 'cladelist ' + str(cladelist)
			return cladelist[0]
		except:
			print str(sequence).strip() + ' not found'









