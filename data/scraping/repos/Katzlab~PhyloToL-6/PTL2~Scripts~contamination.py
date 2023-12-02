import os, sys, re
from Bio import SeqIO
import ete3
import guidance
import trees

	
def get_newick(fname):
	
	newick = ''
	for line in open(fname):
		line = line.split(' ')[-1]
		if(line.startswith('(') or line.startswith('tree1=')):
			newick = line.split('tree1=')[-1].replace("'", '').replace('\\', '')

	return newick


#This function reroots the tree on the largest Ba/Za clade. If there is no prokaryote clade,
#it roots on the largest Op clade, then Pl, then Am, then Ex, then Sr.
def reroot(tree):

	#This nested function returns the largest clade of a given taxonomic group
	def get_best_clade(taxon):

		best_size = 0; best_clade = []; seen_leaves = []
		#Traverse all nodes
		for node in tree.traverse('levelorder'):
			#If the node is big enough and not subsumed by a node we've already accepted
			if len(node) >= 3 and len(list(set(seen_leaves) & set([leaf.name for leaf in node]))) == 0:
				leaves = [leaf.name for leaf in node]
				
				#Create a record of leaves that belong to the taxonomic group
				target_leaves = set()
				for leaf in leaves[::-1]:
					if leaf[:2] in taxon:
						target_leaves.add(leaf[:10])
						leaves.remove(leaf)

				#If this clade is better than any clade we've seen before, grab it
				if len(target_leaves) > best_size and len(leaves) <= 2:
					best_clade = node
					best_size = len(target_leaves)
					seen_leaves.extend([leaf.name for leaf in node])

		return best_clade

	#Get the biggest clade for each taxonomic group (stops once it finds one)
	for taxon in [('Ba', 'Za'), ('Op'), ('Pl'), ('Am'), ('Ex'), ('Sr')]:
		clade = get_best_clade(taxon)
		if len([leaf for leaf in clade if leaf.name[:2] in taxon]) > 3:
			tree.set_outgroup( clade)

			break

	return tree
	
	
def get_subtrees(args, file):

	newick = get_newick(file)	

	tree = ete3.Tree(newick)

	try:
		tree = reroot(tree)
	except:
		print('\nUnable to re-root the tree ' + file + ' (maybe it had only 1 major clade, or an inconvenient polytomy). Skipping this step and continuing to try to grab robust clades from the tree.\n')					

	#Getting a clean list of all target taxa
	if os.path.isfile(args.target):
		try:
			target_codes = [l.strip() for l in open(args.target).readlines() if l.strip() != '']
		except AttributeError:
			print('\n\nError: invalid "target" argument. This must be a comma-separated list of any number of digits/characters to describe focal taxa (e.g. Sr_ci_S OR Am_t), or a file with the extension .txt containing a list of complete or partial taxon codes. All sequences containing the complete/partial code will be identified as belonging to target taxa.\n\n')
	else:
		#make sure that this is how nargs works
		target_codes = [code.strip() for code in args.target if code.strip() != '']

	#Getting a clean list of all "at least" taxa
	if os.path.isfile(args.required_taxa):
		try:
			at_least_codes = [l.strip() for l in open(args.required_taxa).readlines() if l.strip() != '']
		except AttributeError:
			print('\n\nError: invalid "required_taxa" argument. This must be a comma-separated list of any number of digits/characters (e.g. Sr_ci_S OR Am_t), or a file with the extension .txt containing a list of complete or partial taxon codes, to describe taxa that MUST be present in a clade for it to be selected (e.g. you may want at least one whole genome).\n\n')
	else:
		#make sure that this is how nargs works
		at_least_codes = [code.strip() for code in args.required_taxa if code.strip() != '']

	target_codes = list(dict.fromkeys(target_codes + at_least_codes))

	#Creating a record of selected subtrees, and all of the leaves in those subtrees
	selected_nodes = []; seen_leaves = []

	#Iterating through all nodes in tree, starting at "root" then working towards leaves
	for node in tree.traverse('levelorder'):
		#If a node is large enough and is not contained in an already selected clade
		if len(node) >= args.min_target_presence and len(list(set(seen_leaves) & set([leaf.name for leaf in node]))) == 0:
			leaves = [leaf.name for leaf in node]

			#Accounting for cases where e.g. one child is a contaminant, and the other child is a good clade with 1 fewer than the max number of contaminants
			children_keep = 0
			for child in node.children:
				for code in target_codes:
					for leaf in child:
						if leaf.name.startswith(code):
							children_keep += 1
							break

			if children_keep == len(node.children):
				#Creating a record of all leaves belonging to the target/"at least" group of taxa, and any other leaves are contaminants
				target_leaves = set(); at_least_leaves = set()
				for code in target_codes:
					for leaf in leaves[::-1]:
						if leaf.startswith(code):
							target_leaves.add(leaf[:10])

							if code in at_least_codes:
								at_least_leaves.add(leaf[:10])

							leaves.remove(leaf)

				#Grab a clade as a subtree if 1) it has enough target taxa; 2) it has enough "at least" taxa; 3) it does not have too many contaminants
				if len(target_leaves) >= args.min_target_presence and len(at_least_leaves) >= args.n_at_least and ((args.contaminants < 1 and len(leaves) < args.contaminants * len(target_leaves)) or len(leaves) < args.contaminants):
					selected_nodes.append(node)
					seen_leaves.extend([leaf.name for leaf in node])

	#Write the subtrees to output .tre files
	seqs2keep = [leaf.name for node in selected_nodes for leaf in node]

	return seqs2keep


def get_sisters(args, file, contam_per_tax):
				
	seqs2remove = []
	
	#Read the tree using ete3 and reroot it using the above function
	newick = get_newick(file)
	tree = ete3.Tree(newick)

	try:
		tree = reroot(tree)
	except:
		print('\nUnable to re-root the tree ' + file + ' (maybe it had only 1 major clade, or an inconvenient polytomy). Skipping this step and continuing to try to grab robust clades from the tree.\n')

	#For each sequence
	for leaf in tree:

		#This loop will keep moving towards the root of the tree until it finds a node that
		#has leaves from a cell other than the one for which we are looking for sisters
		parent_node = leaf; sister_taxa = {leaf.name[:10]}
		while len(sister_taxa) == 1:
			parent_node = parent_node.up
			for l2 in parent_node:
				sister_taxa.add(l2.name[:10])

		#Create a record of the sister sequences
		sisters = list(dict.fromkeys([sister for sister in parent_node if sister.name[:10] != leaf.name[:10]]))

		bad_sisters = list(dict.fromkeys([contam for tax in contam_per_tax for contam in contam_per_tax[tax] if leaf.name.startswith[tax]]))

		sisters_removable = []
		for contam in bad_sisters:
			for sister in sisters:
				if sister.startswith(contam) and sister not in sisters_removable:
					sisters_removable.append(sister)
		
		if len(sisters_removable) == len(sisters):
			seqs2remove.append(leaf.name)

	return [leaf.name for leaf in tree if leaf.name not in seqs2remove]



def write_new_preguidance(params, seqs2keep, seqs_per_og):

	prefix = '.'.join(tree_file.split('.')[:-1])
	seq_file = [file for file in seqs_per_og if file.startswith(prefix)]
	if len(seq_file) == 0:
		seq_file = [file for file in seqs_per_og if file.startswith(prefix.split('.')[0])]

		if len(seq_file) == 0:
			print('\nNo sequence file found for tree file ' + tree_file + '. Skipping this gene family.\n')
		elif len(seq_file) > 1:
			print('\nMore than one sequence file found matching the tree file ' + tree_file + '. Please make your file names more unique: there should be one sequence file for every tree file, with a matching unique prefix (everything before the first "."). Skipping this gene family.\n')
		
	if len(seq_file) == 1:
		with open(params.output + '/Pre-Guidance/' + seq_file, 'w') as o:
			for rec in seqs_per_og[seq_file]:
				if rec in seqs2keep:
					o.write('>' + rec + '\n' + seqs_per_og[seq_file][rec] + '\n\n')
na
		seqs_removed_from_og = [seq for seq in seqs_per_og[seq_file] if seq not in seqs2keep]


def run(params):

	seqs_removed = []
	completed_ogs = []

	for loop in range(params.nloops):
		if params.start == 'raw':
			seqs_per_og = { file : { rec.id : str(rec.seq) for rec in SeqIO.parse(file, 'fasta') } for file in os.listdir(params.output + '/Output/Pre-Guidance') if file.split('.')[-1] in ('fasta', 'fas', 'faa') }
		elif params.start in ('unaligned', 'aligned', 'trees'):
			seqs_per_og = { file : { rec.id : str(rec.seq).replace('-', '') for rec in SeqIO.parse(file, 'fasta') } for file in os.listdir(params.data) if file.split('.')[-1] in ('fasta', 'fas', 'faa') }

		if loop > 0 or params.start == 'raw':
			os.system('mv ' + params.output + '/Pre-Guidance ' + params.output + '/Pre-Guidance_' + str(loop))
		
		os.mkdir(params.output + '/Pre-Guidance')

		if params.contamination_loop == 'clade':
			for tree_file in params.output + '/Trees':
				if tree_file.split('.')[-1] in ('tre', 'tree', 'treefile', 'nex') and tree_file not in completed_ogs:
					seqs2keep = get_subtrees(params, params.output + '/Trees/' + tree_file)

					seqs_removed_from_og = write_new_preguidance(params, seqs2keep, seqs_per_og)

					if len(seqs_removed_from_og) == 0:
						completed_ogs.append(tree_file)
					else:
						seqs_removed += [seq for seq in seqs_per_og[seq_file] if seq not in seqs2keep]

		elif params.contamination_loop == 'seq':
			contam_per_tax = { line.strip().split('\t')[0] : line.strip().split('\t')[1:] for line in params.sister_rules }

			if params.contamination_loop == 'clade':
				for tree_file in params.output + '/Trees':
					if tree_file.split('.')[-1] in ('tre', 'tree', 'treefile', 'nex') and tree_file not in completed_ogs:
						seqs2keep = get_sisters(params, params.output + '/Trees/' + tree_file, contam_per_tax)

						seqs_removed_from_og = write_new_preguidance(params, seqs2keep, seqs_per_og)

						if len(seqs_removed_from_og) == 0:
							completed_ogs.append(tree_file)
						else:
							seqs_removed += [seq for seq in seqs_per_og[seq_file] if seq not in seqs2keep]

		os.system('mv ' + params.output + '/Trees ' + params.output + '/Trees_' + str(loop))
		os.mkdir(params.output + '/Trees')

		os.system('mv ' + params.output + '/Guidance ' + params.output + '/Guidance_' + str(loop))
		os.mkdir(params.output + '/Guidance')

		params.start = 'unaligned'
		params.end = 'trees'

		guidance.run(params)
		trees.run(params)

	with open('SequencesRemoved_ContaminationLoop.txt', 'w') as o:
		for seq in seqs_removed:
			o.write(seq + '\n')


