#!/usr/bin/python3
import os, sys, re
import contamination
import utils
import preguidance
import guidance
import trees
import concatenate


if __name__ == '__main__':

	params = utils.get_params()
	
	if not (params.concatenate and params.start == 'trees'):
		print('\nCleaning up existing files and organizing output folder\n')
		utils.clean_up(params)

	if params.start == 'raw':
		print('\nRunning preguidance\n')
		preguidance.run(params)
	
	if params.start in ('unaligned', 'raw') and params.end in ('aligned', 'trees'):
		print('\nRunning guidance\n')
		guidance.run(params)

	if params.start != 'trees' and params.end == 'trees':
		print('\nBuilding trees\n')
		trees.run(params)

	if params.contamination_loop != None:
		print('\nRunning contamination loop\n')
		contamination.run(params)

	if params.concatenate:
		print('\nChoosing orthologs and concatenating alignments...\n')
		concatenate.run(params)
	
