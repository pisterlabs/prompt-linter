
from sys import argv as argv
import openai as oai

def logTxn( *args , **kwargs ):
	print('Logging txn')
	print( args )



logTxn(argv)