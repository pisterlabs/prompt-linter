from dotenv import load_dotenv
from langchain.graphs.networkx_graph import NetworkxEntityGraph
from langchain.indexes import GraphIndexCreator
from langchain.llms import LlamaCpp
from PyPDF2 import PdfReader 
from typing import List, Dict
import argparse
import ast
import os

#####################################################################################
# Get pages from a PDF document
#####################################################################################
def getPagesFromPDF(pdfFilePath: str, maxPages: int, startPage: int = 0) -> List[Dict]:
  pages = []
  reader = PdfReader(pdfFilePath)
  n = min(len(reader.pages) - startPage, maxPages) 
  for i in range(startPage, startPage + n):
    text = reader.pages[i].extract_text()
    pages.append({ "page": i, "text": text })

  return pages

#####################################################################################
# Create triples from the pages using LLM
#####################################################################################
def createTriplesFromPages(pages: List[Dict], model: LlamaCpp) -> List[Dict]:
  graphObjects = []
  indexCreator = GraphIndexCreator(llm=model)
  for page in pages:
    if page["text"] != "":
      graph = indexCreator.from_text(page["text"])
      triples = graph.get_triples()
      if len(triples) > 0:
        graphObjects.append({ "page": page["page"], "triples": str(triples) })
        print(triples)

  return graphObjects    

#####################################################################################
# Save triples to a file, indexed by page number
#####################################################################################
def saveTriplesToFile(graphObjects: List[Dict], filePath: str) -> None:
  with open(filePath, "a") as f:
    for graph in graphObjects:
      f.write("{0}: {1}".format(graph["page"], graph["triples"]))
      f.write("\n")

#####################################################################################
# Test reading triples from a file
#####################################################################################
def readTriplesFromFile(filePath: str) -> None:
  test = "21: [('Dany', 'Rhaesh Andahli', 'is from'), ('Andals', 'Rhaesh Andahli', 'are from'), ('The Dothraki', 'Rhaesh Andahli', 'are from')]"
  data = test.split(":", 1)
  print(f"Page: {data[0]}")
  triples = ast.literal_eval(data[1].strip())
  print(f"Triples: {triples}")
  print(f"Node 1: {triples[0][0]}, Node 2: {triples[0][1]}, Edge: {triples[0][2]}")

  
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-c", "--create", type=str, required=False, help="Create an in-memory graph, specify it's name")
  parser.add_argument("-p", "--pdf", type=str, required=False, help="Specify a path to the PDF file")
  parser.add_argument("-s", "--startPage", type=int, required=False, help="Specify the starting page number")
  parser.add_argument("-m", "--maxPages", type=int, required=False, help="Specify the max number of pages")
  args = parser.parse_args()

  load_dotenv()
  PATH = os.path.join(os.getenv("LLM_MODEL_PATH"), os.getenv("LLM_MODEL_FILE"))
  MODEL = LlamaCpp(
    model_path = PATH,
    n_ctx = int(os.getenv("MODEL_PARAM_CONTEXT_LEN")),
    n_batch = int(os.getenv("MODEL_PARAM_BATCH_SIZE")),
    use_mlock = os.getenv("MODEL_PARAM_MLOCK"),
    n_threads = int(os.getenv("MODEL_PARAM_THREADS")),
    n_gpu_layers = 0,
    f16_kv = True,
    verbose = False
  )

  if args.create and args.pdf and args.startPage and args.maxPages:
    pages = getPagesFromPDF(pdfFilePath = args.pdf, maxPages = args.maxPages, startPage = args.startPage)
    graphObjects = createTriplesFromPages(pages = pages, model = MODEL)
    saveTriplesToFile(graphObjects = graphObjects, filePath = args.create)

  else:
    print("Incorrect usage: python triples_creator.py [-h] to get help on command options")
