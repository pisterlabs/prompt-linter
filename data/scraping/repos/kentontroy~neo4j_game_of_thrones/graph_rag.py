from dotenv import load_dotenv
from langchain.chains import GraphCypherQAChain
from langchain.graphs import Neo4jGraph
from langchain.llms import LlamaCpp
import os

if __name__ == "__main__":
  load_dotenv()
  PATH = os.path.join(os.getenv("LLM_MODEL_PATH"), os.getenv("LLM_MODEL_FILE"))
  llm = LlamaCpp(
    model_path = PATH,
    n_ctx = int(os.getenv("MODEL_PARAM_CONTEXT_LEN")),
    n_batch = int(os.getenv("MODEL_PARAM_BATCH_SIZE")),
    use_mlock = os.getenv("MODEL_PARAM_MLOCK"),
    n_threads = int(os.getenv("MODEL_PARAM_THREADS")),
    n_gpu_layers = 0,
    temperature = 0,
    f16_kv = True,
    verbose = False
  )

  graph = Neo4jGraph(
    url="bolt://localhost:7687", username="neo4j", password="cloudera"
  )
  print(graph.schema)

  chain = GraphCypherQAChain.from_llm(llm, graph=graph, verbose=True, return_intermediate_steps=True)
  chain.run("How many pages are in the book Game Of Thrones?")
