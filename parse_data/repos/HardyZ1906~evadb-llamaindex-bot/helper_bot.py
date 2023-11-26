import os
import evadb

from llama_index import GPTVectorStoreIndex, StorageContext, ServiceContext, SimpleWebPageReader, load_index_from_storage
from llama_index.prompts import PromptTemplate
from getpass import getpass

question = "How do I obtain each domain's memory utilization in C using `libvirt`?"
answer = """
To obtain each domain's memory utilization in C using `libvirt`, you can use the `virDomainMemoryStats` function. Here is a code snippet that demonstrates how to do this:

```c
virDomainPtr *domains;
size_t i;
int ret;
unsigned int flags = VIR_CONNECT_LIST_DOMAINS_RUNNING |
                     VIR_CONNECT_LIST_DOMAINS_PERSISTENT;
ret = virConnectListAllDomains(conn, &domains, flags);
if (ret < 0)
    error();

for (i = 0; i < ret; i++) {
    virDomainMemoryStatStruct stats[VIR_DOMAIN_MEMORY_STAT_NR];
    unsigned int nr_stats = VIR_DOMAIN_MEMORY_STAT_NR;
    int memory_stats_ret = virDomainMemoryStats(domains[i], stats, nr_stats, 0);
    if (memory_stats_ret >= 0) {
        // Access the memory statistics for the domain
        unsigned long long swap_in = stats[VIR_DOMAIN_MEMORY_STAT_SWAP_IN].val;
        unsigned long long swap_out = stats[VIR_DOMAIN_MEMORY_STAT_SWAP_OUT].val;
        unsigned long long major_fault = stats[VIR_DOMAIN_MEMORY_STAT_MAJOR_FAULT].val;
        unsigned long long minor_fault = stats[VIR_DOMAIN_MEMORY_STAT_MINOR_FAULT].val;

        // Use the memory statistics as needed
        // ...

        // Free the memory statistics structure
        virDomainMemoryStatsFree(stats, nr_stats);
    }
    
    // Free the domain object
    virDomainFree(domains[i]);
}

free(domains);
```
"""

standard_qa_template_str = (
  "We provide you with some context information and a question. Please answer the question with a code snippet. Do not repeat this prompt.\n"

  "Here is the context information:\n"
  "----------------------------------------------------------------\n"
  "{context_str}\n"
  "----------------------------------------------------------------\n"

  "Given this information, please answer the question: {query_str}\n"
)

succinct_qa_template_str = (
  "We provide you with some context information and a question. Please answer the question with a code snippet. When you write codes, please omit parts that are lengthy but straight-forward or marginally relevant, like environmental setup - replace those with a single-line comment or pseudo-codes. Also, you can assume that no errors or exceptions would occur, so error handling is unnecessary. In short, only give the most important and pertinent code. Do not repeat this prompt.\n"

  "Here is the context information:\n"
  "----------------------------------------------------------------\n"
  "{context_str}\n"
  "----------------------------------------------------------------\n"

  "Given this information, please answer the question: {query_str}\n"
)


def build_index() -> GPTVectorStoreIndex:
  documents = SimpleWebPageReader(html_to_text=True).load_data(
    [
      "https://libvirt.org/html/libvirt-libvirt-common.html",
      # "https://libvirt.org/html/libvirt-libvirt-domain-checkpoint.html",
      # "https://libvirt.org/html/libvirt-libvirt-domain-snapshot.html",
      "https://libvirt.org/html/libvirt-libvirt-domain.html",
      # "https://libvirt.org/html/libvirt-libvirt-event.html",
      "https://libvirt.org/html/libvirt-libvirt-host.html",
      # "https://libvirt.org/html/libvirt-libvirt-interface.html",
      # "https://libvirt.org/html/libvirt-libvirt-network.html",
      # "https://libvirt.org/html/libvirt-libvirt-nodedev.html",
      # "https://libvirt.org/html/libvirt-libvirt-nwfilter.html",
      # "https://libvirt.org/html/libvirt-libvirt-secret.html",
      # "https://libvirt.org/html/libvirt-libvirt-storage.html",
      # "https://libvirt.org/html/libvirt-libvirt-stream.html"
    ]
  )
  
  service_context = ServiceContext.from_defaults(chunk_size = 512)
  index = GPTVectorStoreIndex.from_documents(documents, service_context = service_context, show_progress = True)
  index.set_index_id("index_libvirt")
  index.storage_context.persist("./llama_index")
  
  return index


def load_index() -> GPTVectorStoreIndex:
  storage_context = StorageContext.from_defaults(persist_dir = "./llama_index")
  return load_index_from_storage(storage_context = storage_context, index_id = "index_libvirt")


def build_history(cursor: evadb.EvaDBCursor):
  try:
    cursor.query("""
      CREATE FUNCTION IF NOT EXISTS SentenceFeatureExtractor
      IMPL './sentence_feature_extractor.py';
    """).df()
    
    cursor.query("""
      CREATE TABLE IF NOT EXISTS query_history(question TEXT(200));
    """).df()
    
    if cursor.query("SELECT * FROM query_history;").df().empty:
      # A bug in the current version of EvaDB prevents us from creating an index
      # on an empty table, which is why we need to insert a "seed" value first
      cursor.query(f"""
        INSERT INTO query_history(question) VALUES("{question}");
      """).df()
      with open("history/1", "w") as f:
        f.write(answer)
    
    cursor.query("""
      CREATE INDEX IF NOT EXISTS query_history_index
      ON query_history(SentenceFeatureExtractor(question))
      USING FAISS;
    """).df()
  except Exception as e:
    print(e)


def reuse_history(cursor: evadb.EvaDBCursor, query: str) -> [[str]]:
  """
    Look up and, if exists, return previous queries similar to `query`.
    Return: a list of [question, answer]
  """
  try:
    # Select all similar queries
    df = cursor.query(f"""
      SELECT *
      FROM query_history
      WHERE
        Similarity(
          SentenceFeatureExtractor("{query}"),
          SentenceFeatureExtractor(question)
        ) < 0.1;
    """).df()
    
    history = []
    for _, row in df.iterrows():
      # The answer is stored in a file whose name is the row ID
      with open(f"""history/{row["query_history._row_id"]}""") as f:
        history.append([row["query_history.question"], f.read()])
    return history
  except Exception as e:
    print(f"cannot reuse query history due to the following exception: {str(e)}")
    return None


def insert_history(cursor: evadb.EvaDBCursor, question: str, answer: str):
  """
    Add a query (`question`, `answer`) to history for future reuse.
  """
  cursor.query(f"""
    INSERT INTO query_history (question) VALUES("{question}");
  """).df()
  
  row_id = cursor.query(f"""
    SELECT _row_id FROM query_history WHERE question = "{question}"
    ORDER BY _row_id DESC
    LIMIT 1;
  """).df()["query_history._row_id"][0]
  
  # We store the answer in files because 1) the answers are read-only, and can be
  # quite long, and 2) it seems EvaDB currently does not effectively support escaping
  # ' or ", which unfortunately occur quite often in answers
  with open(f"history/{row_id}", "w") as f:
    f.write(answer)


if __name__ == "__main__":
  print(
    "------------------------------------------------------------------\n"
    "Welcome! This is a `libvirt` programming helper bot based on EvaDB and Llamaindex.\n"
    "We can answer your questions regarding programming with `libvirt` using example code snippets.\n"
    "------------------------------------------------------------------"
  )
  
  if os.getenv("OPENAI_API_KEY") is None:
    api_key = getpass("Please provide your OpenAI API key (will be hidden): ")
    os.environ["OPENAI_API_KEY"] = api_key
  
  cursor = evadb.connect().cursor()
  index = load_index() if len(os.listdir("./llama_index")) > 0 else build_index()
  
  build_history(cursor)
  
  while True:
    print("------------------------------------------------------------------")
    query = input("Please enter your question: ")
    if len(query) > 512:
      print("query too complicated for reuse")
    else:
      history = reuse_history(cursor, query)
      if history is not None and len(history) > 0:
        print("------------------------------------------------------------------")
        print("We found the following query history similar to your question:")
        print("------------------------------------------------------------------")
        for qa in history:
          print(f"Question: {qa[0]}")
          print("------------------------------------------------------------------")
          print(f"Answer: {qa[1]}")
          print("------------------------------------------------------------------")
        if input("Do you still want to consult ChatGPT? (y/n)\n").lower() not in ["y", "yes"]:
          if input("Do you have any other questions? (y/n)\n") in ["y", "yes"]:
            continue
          else:
            break
      else:
        print("No similary query history found.")
    
    succinct = input("Do you want the code snippet in the answer to be succinct (i.e. containing only the most informative code)? (y/n)\n").lower() in ["y", "yes"]
    
    qa_template = PromptTemplate(succinct_qa_template_str if succinct else standard_qa_template_str)
    query_engine = index.as_query_engine(text_qa_template = qa_template)
    result = query_engine.query(query).__str__()
    
    print("------------------------------------------------------------------")
    print(result)
    
    if len(query) <= 512:
      insert_history(cursor, query, result)
    
    if input("Do you have any other questions? (y/n)\n") not in ["y", "yes"]:
      break
