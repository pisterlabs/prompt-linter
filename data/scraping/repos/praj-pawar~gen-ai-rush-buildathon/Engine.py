from llama_index import VectorStoreIndex
from llama_index.schema import TextNode
import openai
openai.api_key = "your api key"


def load_engine(timechunks):
    nodes = [TextNode(text=transcript['text'], id_=transcript['time'])
             for transcript in timechunks]
    index = VectorStoreIndex(nodes)
    query_engine = index.as_query_engine()
    return query_engine


def get_summary(engine):
    query = 'divide the video into main key components and give a title for each component, give the titles in order of their occurance in video and use the word part before each component title'
    response = engine.query(query)
    l = response.response.split('\n')
    b = []
    for i in range(len(l)):
        if l[i] == '':
            pass
        else:
            b.append(l[i][7:])
    sum_t = []
    for s in b:
        r = engine.query(s)
        sum_t.append(r.source_nodes[0])
    seq_dict = {}
    for i in range(len(b)):
        seq_dict[sum_t[i].node.id_] = b[i]
    k = []
    for i in seq_dict:
        k.append(float(i))
    k.sort()
    seq_order = []
    for i in range(len(k)):
        seq_order.append(
            str(i+1)+'. '+seq_dict[str(k[i])]+" - %02d:%02d" % (int(k[i]/60), int(k[i] % 60)))
    return seq_order


def get_query(engine, query):
    print("query"+"hi",engine)
    response = engine.query(query)
    timestamp = response.source_nodes[0].node.id_
    timestamp = int(float(timestamp))
    timestamp = "%02d:%02d" % ((timestamp/60), (timestamp % 60))
    return response, timestamp
