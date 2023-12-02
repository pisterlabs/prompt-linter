from keyword_explorer.OpenAI.OpenAIComms import OpenAIComms, ChatUnit
from keyword_explorer.utils.NetworkxGraphing import NetworkxGraphing
import openai.embeddings_utils as oaiu
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from collections import deque

from typing import List, Dict

class TopicNode:
    name:str
    average_embedding:np.array
    known_good_topics_list:List
    known_good_embeddings_list:List
    provisional_topics_list:List
    provisional_embeddings_list:List
    reject_threshold:float
    inbound_node_list:List
    outbound_node_list:List

    oac:OpenAIComms

    def __init__(self, name:str, oac:OpenAIComms):
        print("TopicNode.__init__()")
        self.reject_threshold = 0.1
        self.name = name.strip()
        self.oac = oac
        embd_list = self.oac.get_embedding_list([self.name])
        d = embd_list[0]
        self.average_embedding = np.array(d['embedding'])
        # print("\t{}, {}".format(self.name, self.average_embedding))
        self.known_good_topics_list = [self.name]
        self.known_good_embeddings_list = [self.average_embedding]
        self.provisional_embeddings_list = []
        self.provisional_topics_list = []
        self.inbound_node_list = []
        self.outbound_node_list = []

    def tolist(self, a:np.array) -> List:
        return a.tolist()

    def remove_outliers(self, data:np.ndarray, low_pct:float = 0.33, high_pct = 0.66):
        import numpy as np

        q1 = np.percentile(data, low_pct)
        q3 = np.percentile(data, high_pct)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        return [x for x in data if lower_bound <= x <= upper_bound]

    def add_known_good_list(self, topics:List):
        print("TopicNode.add_known_good_list()")
        embd_list = self.oac.get_embedding_list(topics)
        x:Dict
        dist_array = np.array([x['embedding'] for x in embd_list])
        average_embedding = np.average(dist_array, axis=0)
        dists = np.array(oaiu.distances_from_embeddings(self.tolist(average_embedding), self.tolist(dist_array), distance_metric='cosine'))
        a = np.array(dists)
        no_outlier_list = np.array(self.remove_outliers(a))
        self.reject_threshold = no_outlier_list.max()*2.0
        print("\t[{}] reject threshold = {:.4f} dists = {}".format(self.name, self.reject_threshold, dists))

        for i in range(len(topics)):
            cur_topic = embd_list[i]['text']
            cur_embed = embd_list[i]['embedding']
            cur_dist = dists[i]
            # print("cur_topic = {}".format(cur_topic))
            # print("cur_embed = {}".format(cur_embed))
            if cur_dist < self.reject_threshold:
                self.known_good_topics_list.append(cur_topic)
                self.known_good_embeddings_list.append(cur_embed)
            else:
                print("\t[{}] is an outlier. Not adding to [{}] known goodlist ".format(cur_topic, self.name))

        a = np.array(self.known_good_embeddings_list)
        self.average_embedding = np.average(a, axis=0)
        # print("average embedding = {}".format(self.average_embedding))
        # dists = np.array(oaiu.distances_from_embeddings(self.tolist(self.average_embedding), self.known_good_embeddings_list, distance_metric='cosine'))
        # a = np.array(dists)
        # self.reject_threshold = a.max()*2.0
        # print("\treject threshold = {:.4f} dists = {}".format(self.reject_threshold, dists))


    def test_add_topic(self, test:str) -> bool:
        test = test.strip()
        embd_list = self.oac.get_embedding_list([test])
        d = embd_list[0]
        test_embed = d['embedding']
        dist_list = oaiu.distances_from_embeddings(self.tolist(self.average_embedding), [test_embed], distance_metric='cosine')
        dist:float = dist_list[0]
        s = 'REJECT'
        if dist < self.reject_threshold:
            s = 'ACCEPT'
            self.provisional_topics_list.append(test)
            self.provisional_embeddings_list.append(test_embed)
            # print("'{}' is {:.4f} away from '{}' {}".format(self.name, dist, test, s))
            return True

        # print("'{}' is {:.4f} away from '{}' {}".format(self.name, dist, test, s))
        return False

    def to_string(self) -> str:
        s = "Topic '{}' includes:".format(self.name)
        for topic in self.known_good_topics_list:
            s += "\n\t'{}'".format(topic)
        s += "\n\treject_threshold = {:.5f}".format(self.reject_threshold)
        s += "\n\tInbound links = {}".format(len(self.inbound_node_list))
        for tl in self.inbound_node_list:
            s += "\n\t\t{}".format(tl.to_string())
        s += "\n\tOutbound links = {}".format(len(self.outbound_node_list))
        for tl in self.outbound_node_list:
            s += "\n\t\t{}".format(tl.to_string())
        return s

class NodeLink:
    source:TopicNode
    target:TopicNode
    count:int

    def __init__(self, source:TopicNode, target:TopicNode):
        self.count = 1
        self.source = source
        self.target = target

    # See if a source-target set of nodes belong in this list. If it
    # does, increase the count and return True. If not, return False
    # the idea is that we can iterate over this list and exit at the
    # first True. Otherwise, set up a new connection
    def add_test(self, source:TopicNode, target:TopicNode) -> bool:
        if self.source == source and self.target == target:
            self.count += 1
            return True
        return False

    def to_string(self) -> str:
        return "[{}] -> [{}]".format(self.source.name, self.target.name)

def parse_to_list(to_parse:str, regex_str = r"(\n\d+[\): \.])|^(\d[\):\.])") -> List:
    pattern = re.compile(regex_str)
    result = pattern.split(to_parse)
    # Filter out the items that match the regex pattern
    item:str
    result = [item.strip() for item in result if item and not pattern.match(item)]
    return result



def create_topics() -> List:
    engine="gpt-4-0314"
    # initiate the stack with
    query_q = deque(['vaccines cause autism'])
    oac = OpenAIComms()

    max_character_length = 50
    max_topics = 15
    topic_count = 0
    node_list = []
    while len(query_q) > 0:
        print("\nTopic count = {}".format(topic_count))
        query = query_q.pop()
        same_prompt = "Produce a list of the 5 most common phrases that mean the same thing as '{}'. Use concise language (10 words or less).\nList:\n".format(query)
        print("\nPrompt = {}".format(same_prompt))
        cu = ChatUnit(same_prompt)
        response = oac.get_chat_complete([cu], engine=engine)
        print("\tRaw response = {}\n".format(response))
        known_good = parse_to_list(response)
        print("\tcreating node '{}' with known good = {}".format(query, known_good))
        source_node = TopicNode(query, oac)
        source_node.add_known_good_list(known_good)
        node_list.append(source_node)

        related_prompt = "Produce a list of 5 conspiracy theories that are likely to be believed by the same people who believe '{}'. Use concise language (10 words or less).\nList:\n".format(query)
        print("\tPrompt = {}".format(related_prompt))
        cu = ChatUnit(related_prompt)
        response = oac.get_chat_complete([cu], engine=engine)
        related_list = parse_to_list(response)
        print("\trelated list = {}".format(related_list))

        # look through all the responses
        s:str
        for s in related_list:
            if len(s) > max_character_length:
                print("\tSkipping '{}' (exceeds {} chars) ".format(s, max_character_length))
                continue

            print("\ttesting '{}'".format(s))
            good_match = False
            target_node:TopicNode
            for target_node in node_list:
                belongs = target_node.test_add_topic(s)
                if belongs:
                    if target_node != source_node:
                        nl = NodeLink(source_node, target_node)
                        source_node.outbound_node_list.append(nl)
                        target_node.inbound_node_list.append(nl)
                        print("\t\tConnecting {}".format(nl.to_string()))
                    good_match = True
                    break
            if not good_match:
                query_q.append(s)
                print("\t\tAdding '{}' to queue".format(s))
        topic_count += 1
        if topic_count >= max_topics:
            break

    #print out what we have
    print("\nUnhandled topics: {}".format(query_q))


    print("\nNode details ({} TopicNodes)".format(len(node_list)))
    for topic_node in node_list:
        print("\n{}\n".format(topic_node.to_string()))

    return node_list

def main():
    node_list = create_topics()
    ng = NetworkxGraphing(name="Conspiracy", creator="Phil")
    tn:TopicNode
    nl:NodeLink
    for tn in node_list:
        for nl in tn.inbound_node_list:
            ng.add_connected_nodes(nl.source.name, nl.target.name)
        for nl in tn.outbound_node_list:
            ng.add_connected_nodes(nl.source.name, nl.target.name)

    ng.draw("Conspiracy", draw_legend=False, do_show=False, scalar=10)
    plt.show()

    filename = "topicnode_conspiracy.graphml"
    ng.to_gml(filename, graph_creator="phil", node_attributes=['weight'])

if __name__ == "__main__":
    main()