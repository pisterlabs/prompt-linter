#%%
import jsonlines
import pandas as pd
import os
import numpy as np
import re
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
from networkx.algorithms import bipartite
import networkx as nx
from igraph import Graph 
import igraph as ig
import uunet.multinet as ml
from langchain import PromptTemplate
from langchain.llms import OpenAI
from dotenv import load_dotenv
from langchain.chains import LLMChain
from sentence_transformers import SentenceTransformer
import json
import openai
import datetime
load_dotenv()

os.environ['TOKENIZERS_PARALLELISM'] = 'false' # to avoid a warning 
openai.api_key = os.getenv("OPENAI_API_KEY")    


#%%
class Tweets_to_network:
    """
    A class used to represent the entire pipeline from json to graph

    ...
    Attributes
    ----------
    file_tweets : str
        path to the json file containing the tweets
    file_user : str
        path to the json file containing the users
    path : str
        path to the folder containing the json files
    path_cache : str
        path to the folder containing the cache files
    name : str
        name of the folder
    df_tweets : pandas.DataFrame
        dataframe containing all the tweets
    df_original : pandas.DataFrame
        dataframe containing all the tweets that are not retweets, quotes or replies
    df_original_labeled : pandas.DataFrame
        dataframe containing all the tweets that are not retweets, quotes or replies and are labeled with the topic
    df_retweets : pandas.DataFrame
        dataframe containing all the retweets
    df_retweets_labeled : pandas.DataFrame
        dataframe containing all tweets and retweets and are labeled with the topic 
    df_quotes : pandas.DataFrame
        dataframe containing all the quotes
    df_quotes_labeled : pandas.DataFrame
        dataframe containing all tweets and quotes and are labeled with the topic
    df_reply : pandas.DataFrame
        dataframe containing all the replies
    df_reply_labeled : pandas.DataFrame
        dataframe containing all tweets and replies and are labeled with the topic
    proj_graphs : dict
        dictionary containing all the projected graph of users 
    ml_network : uunet.multinet
        multinet object containing the network of the users each layer is a topic

    Methods
    -------
    process_json()
        process the json files and create the dataframes
    get_topics()
        get the topics of the original tweets
    create_network(df -> pandas.DataFrame, title -> str)
        create the network of the given dataset and save in gml format in the network folder
    project_network(path -> dict)
        project the network into multiple one mode network, one for each topic, saved in a dict 
        using networkx 
    create_multilayer_network()
        create the multilayer network using uunet

    """

    def __init__(self,  file_tweets, file_user, n_cop):
        """

        Parameters
        ----------
        file_tweets : str
            path to the json file containing the tweets
        file_user : str
            path to the json file containing the users
        """

        self.file_user = file_user
        self.file_tweets = file_tweets
        self.path = file_tweets.split('/')[:-1]
        self.path = '/'.join(self.path)
        self.path_cache = os.path.join(self.path, 'cache')
        self.name = self.file_tweets.split('/')[-1].split('.')[0]
        self.graph_dir = os.path.join(self.path, 'networks')
        self.model = None
        self.df_tweets = None
        self.df_original = None
        self.df_original_labeled = None
        self.df_retweets = None
        self.df_retweets_labeled = None
        self.df_quotes = None
        self.df_quotes_labeled = None
        self.df_reply = None
        self.df_reply_labeled = None
        self.proj_graphs = {}
        self.ml_network = ml.empty()
        self.topic_labels = None
        self.text = None
        self.n_cop = n_cop
        self.df_original_influencers = None




    def process_json(self):
        """
        Process the json files and create the dataframes

        """
        file = os.path.join(self.path_cache,'tweets_'+self.name+'.pkl')
        file_original = os.path.join(self.path_cache,'tweets_original_'+self.name+'.pkl')

        if os.path.exists(file_original):
            self.df_original_influencers = pd.read_pickle(file_original)
            print('using cached file for original influencer processing')

        # if pkl file tweets_cop22 exists, load it
        if os.path.exists(file):
            print('using cached file for json processing')
            df_tweets = pd.read_pickle(file)
            
        else:
            print('looking into users json file')
            users = {}
            with jsonlines.open(self.file_user) as reader: # open file
                for obj in reader:
                    users[obj['id']] = {'username': obj['username'], 'tweet_count': obj['public_metrics']['tweet_count'], 'followers' : obj['public_metrics']['followers_count'], 'following' : obj['public_metrics']['following_count']}

            df_user = pd.DataFrame(users).T

            print('looking into tweets json file')
            tweets = {}
            attachments = []
            with jsonlines.open(self.file_tweets) as reader: # open file
                for obj in reader:
                    if obj.get('attachments') is  None : # avoid tweets with attachments
                        tweets[obj.get('id', 0)] = {'author': obj['author_id'], 
                                                        'author_name': users.get(obj['author_id'], {}).get('username'),
                                                        'text': obj.get('text', ''), 
                                                        'date': obj.get('created_at', ''),
                                                        'lang':obj.get('lang', ''),
                                                        'reply_count': obj.get('public_metrics', {}).get('reply_count', 0), 
                                                        'retweet_count': obj.get('public_metrics', {}).get('retweet_count', 0), 
                                                        'like_count': obj.get('public_metrics', {}).get('like_count', 0), 
                                                        'quote_count': obj.get('public_metrics', {}).get('quote_count', 0),
                                                        #'impression_count': obj['public_metrics']['impression_count'],
                                                        'conversation_id': obj.get('conversation_id', None),
                                                        'referenced_type': obj.get('referenced_tweets', [{}])[0].get('type', None),
                                                        'referenced_id': obj.get('referenced_tweets', [{}])[0].get('id', None),
                                                        'mentions_name': [ann.get('username', '') for ann in obj.get('entities',  {}).get('mentions', [])],
                                                        'mentions_id': [ann.get('id', '') for ann in obj.get('entities',  {}).get('mentions', [])],
                                                        'cop':  self.n_cop
                                                    # 'context_annotations': [ann.get('entity').get('name') for ann in obj.get('context_annotations', [])]
                                                    }
                    else:
                        attachments.append(obj)

                print('discarded ',len(attachments) ,'tweets with attachments')

            df_tweets = pd.DataFrame(tweets).T

            # create cache folder if not exists and then
            if not os.path.exists(self.path_cache):
                os.makedirs(self.path_cache)
            # save file in the cache folder both csv and pkl format
            df_user.to_csv(os.path.join(self.path_cache,'users_'+self.name+'.csv'))
            df_tweets.to_csv(os.path.join(self.path_cache,'tweets_'+self.name+'.csv'))
            df_tweets.to_pickle(file)
        
        df_tweets = df_tweets[df_tweets['lang'] == 'en'] # consider only english tweets
        df_tweets['author_name'] = df_tweets['author_name'].fillna(df_tweets['author'])
        
        # create the dataframes, devide the tweets in original, retweets, quotes and replies
        self.df_tweets = df_tweets
        self.df_original = df_tweets[df_tweets['referenced_type'].isna()]
        self.df_original_no_retweets = self.df_original[self.df_original['retweet_count'] != 0]
        self.df_retweets = df_tweets[df_tweets['referenced_type'] == 'retweeted']
        self.df_quotes = df_tweets[df_tweets['referenced_type'] == 'quoted']
        self.df_reply = df_tweets[df_tweets['referenced_type'] == 'replied_to']


        return df_tweets

    def get_topics(self, name = 'openai', df = None):
        """
        Get the topics of the original tweets
           
        """
        time = datetime.datetime.now()

        if df is None:
            df = self.df_original

        file = os.path.join(self.path_cache, 'tweets_'+self.name+'_topics.pkl')

        # if the original dataframe is not created, run process_json
        if(self.df_original is None):
            self.process_json()

        # if pkl file tweets_cop22_topics exists, load it
        if os.path.exists(file):
            print('using cached topics')
            df_cop = pd.read_pickle(file)
            self.model = BERTopic.load(os.path.join(self.path_cache,'model_'+self.name+'.pkl'))
        else:
            print('running topic modeling')

            df_cop = df
            # prepare documents for topic modeling
            docs = df_cop['text'].tolist()
            docs = [re.sub(r"http\S+", "", doc) for doc in docs]
            docs = [re.sub(r"@\S+", "", doc) for doc in docs] #  remove mentions 
            docs = [re.sub(r"#\S+", "", doc) for doc in docs] #  remove hashtags
            docs = [re.sub(r"\n", "", doc) for doc in docs] #  remove new lines
            docs = [doc.strip() for doc in docs]#strip 
            
            if(name == 'openai'):
                embs = openai.Embedding.create(input = docs, model="text-embedding-ada-002")['data']
                self.embedder = None
                embeddings = np.array([np.array(emb['embedding']) for emb in embs])
            else:
                self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
                embeddings = self.embedder.encode(docs)

            # fit model
            

            # topic modeling
            vectorizer_model = CountVectorizer(stop_words="english")
            ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
            model = BERTopic( 
                                vectorizer_model =   vectorizer_model,
                                ctfidf_model      =   ctfidf_model,
                                nr_topics        =  'auto',
                                min_topic_size   =   max(int(len(docs)/1000),10),
                                embedding_model  = self.embedder,
                            )


            try:
                topics ,probs = model.fit_transform(docs, embeddings = embeddings)
                df_cop['topic'] = topics    
                df_cop['topic_prob'] = probs        
                model.get_topic_info().to_csv(os.path.join(self.path_cache,'topics_cop22.csv'))
                self.model = model          

            except Exception as e:
                print(e)
                print('error in topic modeling')
                df_cop['topic'] = -1

            # create cache folder if not exists 
            if not os.path.exists(self.path_cache):
                os.makedirs(self.path_cache)

            # save file in the cache folder 
            df_cop.to_pickle(file)
            #save model 
            model.save(os.path.join(self.path_cache,'model_'+self.name+'.pkl'))

        print('topics created in ', datetime.datetime.now() - time)

        # add topics label to the originaldataframe and for the not original tweet put the reference of the original tweet in that field 
        self.df_original_labeled = df_cop
        self.df_retweets['topic'] = self.df_retweets['referenced_id']
        self.df_quotes['topic'] = self.df_quotes['referenced_id']
        self.df_reply['topic'] = self.df_reply['referenced_id']



        print('added topics in ', datetime.datetime.now() - time)
        # merge the dataframes
        self.df_retweets_labeled = pd.concat([self.df_original_labeled, self.df_retweets])
        self.df_quotes_labeled = pd.concat([self.df_original_labeled, self.df_quotes])
        self.df_reply_labeled = pd.concat([self.df_original_labeled, self.df_reply])

        print('merged topics in ', datetime.datetime.now() - time)
        # this is required for propagating the topic to the retweets
        # def resolve_topic(df, row_id,):
        #     if row_id is not  None:
        #         if  isinstance(row_id, int): # the topic 
        #             return int(row_id)
        #         else: # the pointer 
        #             try:
        #                 topic = df.loc[row_id, 'topic']
        #                 return resolve_topic(df, topic)
        #             except: # if there is not the referenced tweet we discard the tweet
        #                 return -1

        # #self.df_retweets_labeled['topic'] = self.df_retweets_labeled['topic'].map(lambda row: resolve_topic(self.df_retweets_labeled, row))
        # self.df_retweets_labeled['topic'] = np.vectorize(resolve_topic)(self.df_retweets_labeled, self.df_retweets_labeled['topic'])
        df = self.df_retweets_labeled
        topic_dict = df['topic'].to_dict()
        for key, value in topic_dict.items():
            while isinstance(value, str):
                if value not in topic_dict:
                    break
                value = topic_dict[value]
            topic_dict[key] = value
        self.df_retweets_labeled['topic'] = df.index.map(topic_dict)

        # count how many tweets have a string topics 
        print('counting string topics')
        print(len(self.df_retweets_labeled[self.df_retweets_labeled['topic'].apply(lambda x: isinstance(x, str))]))
        # discard them 
        self.df_retweets_labeled = self.df_retweets_labeled[self.df_retweets_labeled['topic'].apply(lambda x: not isinstance(x, str))]


        #self.df_quotes_labeled['topic'] = self.df_quotes_labeled['topic'].map(lambda row: resolve_topic(self.df_quotes_labeled, row))
        #self.df_reply_labeled['topic'] = self.df_reply_labeled['topic'].map(lambda row: resolve_topic(self.df_reply_labeled, row))

        print('topic resolved', datetime.datetime.now() - time)


        # remove the tweets that have not a topic
        self.df_retweets_labeled = self.df_retweets_labeled[self.df_retweets_labeled['topic'].notna()]
       # self.df_quotes_labeled = self.df_quotes_labeled[self.df_quotes_labeled['topic'].notna()]
        #self.df_reply_labeled = self.df_reply_labeled[self.df_reply_labeled['topic'].notna()]

        print('removed tweets without topic', datetime.datetime.now() - time)

        # topic to int 
        self.df_retweets_labeled['topic'] = self.df_retweets_labeled['topic'].astype(int)
        #self.df_quotes_labeled['topic'] = self.df_quotes_labeled['topic'].astype(int)
        #self.df_reply_labeled['topic'] = self.df_reply_labeled['topic'].astype(int)

        print('topic to int', datetime.datetime.now() - time)
        # save df_retwets_labeled
        self.df_retweets_labeled.to_pickle(os.path.join(self.path_cache,'retweets_labeled_'+self.name+'.pkl'))

        print('saved df_retweets_labeled', datetime.datetime.now() - time)
        return df_cop

    def create_network(self, df_tweets, title, project = True):
        """
        Create a network from the dataframe of tweets and save it in a gml file
        """
        # if author name is none put author id 
        

        print('create network ' + title)
       
        A = df_tweets['author_name'].unique() # actors
        M = df_tweets.index                   # tweets 
        x = df_tweets['text'].to_dict()
        topics = df_tweets['topic'].to_dict()
        author = df_tweets['author_name'].to_dict()
        is_retweet = df_tweets['referenced_type'].to_dict()
        # none to 'original
        is_retweet = {k: 'original' if v is None else v for k, v in is_retweet.items()}
        g = nx.DiGraph()
        g.add_nodes_from(A, bipartite=0) # author of tipe 0
        g.add_nodes_from(M, bipartite=1) # tweets of type 1


        # list of tuples between author_nname and index 
        edges = list(zip(df_tweets['author_name'], df_tweets.index)) # author-> tweet
        ref_edges = list(zip( df_tweets.index, df_tweets['referenced_id'])) # retweet -> tweet
        ref_edges = [i for i in ref_edges if i[1] is not None] # remove all none values
        men_edges = [(row.Index, mention) for row in df_tweets.itertuples() for mention in row.mentions_name]


        g.add_edges_from(edges, weight = 10 )
        g.add_edges_from(ref_edges, weight = 1)

   

        # remove all nodes authomatically addd 

        #add attribute bipartite to all nides without it, required for the new tweets added with the ref
        nodes_to_remove = [node for node in g.nodes if 'bipartite' not in g.nodes[node]]
        g.remove_nodes_from(nodes_to_remove)
                # g.nodes[i]['bipartite'] = 1


        date_lookup = {row.Index: row.date for row in df_tweets.itertuples()}

        t = {e: date_lookup[e[1]] for e in g.edges()}
        g.add_edges_from(men_edges)

        # set bipartite = 0 ( so actors) to the new nodes(users) added 
        nodes_to_set = [node for node in g.nodes if 'bipartite' not in g.nodes[node]]
        [g.nodes[node].setdefault('bipartite', 0) for node in nodes_to_set]


        nx.set_edge_attributes(g, t, 'date')
        nx.set_node_attributes(g, x, 'text')
        nx.set_node_attributes(g, topics, 'topics')
        nx.set_node_attributes(g, author, 'author')
        nx.set_node_attributes(g, is_retweet, 'is_retweet')

        #self.graph_dir = os.path.join(self.path, 'networks')

        if not os.path.exists(self.graph_dir):
            os.makedirs(self.graph_dir)

        filename = os.path.join(self.graph_dir,self.name+'_'+title+'.gml')
        nx.write_gml(g, filename)
        print('network created at', filename)

        if project:
            self.project_network(path = filename , title = title)

        self.text = x

        return (g, x, t)
    
    def retweet_network(self, df = None):
        if df is None:
            try:
                df = self.df_retweets_labeled
            except:
                self.get_topics(name = 'bert')
                df = self.df_retweets_labeled
        

        
        G = nx.DiGraph()
        
        authors = df['author'].unique()
        G.add_nodes_from(authors)

        for i, row in df.iterrows():
            ref_id = row['referenced_id']

            if ref_id is not None:
            # if the edge already exists add 1 to the weight
                if G.has_edge(row['author'], df.loc[str(ref_id)]['author']):
                    G[row['author']][df.loc[str(ref_id)]['author']]['weight'] += 1
                else:
                    G.add_edge(row['author'], df.loc[str(ref_id)]['author'], weight=1)
            
        self.retweet_graph = G

        # save GML file
        if not os.path.exists(self.graph_dir):
            os.makedirs(self.graph_dir)

        filename = os.path.join(self.graph_dir,self.name+'_retweet_network.gml')
        nx.write_gml(G, filename)

        return G
    
    def get_n_influencers(self, n = 1000):
        '''
        Get the top n most retweeted users  of the network and get their original tweets
        '''

        indegree = self.retweet_graph.in_degree()

        indegree_df = pd.DataFrame(indegree, columns = ['author', 'indegree']).set_index('author').sort_values(by = 'indegree', ascending = False)
        rt_count = self.df_retweets_labeled.groupby('author')[['author_name', 'retweet_count']].sum()

        #merge the two df
        indegree_df = indegree_df.merge(rt_count, left_index = True, right_index = True)

        influencers = indegree_df.head(n).index

        #get df_original of top 1000
        infleuncers_df = self.df_original[self.df_original['author'].isin(influencers)]

        return infleuncers_df



    def retweet_network_ml(self, df = None):
        if df is None:
            df = self.df_retweets_labeled
        
        topics = df['topic'].unique()

        ml_network = ml.empty()

        for topic in topics:
            G = nx.DiGraph()
            df_tmp = df[df['topic'] == topic]
            G.add_nodes_from(df_tmp['author'].unique())

            for i, row in df_tmp.iterrows():
                ref_id = row['referenced_id']

                if ref_id is not None:
                # if the edge already exists add 1 to the weight
                    if G.has_edge(row['author'], df_tmp.loc[str(ref_id)]['author']):
                        G[row['author']][df_tmp.loc[str(ref_id)]['author']]['weight'] += 1
                    else:
                        G.add_edge(row['author'], df_tmp.loc[str(ref_id)]['author'], weight=1)
                        
            ml.add_nx_layer(ml_network, G , str(topic))

        self.ml_network = ml_network
        
        # save GML file
        if not os.path.exists(self.graph_dir):
            os.makedirs(self.graph_dir)

        filename = os.path.join(self.graph_dir,self.name+'_retweet_network_ml.gml')
        ml.write(ml_network, file = filename)


        return ml_network

    def project_network(self, path = None, nx_graph = None, title = None):
        """
        Project a network from a gml file into multiple networks based on the topic of the tweets
        """


        def recursive_explore(graph, node,start_node, previous_node = None , edges = None, topic= None, depth = 0):

            neighbors = graph.neighborhood(node, mode='out') 
            
            # if it is the first we initialize the edges
            if edges is None:
                edges = {}
            
            # it is a user
            if node['bipartite'] == 0.0 :
                # if there is only one node in the middle it is a mention
                if depth == 2 :
                    edges.setdefault(topic, []).append((start_node['label'], node['label']))
                    return
                # in this  case we have a retweet  
                elif depth > 2 :
                    edges.setdefault(topic, []).append((start_node['label'], previous_node['author']))
                    return
            # it is a tweet
            else:
                if topic is None:
                    topic = node['topics']
                # end of the chain it is a retweet without mention
                if (len(neighbors) == 1):
                    edges.setdefault(topic, []).append((start_node['label'], node['author']))
                    return

            # ecplore all the neighbors
            for neighbor in neighbors[1:]:
                new_node = g.vs[neighbor]
                recursive_explore(graph, node = new_node, previous_node = node, start_node = start_node, depth = depth+1, edges= edges, topic= topic)

            return edges

        if path is not None:
            g =  Graph.Read_GML(path)
            
        elif nx_graph is not None:
            g = Graph.from_networkx(nx_graph)
            g.vs['label'] = g.vs['_nx_name']
            
        else:
            print('provide a graph of a gml file o a networkx graph')
            return None

        edges = {}
        ml_network = ml.empty()

        # for each user
        for n in g.vs.select(bipartite=0):
            # get all neighbors of g
            visited = set()
            result = recursive_explore(g, n, start_node = n)
            #print(result)
            edges = {key: edges.get(key, []) + result.get(key, []) for key in set(edges) | set(result)}

        edges = {e: set(edges[e]) for e in edges }
        edges.pop(None, None)
        # drop key 
        print('projected network created')

        prj_dir = os.path.join(self.graph_dir, 'projected')

        if not os.path.exists(prj_dir):
            os.makedirs(prj_dir)
       
        for t, e in edges.items():
            print(t, len(e))
            if t != 'NaN':
                self.proj_graphs[t] = nx.from_edgelist(e, create_using=nx.DiGraph())
                nx.write_gml(self.proj_graphs[t], os.path.join(prj_dir, self.name+'_'+title+'_prj_'+str(t)+'.gml'))
                ml.add_nx_layer(ml_network, self.proj_graphs[t], str(t))
        
        ml.write(ml_network, file = os.path.join(self.graph_dir, self.name+'_'+title+'_ml.gml'))
        # save file 
       # Graph.write_gml(g, os.path.join(self.graph_dir, self.name+title+'_prj.gml'))

        
        return self.proj_graphs

    def create_multilayer_network(self, title = None):
        """
        Create a multilayer network from the projected networks using the library 
        developed by infolab at Uppsala university

        """
        if self.proj_graphs == {}:
            print('project the network first')
        else:
        
            for t, g in self.proj_graphs.items():
                ml.add_nx_layer(self.ml_network, g, str(t))
        
        prj_dir = os.path.join(self.graph_dir, 'projected')
        
        ml.write(self.ml_network, file = os.path.join(prj_dir, title+'_ml.gml'))
            
        return self.ml_network

    def label_topics(self):
        llm = OpenAI(temperature=0.3)

        template = """I want you to act as a tweet labeler, you are given representative words
            from a topic and three representative tweets, give more attention to the words, all the tweets are related to climate change, and COP, no need to mention it, detect subtopics.
            start with "label:" and avoid hashtags,
            which is a good short label for the topic containing the words [{words}], here you are 3 tweets to help you:
            first = \"{tweet1}\", second = \"{tweet2}\", third = \"{tweet3}\""""


        prompt = PromptTemplate(
            input_variables=["words", "tweet1", "tweet2", "tweet3"],
            template=template,
        )


        chain = LLMChain(llm=llm, prompt=prompt)

        topics = list(self.model.get_topic_info()['Topic']) # get inferred topics 
        topic_words = self.model.get_topics() # get words for each topic
        labels = {}

        for topic in topics:
            tweets = self.model.get_representative_docs(topic)
            words = [word[0] for word in topic_words[topic]]
            labels[topic] = chain.run(words=words, tweet1=tweets[0], tweet2=tweets[1], tweet3=tweets[2])

        # remove \n from values of labels 

        labels = {key: value.replace('\n', '') for key, value in labels.items()} 
        labels = {key: value.replace('Label:', '') for key, value in labels.items()}
        #strip 
        labels = {key: value.strip() for key, value in labels.items()}

        self.topic_labels = labels

        #save in file 
        with open(os.path.join(self.path_cache, 'labels_'+self.name+'.json'), 'w') as fp:
            json.dump(labels, fp)
            
        return labels


# %%

