"""Main class
"""

from pytwanalysis.py_twitter_db import TwitterDB
from pytwanalysis.py_twitter_graphs import TwitterGraphs
from pytwanalysis.py_twitter_topics import TwitterTopics

#from pyTwitterGraphAnalysis import tw_graph
#from pyTwitterDB import tw_database
#from pyTwitterTopics import tw_topics


from pymongo import MongoClient
import networkx as nx
import numpy as np
import os
import datetime
import csv
import pandas as pd
import matplotlib.pyplot as plt
import time

import warnings
warnings.filterwarnings("ignore")


MIN_NO_OF_NODES_TO_REDUCE_GRAPH = 100



class TwitterAnalysis(TwitterGraphs, TwitterDB, TwitterTopics):
    """
    Main class - It inherits TwitterGraphs, TwitterDB, and TwitterTopics classes.    
    """
    def __init__(
            self, 
            base_folder_path, 
            mongoDB_database):
        
        TwitterGraphs.__init__(self, base_folder_path)
        TwitterDB.__init__(self, mongoDB_database)
        TwitterTopics.__init__(self, base_folder_path, mongoDB_database)

        self.type_of_graph = 'user_conn_all'
        self.is_bot_Filter = None
        self.period_arr = None
        self.create_nodes_edges_files_flag = 'Y'
        self.create_graphs_files_flag ='Y' 
        self.create_topic_model_files_flag = 'Y' 
        self.create_ht_frequency_files_flag = 'Y' 
        self.create_words_frequency_files_flag = 'Y'
        self.create_timeseries_files_flag = 'Y'
        self.create_top_nodes_files_flag = 'Y' 
        self.create_community_files_flag = 'N'
        self.create_ht_conn_files_flag = 'Y'
        self.num_of_topics = 4
        self.top_no_word_filter = None
        self.top_ht_to_ignore = None
        self.graph_plot_cutoff_no_nodes = 500
        self.graph_plot_cutoff_no_edges = 2000
        self.create_graph_without_node_scale_flag = 'N'
        self.create_graph_with_node_scale_flag = 'Y'
        self.create_reduced_graph_flag = 'Y'
        self.reduced_graph_comty_contract_per = 90 
        self.reduced_graph_remove_edge_weight = None
        self.reduced_graph_remove_edges = 'Y'
        self.top_degree_start = 1
        self.top_degree_end = 10
        self.period_top_degree_start = 1
        self.period_top_degree_end = 5
        self.commty_edge_size_cutoff = 200
        self.user_conn_filter = None
        self.edge_prefix_str = 'UserConnections_'
        

        
    #####################################
    # Method: setConfigs
    # Description: Configure objects settings
    def setConfigs(
            self,
            type_of_graph='user_conn_all', 
            is_bot_Filter=None, 
            period_arr=None, 
            create_nodes_edges_files_flag='Y', 
            create_graphs_files_flag='Y', 
            create_topic_model_files_flag='Y', 
            create_ht_frequency_files_flag='Y', 
            create_words_frequency_files_flag='Y', 
            create_timeseries_files_flag='Y',   
            create_top_nodes_files_flag = 'Y', 
            create_community_files_flag = 'N', 
            create_ht_conn_files_flag='Y',
            num_of_topics=4, 
            top_no_word_filter=None, 
            top_ht_to_ignore=None, 
            graph_plot_cutoff_no_nodes=500, 
            graph_plot_cutoff_no_edges=2000,
            create_graph_without_node_scale_flag='N', 
            create_graph_with_node_scale_flag='Y', 
            create_reduced_graph_flag='Y',
            reduced_graph_comty_contract_per=90,  
            reduced_graph_remove_edge_weight=None, 
            reduced_graph_remove_edges='Y',                            
            top_degree_start=1, 
            top_degree_end=10, 
            period_top_degree_start=1, 
            period_top_degree_end=5, 
            commty_edge_size_cutoff=200):
            
        """
        Configure the current object settings to drive the automation of the analysis files
                
        Parameters
        ----------               
        type_of_graph : (Optional)
            This setting defines the type of graph to analyze. Six different options are available: user_conn_all, user_conn_retweet, user_conn_quote, user_conn_reply, user_conn_mention, and ht_conn.
            (Default='user_conn_all')
        
        is_bot_Filter : (Default=None)
        
        period_arr : (Optional)
            An array of start and end dates can be set so that the pipeline creates a separate analysis folder for each of the periods in the array. (Default=None)
        
        create_nodes_edges_files_flag : (Optional)
            If this setting is set to 'Y', the pipeline will create two files for each graph and sub-graph. One file with the edge list, and one with the node list and their respective degree.(Default='Y') 
        
        create_graphs_files_flag : (Optional)
            If this setting is set to 'Y', the pipeline will plot the graph showing all the connections.
            (Default='Y')
        
        create_topic_model_files_flag : (Optional)
            If this setting is set to 'Y', the pipeline will create topic discovery related files for each folder. It will create a text file with all the tweets that are part of that folder, it will also train a LDA model based on the tweets texts and plot a graph with the results.
            (Default='Y')
        
        create_ht_frequency_files_flag : (Optional)
            If this setting is set to 'Y', the pipeline will create hashtag frequency files for each folder. It will create a text file with the full list of hashtags and their frequency, a wordcloud showing the most frequently used hashtags, and barcharts showing the top 30 hashtags.
            (Default='y')'
        
        create_words_frequency_files_flag : (Optional)
            If this setting is set to 'Y', the pipeline will create word frequency files for each folder. It will create a text file with a list of words and their frequency, a wordcloud showing the most frequently used words, and barcharts showing the top 30 words.
            (Default='Y') 
        
        create_timeseries_files_flag : (Optional)
            If this setting is set to 'Y', the pipeline will create timeseries graphs for each folder representing the tweet count by day, and the top hashtags frequency count by day.
            (Default='Y')    
        
        create_top_nodes_files_flag : (Optional)
            If this setting is set to 'Y', the pipeline will create separate analysis folders for all the top degree nodes.
            (Default='Y') 
        
        create_community_files_flag : (Optional)
            If this setting is set to 'Y', the pipeline will use the louvain method to assign each node to a community. A separate folder for each of the communities will be created with all the analysis files.
            (Default='N') 
        
        create_ht_conn_files_flag : (Optional)
            If this setting is set to 'Y', the pipeline will plot hashtag connections graphs. This can be used when user connections are being analyzed, but it could still be interesting to see the hashtags connections made by that group of users.
            (Default='Y') 
        
        num_of_topics : (Optional)
            If the setting *CREATE_TOPIC_MODEL_FILES_FLAG* was set to 'Y', then this number will be used to send as input to the LDA model. If no number is given, the pipeline will use 4 as the default value.
            (Default=4) 
        
        top_no_word_filter : (Optional)
            If the setting *CREATE_WORDS_FREQUENCY_FILES_FLAG* was set to 'Y', then this number will be used to decide how many words will be saved in the word frequency list text file. If no number is given, the pipeline will use 5000 as the default value.
            (Default=None)
        
        top_ht_to_ignore : (Optional)
            If the setting *CREATE_HT_CONN_FILES_FLAG* was set to 'Y', then this number will be used to choose how many top hashtags can be ignored. Sometimes ignoring the main hashtag can be helpful in visualizations to discovery other interesting structures within the graph.
            (Default=None)
        
        graph_plot_cutoff_no_nodes : (Optional)
            Used with the graph_plot_cutoff_no_edges parameter. For each graph created, these numbers will be used as cutoff values to decide if a graph is too large to be plot or not. Choosing a large number can result in having the graph to take a long time to run. Choosing a small number can result in graphs that are too reduced and with little value or even graphs that can't be printed at all because they can't be reduce further. 
            (Default=500) 
        
        graph_plot_cutoff_no_edges : (Optional) 
            Used with the graph_plot_cutoff_no_nodes parameter. For each graph created, these numbers will be used as cutoff values to decide if a graph is too large to be plot or not. Choosing a large number can result in having the graph to take a long time to run. Choosing a small number can result in graphs that are too reduced and with little value or even graphs that can't be printed at all because they can't be reduce further. 
            (Default=2000)
        
        create_graph_without_node_scale_flag : (Optional)
            For each graph created, if this setting is set to 'Y', the pipeline will try to plot the full graph with no reduction and without any logic for scaling the node size.
            (Default='N') 
        
        create_graph_with_node_scale_flag : (Optional)
            For each graph created, if this setting is set to 'Y', the pipeline will try to plot the full graph with no reduction, but with additional logic for scaling the node size.
            (Default='Y') 
        
        create_reduced_graph_flag : (Optional)
            For each graph created, if this setting is set to 'Y', the pipeline will try to plot the reduced form of the graph.
            (Default='Y') 
        
        reduced_graph_comty_contract_per : (Optional)
            If the setting *CREATE_REDUCED_GRAPH_FLAG* was set to 'Y', then this number will be used to reduce the graphs by removing a percentage of each community found in that particular graph. The logic can be run multiple times with different percentages. For each time, a new graph file will be saved with a different name according to the parameter given.
            (Default=90) 
        
        reduced_graph_remove_edge_weight : (Optional)
            If the setting *CREATE_REDUCED_GRAPH_FLAG* was set to 'Y', then this number will be used to reduce the graphs by removing edges that have weights smaller then this number. The logic can be run multiple times with different percentages. For each time, a new graph file will be saved with a different name according to the parameter given.
            (Default=None)
        
        reduced_graph_remove_edges : (Optional)
            If this setting is set to 'Y', and the setting *CREATE_REDUCED_GRAPH_FLAG was set to 'Y', then the pipeline will continuously try to reduce the graphs by removing edges of nodes with degrees smaller than this number. It will stop the graph reduction once it hits the the values set int the GRAPH_PLOT_CUTOFF parameters.
            (Default='Y')     
        
        top_degree_start : (Optional)
            If the setting  *CREATE_TOP_NODES_FILES_FLAG* was set to 'Y', then these numbers will define how many top degree node sub-folders to create.
            (Default=1) 
        
        top_degree_end : (Optional)
            If the setting  *CREATE_TOP_NODES_FILES_FLAG* was set to 'Y', then these numbers will define how many top degree node sub-folders to create.
            (Default=10) 
        
        period_top_degree_start : (Optional)
            If the setting  *CREATE_TOP_NODES_FILES_FLAG* was set to 'Y', then these numbers will define how many top degree node sub-folders for each period to create.
            (Default=1) 
        
        period_top_degree_end : (Optional)
            If the setting  *CREATE_TOP_NODES_FILES_FLAG* was set to 'Y', then these numbers will define how many top degree node sub-folders for each period to create.
            (Default=5) 
        
        commty_edge_size_cutoff : (Optional)
            If the setting textit{CREATE_COMMUNITY_FILES_FLAG} was set to 'Y', then this number will be used as the community size cutoff number. Any communities that have less nodes then this number will be ignored. If no number is given, the pipeline will use 200 as the default value.
            (Default=200)                 
        
        Examples
        --------          
        ...:
        
            >>> setConfigs(type_of_graph=TYPE_OF_GRAPH,
            >>>             is_bot_Filter=IS_BOT_FILTER,
            >>>             period_arr=PERIOD_ARR,
            >>>             create_nodes_edges_files_flag=CREATE_NODES_EDGES_FILES_FLAG, 
            >>>             create_graphs_files_flag=CREATE_GRAPHS_FILES_FLAG,
            >>>             create_topic_model_files_flag=CREATE_TOPIC_MODEL_FILES_FLAG,
            >>>             create_ht_frequency_files_flag=CREATE_HT_FREQUENCY_FILES_FLAG,
            >>>             create_words_frequency_files_flag=CREATE_WORDS_FREQUENCY_FILES_FLAG,
            >>>             create_timeseries_files_flag=CREATE_TIMESERIES_FILES_FLAG,
            >>>             create_top_nodes_files_flag=CREATE_TOP_NODES_FILES_FLAG, 
            >>>             create_community_files_flag=CREATE_COMMUNITY_FILES_FLAG,
            >>>             create_ht_conn_files_flag=CREATE_HT_CONN_FILES_FLAG,
            >>>             num_of_topics=NUM_OF_TOPICS, 
            >>>             top_no_word_filter=TOP_NO_WORD_FILTER, 
            >>>             top_ht_to_ignore=TOP_HT_TO_IGNORE,
            >>>             graph_plot_cutoff_no_nodes=GRAPH_PLOT_CUTOFF_NO_NODES, 
            >>>             graph_plot_cutoff_no_edges=GRAPH_PLOT_CUTOFF_NO_EDGES,
            >>>             create_graph_without_node_scale_flag=CREATE_GRAPH_WITHOUT_NODE_SCALE_FLAG, 
            >>>             create_graph_with_node_scale_flag=CREATE_GRAPH_WITH_NODE_SCALE_FLAG, 
            >>>             create_reduced_graph_flag=CREATE_REDUCED_GRAPH_FLAG,
            >>>             reduced_graph_comty_contract_per=REDUCED_GRAPH_COMTY_PER,
            >>>             reduced_graph_remove_edge_weight=REDUCED_GRAPH_REMOVE_EDGE_WEIGHT, 
            >>>             reduced_graph_remove_edges=REDUCED_GRAPH_REMOVE_EDGES_UNTIL_CUTOFF_FLAG,                            
            >>>             top_degree_start=TOP_DEGREE_START, 
            >>>             top_degree_end=TOP_DEGREE_END, 
            >>>             period_top_degree_start=PERIOD_TOP_DEGREE_START, 
            >>>             period_top_degree_end=PERIOD_TOP_DEGREE_END, 
            >>>             commty_edge_size_cutoff=COMMTY_EDGE_SIZE_CUTOFF
            >>>             )            
            
        """
                
        self.type_of_graph = type_of_graph
        self.is_bot_Filter = is_bot_Filter
        self.period_arr = period_arr
        self.create_nodes_edges_files_flag = create_nodes_edges_files_flag
        self.create_graphs_files_flag = create_graphs_files_flag
        self.create_topic_model_files_flag = create_topic_model_files_flag
        self.create_ht_frequency_files_flag = create_ht_frequency_files_flag
        self.create_words_frequency_files_flag = create_words_frequency_files_flag
        self.create_timeseries_files_flag = create_timeseries_files_flag
        self.create_top_nodes_files_flag = create_top_nodes_files_flag
        self.create_community_files_flag = create_community_files_flag
        self.create_ht_conn_files_flag = create_ht_conn_files_flag
        self.num_of_topics = num_of_topics
        self.top_no_word_filter = top_no_word_filter
        self.top_ht_to_ignore = top_ht_to_ignore
        self.graph_plot_cutoff_no_nodes = graph_plot_cutoff_no_nodes
        self.graph_plot_cutoff_no_edges = graph_plot_cutoff_no_edges
        self.create_graph_without_node_scale_flag = create_graph_without_node_scale_flag
        self.create_graph_with_node_scale_flag = create_graph_with_node_scale_flag
        self.create_reduced_graph_flag = create_reduced_graph_flag
        self.reduced_graph_comty_contract_per = reduced_graph_comty_contract_per
        self.reduced_graph_remove_edge_weight = reduced_graph_remove_edge_weight
        self.reduced_graph_remove_edges = reduced_graph_remove_edges
        self.top_degree_start = top_degree_start
        self.top_degree_end = top_degree_end
        self.period_top_degree_start = period_top_degree_start
        self.period_top_degree_end = period_top_degree_end
        self.commty_edge_size_cutoff = commty_edge_size_cutoff
                
        if self.type_of_graph == 'user_conn_all':
            self.edge_prefix_str = 'UserConnections_'                
        elif self.type_of_graph == 'user_conn_mention':
            self.edge_prefix_str = 'MentionUserConnections_'
            self.user_conn_filter = 'mention'
        elif self.type_of_graph == 'user_conn_retweet':
            self.edge_prefix_str = 'RetweetUserConnections_'
            self.user_conn_filter = 'retweet'
        elif self.type_of_graph == 'user_conn_reply':
            self.edge_prefix_str = 'ReplyUserConnections_'
            self.user_conn_filter = 'reply'
        elif self.type_of_graph == 'user_conn_quote':
            self.edge_prefix_str = 'QuoteUserConnections_'
            self.user_conn_filter = 'quote'
        elif self.type_of_graph == 'ht_conn':
            self.edge_prefix_str = 'HTConnection_'
            self.export_type = 'ht_edges'
        
        
    #####################################
    # Method: create_path
    # Description: creates a path to add the files for this node        
    def create_path(self, path):       
        if not os.path.exists(path):
            os.makedirs(path)
            
    #####################################
    # Method: get_now_dt
    # Description: returns formated current timestamp to be printed
    def get_now_dt(self):
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    #####################################
    # Method: concat_edges
    # Description: aux function to concatenate edges to help filter in mongoDB
    def concat_edges(self, G):
        """
        Aux function to concatenate edges to help filter in mongoDB
                
        Parameters
        ----------               
        G : 
            undirected networkx graph created from the Twitter data                               
        
        Returns
        -------
        arr_edges
            the array with the concatenatd edges            
    
        Examples
        --------          
        Create an array of concatenated edges from a networkx graph:
        
            >>> arr_edges = concat_edges(G)
        """
        
        arr_edges = []                        
        for u,v,a in G.edges(data=True):
            arr_edges.append(u.lower() + '-' +v.lower())
            arr_edges.append(v.lower() + '-' +u.lower())
                                    
        return arr_edges

    #####################################
    # Method: build_db_collections
    # Description: Call methods to create all collections in mongoDB
    def build_db_collections(self, inc=100000, bots_ids_list_file=None):
        """
        This method is in charge of extracting, cleaning, and loading the data 
        into all the collections in MongoDB. 
       
        Parameters
        ----------               
        inc : (Optional)
            used to determine how many tweets will be processed at a time - (Default=100000).
            A large number may cause out of memory errors, and a low number may take a long time to run, 
            so the decision of what number to use should be made based on the hardware specification.
              
        bots_ids_list_file : (Optional)
            a file that contains a list of user ids that are bots. 
            It creates flags in the MongoDB collection to indentify 
            which tweets and user are in the bots list. - (Default=None)
                 
    
        Examples
        --------          
        Load all data into all collections in MongoDB:
            
            >>> inc = 50000
            >>> build_db_collections(inc)
        """
                
        ### Loading Focused Data into MongoDB
        self.loadFocusedData(inc)
        
        ### Loading user information to collection
        # Loading user information for the actual tweet document
        self.loadUsersData(inc, 'tweet')
        # Loading user information for the original tweet in case of retweets
        self.loadUsersData(inc, 'retweet')
        # Loading user information for the quoted tweet
        self.loadUsersData(inc, 'quote')
        # Loading user information for replies - 
        # (in this case we we don't have full information about the user. Only screen_name and user_id)
        self.loadUsersData(inc, 'reply')
        # Loading user information for mention - 
        # (in this case we we don't have full information about the user. Only screen_name and sometimes user_id)
        self.loadUsersData(inc, 'mention')
        
        ### Breaking tweets into Words        
        self.loadWordsData(inc)
        
        ### Loading tweet connections - 
        # These are the edges formed between users by replies, retweets, quotes and mentions
        self.loadTweetConnections(inc) 
                                
        ### Loading tweet hashtag connections - 
        # These are the edges formed between hash tags being used together in the same tweet        
        self.loadTweetHTConnections(inc) 
        
        #####
        ### loading aggregate collections
        self.loadAggregations('tweetCountByFile')
        self.loadAggregations('tweetCountByLanguageAgg')
        self.loadAggregations('tweetCountByMonthAgg')

        
        # Loading bots list from file - (List of user ids that are bots)
        # SKIP this step if you don't have a bots list        
        if bots_ids_list_file is not None:
            bots_list_id_str = []
            with open(bots_ids_list_file,'r') as f:
                for line in f:
                    line = line.rstrip("\n")
                    bots_list_id_str.append(line)

            self.set_bot_flag_based_on_arr(bots_list_id_str, 10000) 

        


                            
    #####################################
    # Method: plot_graph_contracted_nodes
    # Description: aux function to plot graph. 
    # This steps repets in different parts of this code, so creating a function to avoid repetition
    def plot_graph_contracted_nodes(self, G, file):
        """
        Method to compress and plot graph based on the graph reduction 
        settings that can be updated using the *setConfigs* method.                
                
        Parameters
        ----------               
        G : 
            undirected networkx graph created from the Twitter data
        file :
            the path and name of the graph you want to save
                
        Example
        --------                          
            >>> plot_graph_contracted_nodes(G, 'c:\\Data\\MyGraph.png')
        """

        G2 = G.copy()   
        
        if len(G2.nodes()) > MIN_NO_OF_NODES_TO_REDUCE_GRAPH:                    
                    
            contraction_name = ''

            print("Graph to plot before changes: nodes=" + str(len(G2.nodes)) + " edges=" + str(len(G2.edges)))
            
            
            #in case we want to reduce the graph with edges of weight less than a cutoff number 
            if self.reduced_graph_remove_edge_weight is not None:               
                #find list of edges that have that weigh cutoff
                edges_to_remove = [edge for edge in list(G2.edges(data=True)) if edge[2]['weight'] <= self.reduced_graph_remove_edge_weight]                
                #remove edges for the list
                G2.remove_edges_from(edges_to_remove)                
                #get the largest connected component
                G2 = self.largest_component_no_self_loops(G2)                
                contraction_name = contraction_name + "[RemEdgeWeightLessThan" + str(self.reduced_graph_remove_edge_weight) + "]"
            

            #reduce graph based on a percentage of the nodes for each community
            if self.reduced_graph_comty_contract_per is not None and len(G2.nodes()) > MIN_NO_OF_NODES_TO_REDUCE_GRAPH:
                att = 'community_louvain'                
                G2 = self.contract_nodes_commty_per(G2, self.reduced_graph_comty_contract_per, att)                
                G2 = self.largest_component_no_self_loops(G2)                
                contraction_name = contraction_name + "[RemPercOfComty=" + str(self.reduced_graph_comty_contract_per) + "]"

                
            #In case we want to continue to remove until we get to a cutoff number, another level of contraction
            if self.reduced_graph_remove_edges == 'Y' and len(G2.nodes()) > MIN_NO_OF_NODES_TO_REDUCE_GRAPH:                                
                if len(G2.edges()) > 100000:
                    cutoff_no = 3
                    G2 = self.remove_edges_eithernode(G2, cutoff_no)
                    contraction_name = contraction_name + '[RemEdgeEitherNodeDegCutoff=' + str(cutoff_no) +  ']'
                                    
                cutoff_no = 5
                if (len(G2.nodes()) > self.graph_plot_cutoff_no_nodes) or (len(G2.edges()) > self.graph_plot_cutoff_no_edges):
                    while (len(G2.nodes()) > self.graph_plot_cutoff_no_nodes) or (len(G2.edges()) > self.graph_plot_cutoff_no_edges):
                        
                        G2 = self.remove_edges(G2, cutoff_no)
                        if len(G2.nodes()) > 0:
                            G2 = self.largest_component_no_self_loops(G2)

                        if cutoff_no < 150:
                            cutoff_no += 10
                        elif cutoff_no < 1000:
                            cutoff_no += 100
                        elif cutoff_no < 10000:
                            cutoff_no += 500
                        else:
                            cutoff_no += 1000
                             
                    contraction_name = contraction_name + '[RemEdgeBothNodesDegLessThan=' + str(cutoff_no) +  ']'

            #set up final file name with reduction parameters
            file = file.replace('.', contraction_name + '.')
                

            #get largest connected component after all removals
            if len(G2.edges()) > 0:
                G2 = self.largest_component_no_self_loops(G2)


            #find best settings for the graphs depending on size. You can change these to get better graphs
            if len(G2.edges()) < 450:
                v_scale = 0.01; v_k =0.7; v_iterations=50; v_node_size=2
            elif len(G2.edges()) < 5000:
                v_scale = 2; v_k = 0.6; v_iterations=200; v_node_size=0.8
            elif len(G2.edges()) < 10000:
                v_scale = 1; v_k = 0.1; v_iterations=200; v_node_size=0.6
            elif len(G2.edges()) >= 10000:
                v_scale = 1; v_k = 0.05; v_iterations=500; v_node_size=0.6

            print("Graph to plot after changes: nodes=" + str(len(G2.nodes)) + " edges=" + str(len(G2.edges)))

            
            if (len(G2.nodes()) < self.graph_plot_cutoff_no_nodes and len(G2.edges()) < self.graph_plot_cutoff_no_edges) and len(G2.edges()) != 0:                
                if not os.path.exists(file):
                    G_to_plot, labels2, k = self.calculate_louvain_clustering(G2)                    
                    self.plotSpringLayoutGraph(G_to_plot, 
                                               file, 
                                               v_scale, 
                                               v_k, 
                                               v_iterations, 
                                               cluster_fl='Y', 
                                               v_labels=list(list(labels2)), 
                                               replace_existing_file=False) 



    #####################################
    # Method: export_mult_types_edges_for_input
    # Description: export edges that will be used to create graphs
    # User can choose only one type of graph to export the edges, or export them all
    def export_mult_types_edges_for_input(self, period_arr=None, bot_filter_fl='N', type_of_graph='all'):
        """
        This method will export edges from mongoDB data that can be used to create graphs.
        The user can choose only one type of graph to export the edges, or export them all
                
        Parameters
        ----------
        period_arr : (Optional)
            An array with showing the different periods to be analyzed separatly in the data.
            (Default = None)
            
        bot_filter_fl : (Optional)
            A flag to identify if you want to create extra edge files separating tweets by bots or not.
            This option is only available when the bot flag was updated in mongoDB using method set_bot_flag_based_on_arr.
            (Default='N')
            
        type_of_graph : (Optional)
            the type of graph to export the edges for. 
            Available options: user_conn_all, user_conn_mention, 
            user_conn_retweet, user_conn_reply, user_conn_quote, ht_conn, or all
            (Default='all')
                
        Example
        --------                          
            >>> # Set up the periods you want to analyse 
            >>> # Set period_arr to None if you don't want to analyze separate periods
            >>> # Format: Period Name, Period Start Date, Period End Date
            >>> period_arr = [['P1', '10/08/2017 00:00:00', '10/15/2017 00:00:00'],             
            >>>               ['P2', '01/21/2018 00:00:00', '02/04/2018 00:00:00'],              
            >>>               ['P3', '02/04/2018 00:00:00', '02/18/2018 00:00:00'],
            >>>               ['P4', '02/18/2018 00:00:00', '03/04/2018 00:00:00']]
            >>> 
            >>> 
            >>> ## TYPE OF GRAPH EDGES
            >>> ########################################################
            >>> # You can export edges for one type, or for all
            >>> # Options: user_conn_all,       --All user connections
            >>> #          user_conn_mention,   --Only Mentions user connections
            >>> #          user_conn_retweet,   --Only Retweets user connections
            >>> #          user_conn_reply,     --Only Replies user connections
            >>> #          user_conn_quote,     --Only Quotes user connections
            >>> #          ht_conn              --Hashtag connects - (Hashtgs that wereused together)
            >>> #          all                  --It will export all of the above options
            >>> 
            >>> TYPE_OF_GRAPH = 'all'
            >>> 
            >>> export_mult_types_edges_for_input(period_arr=period_arr, type_of_graph=TYPE_OF_GRAPH)
        """
        
        if type_of_graph == 'all' or type_of_graph == 'user_conn_all':
            self.export_all_edges_for_input(period_arr, bot_filter_fl, type_of_graph='user_conn_all')
        if type_of_graph == 'all' or type_of_graph == 'user_conn_mention':
            self.export_all_edges_for_input(period_arr, bot_filter_fl, type_of_graph='user_conn_mention')
        if type_of_graph == 'all' or type_of_graph == 'user_conn_retweet':
            self.export_all_edges_for_input(period_arr, bot_filter_fl, type_of_graph='user_conn_retweet')
        if type_of_graph == 'all' or type_of_graph == 'user_conn_reply':
            self.export_all_edges_for_input(period_arr, bot_filter_fl, type_of_graph='user_conn_reply')
        if type_of_graph == 'all' or type_of_graph == 'user_conn_quote':
            self.export_all_edges_for_input(period_arr, bot_filter_fl, type_of_graph='user_conn_quote')
        if type_of_graph == 'all' or type_of_graph == 'ht_conn':
            self.export_all_edges_for_input(period_arr, bot_filter_fl, type_of_graph='ht_conn')


        
    #####################################
    # Method: export_all_edges_for_input
    # Description: export edges that will be used to create graphs
    def export_all_edges_for_input(self, period_arr=None, bot_filter_fl='N', type_of_graph='user_conn_all'):
                                
        # Creates path to add the edge files to be used as input
        input_files_path = self.folder_path + '\\data_input_files'
        self.create_path(input_files_path)                    
        
        # 
        edge_prefix_str = ''
        user_conn_filter = None
        export_type = 'edges'
        if type_of_graph == 'user_conn_all':
            edge_prefix_str = 'UserConnections_'                
        elif type_of_graph == 'user_conn_mention':
            edge_prefix_str = 'MentionUserConnections_'
            user_conn_filter = 'mention'
        elif type_of_graph == 'user_conn_retweet':
            edge_prefix_str = 'RetweetUserConnections_'
            user_conn_filter = 'retweet'
        elif type_of_graph == 'user_conn_reply':
            edge_prefix_str = 'ReplyUserConnections_'
            user_conn_filter = 'reply'
        elif type_of_graph == 'user_conn_quote':
            edge_prefix_str = 'QuoteUserConnections_'
            user_conn_filter = 'quote'
        elif type_of_graph == 'ht_conn':
            edge_prefix_str = 'HTConnection_'
            export_type = 'ht_edges'                    

        print("** exporting edges - Graph type=" + type_of_graph )        

        # Export ALL edges for ALL periods
        print("**   exporting edges for AllPeriods " + self.get_now_dt())
        self.exportData(export_type, 
                        input_files_path + '\\' + edge_prefix_str + 'AllPeriods_', 
                        0, 
                        user_conn_filter=user_conn_filter, 
                        replace_existing_file=False)        
                
            
        if bot_filter_fl == 'Y':
            # Export edges for ALL periods, excluding edges associated with bots
            print("**   exporting edges for AllPeriods_ExcludingBots - " + self.get_now_dt())
            self.exportData(export_type, 
                            input_files_path + '\\' + edge_prefix_str + 'AllPeriods_ExcludingBots_',
                            0, 
                            is_bot_Filter = '0', 
                            user_conn_filter=user_conn_filter, 
                            replace_existing_file=False)

            # Export edges for ALL periods, only edges associated with bots
            print("**   exporting edges for AllPeriods_BotsOnly - " + self.get_now_dt())
            self.exportData(export_type, 
                            input_files_path + '\\' + edge_prefix_str + 'AllPeriods_BotsOnly_', 
                            0, 
                            is_bot_Filter = '1', 
                            user_conn_filter=user_conn_filter, 
                            replace_existing_file=False)                        


        # Export edges by period using the dates set on array period_arr
        if period_arr is not None:

            for idx, period in enumerate(period_arr):

                # Export ALL edges for this period    
                print("**   exporting edges for " + period[0] + " - "  + self.get_now_dt())
                edges = self.exportData(export_type, 
                                        input_files_path + '\\' + edge_prefix_str + '' + period[0] + '_', 0, 
                                        startDate_filter=period[1], 
                                        endDate_filter=period[2], 
                                        is_bot_Filter=None, 
                                        user_conn_filter=user_conn_filter, 
                                        replace_existing_file=False)

                if bot_filter_fl == 'Y':
                    # Export edges for this period, excluding edges associated with bots
                    print("**   exporting edges for " + period[0] + "_ExcludingBots - "  + self.get_now_dt())
                    edges = self.exportData(export_type, 
                                            input_files_path + '\\' + edge_prefix_str + '' + period[0] + '_ExcludingBots_', 0, 
                                            startDate_filter=period[1], 
                                            endDate_filter=period[2], 
                                            is_bot_Filter='0', 
                                            user_conn_filter=user_conn_filter, 
                                            replace_existing_file=False) 

                    # Export edges for this period, only edges associated with bots        
                    print("**   exporting edges for " + period[0] + "_BotsOnly - "  + self.get_now_dt())
                    edges = self.exportData(export_type, 
                                            input_files_path + '\\' + edge_prefix_str + '' + period[0] + '_BotsOnly_', 
                                            0, 
                                            startDate_filter=period[1], 
                                            endDate_filter=period[2], 
                                            is_bot_Filter='1', 
                                            user_conn_filter=user_conn_filter, 
                                            replace_existing_file=False)
                        
        print("** exporting edges - END *** - " + self.get_now_dt())


        
        
    #####################################
    # Method: nodes_edges_analysis_files
    # Description: creates nodes and edges files       
    def nodes_edges_analysis_files(self, G, path): 
        """
        Given a graph G, it exports nodes with they degree, edges with their weight, 
        and word clouds representing the nodes scaled  by their degree
                
        Parameters
        ----------               
        G : 
            undirected networkx graph created from the Twitter data                               
        path :
            the path where the files should be saved
    
        Examples
        --------          
            Saved node and edges files into path:
        
            >>> nodes_edges_analysis_files(G, 'C:\\Data\\MyFilePath')
        """            
        
        print("****** Exporting nodes and edges to file - " + self.get_now_dt())
        self.export_nodes_edges_to_file(G, path + "\\G_NodesWithDegree.txt", path + "\\G_Edges.txt")

        print("****** Ploting Nodes Wordcloud - " + self.get_now_dt())        
        node_file_name = path + '\\G_NodesWithDegree.txt'
        df = self.read_freq_list_file(node_file_name,' ')    
        self.plot_word_cloud(df, file=path +'\\G_Nodes_WordCloud.png')
        print("\n")
        
    #####################################
    # Method: lda_analysis_files
    # Description: creates topic model files
    # tweet texts, lda model visualization
    def lda_analysis_files(self, path, startDate_filter=None, endDate_filter=None, arr_edges=None, arr_ht_edges=None):            
        """
        Creates topic model files. Export a files with tweet texts and a lda model visualization.
        The data comes from the mongoDB database and is filtered based on the parameters.
                
        Parameters
        ----------                                       
        path :
            the path where the files should be saved
        
        startDate_filter : (Optional)
            filter by a certain start date
            
        endDate_filter : (Optional)
            filter by a certain end date
        
        arr_edges : (Optional)
            and array of concatenated edges that will be used to filter certain connection only.
            the method concat_edges can be used to create that array.
        
        arr_ht_edges : (Optional)
            and array of concatenated hashtag edges that will be used to filter certain ht connection only.
            the method concat_edges can be used to create that array. 
    
        Examples
        --------          
            Save lda analysis files into path:
        
            >>> lda_analysis_files(
            >>>     'D:\\Data\\MyFiles', 
            >>>     startDate_filter='09/20/2020 19:00:00', 
            >>>     endDate_filter='03/04/2021 00:00:00')
        """ 
        
        #export text for topic analysis
        print("****** Exporting text for topic analysis - " + self.get_now_dt())    
        self.exportData('text_for_topics', 
                        path + "\\" , 0, 
                        startDate_filter, 
                        endDate_filter, 
                        self.is_bot_Filter, 
                        arr_edges, 
                        arr_ht_edges=arr_ht_edges, 
                        replace_existing_file=False)

        # Train LDA models and print topics
        print("****** Topic discovery analysis (lda model) ****** - " + self.get_now_dt())        
        model_name = "Topics"
        topics_file_name = path + '\\T_tweetTextsForTopics.txt'
        if not os.path.exists(path + '\\Topics-(LDA model).png'):
            self.train_model_from_file(topics_file_name, self.num_of_topics, model_name, model_type='lda')
            self.plot_topics(path + '\\Topics-(LDA model).png', self.num_of_topics, 'lda', replace_existing_file=False)
        


    #####################################
    # Method: ht_analysis_files
    # Description: creates hashtag frequency files
    # frequency file text, wordcloud, and barcharts
    def ht_analysis_files(self, path, startDate_filter=None, endDate_filter=None, arr_edges=None, arr_ht_edges=None):        
        """
        Creates hashtag frequency files. Frequency text file, wordcloud, and barcharts.
        The data comes from the mongoDB database and is filtered based on the parameters.
                
        Parameters
        ----------                                       
        path :
            the path where the files should be saved
        
        startDate_filter : (Optional)
            filter by a certain start date
            
        endDate_filter : (Optional)
            filter by a certain end date
        
        arr_edges : (Optional)
            and array of concatenated edges that will be used to filter certain connection only.
            the method concat_edges can be used to create that array.
        
        arr_ht_edges : (Optional)
            and array of concatenated hashtag edges that will be used to filter certain ht connection only.
            the method concat_edges can be used to create that array.
    
        Examples
        --------          
            Save hashtag frequency files into path:
        
            >>> ht_analysis_files(
            >>>     'D:\\Data\\MyFiles', 
            >>>     startDate_filter='09/20/2020 19:00:00', 
            >>>     endDate_filter='03/04/2021 00:00:00')
        """
        
        #export ht frequency list         
        print("\n****** Exporting ht frequency list - " + self.get_now_dt())
        self.exportData('ht_frequency_list', 
                        path + "\\" , 0, 
                        startDate_filter, 
                        endDate_filter, 
                        self.is_bot_Filter, 
                        arr_edges, 
                        arr_ht_edges=arr_ht_edges, 
                        replace_existing_file=False)        
        
        print("****** Ploting HashTags Barchart and Wordcloud - " + self.get_now_dt())                   
        ht_file_name = path + '\\T_HT_FrequencyList.txt'
        
        if os.stat(ht_file_name).st_size != 0:                        
            df = self.read_freq_list_file(ht_file_name)            
            self.plot_top_freq_list(df, 30, 'HashTag', exclude_top_no=0, file=path + '\\T_HT_Top30_BarChart.png', replace_existing_file=False)
            self.plot_top_freq_list(df, 30, 'HashTag', exclude_top_no=1, file=path + '\\T_HT_Top30_BarChart-(Excluding Top1).png', replace_existing_file=False)
            self.plot_top_freq_list(df, 30, 'HashTag', exclude_top_no=2, file=path + '\\T_HT_Top30_BarChart-(Excluding Top2).png', replace_existing_file=False)
            self.plot_word_cloud(df, file=path + '\\T_HT_WordCloud.png', replace_existing_file=False)
        
        
    #####################################
    # Method: words_analysis_files
    # Description: creates words frequency files
    # frequency file text, wordcloud, and barcharts
    def words_analysis_files(self, path, startDate_filter=None, endDate_filter=None, arr_edges=None, arr_ht_edges=None):        
        """
        Creates words frequency files. Frequency text file, wordcloud, and barcharts.
        The data comes from the mongoDB database and is filtered based on the parameters.
                
        Parameters
        ----------                                       
        path :
            the path where the files should be saved
        
        startDate_filter : (Optional)
            filter by a certain start date
            
        endDate_filter : (Optional)
            filter by a certain end date
        
        arr_edges : (Optional)
            and array of concatenated edges that will be used to filter certain connection only.
            the method concat_edges can be used to create that array.
        
        arr_ht_edges : (Optional)
            and array of concatenated hashtag edges that will be used to filter certain ht connection only.
            the method concat_edges can be used to create that array.
    
        Examples
        --------          
            Save words frequency files into path:
        
            >>> words_analysis_files(
            >>>     'D:\\Data\\MyFiles', 
            >>>     startDate_filter='09/20/2020 19:00:00', 
            >>>     endDate_filter='03/04/2021 00:00:00')
            
        """                
        #export words frequency list 
        print("\n****** Exporting words frequency list - " + self.get_now_dt())        
        self.exportData('word_frequency_list', 
                        path + "\\" , 0, 
                        startDate_filter, 
                        endDate_filter, 
                        self.is_bot_Filter, 
                        arr_edges, 
                        arr_ht_edges, 
                        self.top_no_word_filter, 
                        replace_existing_file=False)
                        

        print("****** Ploting Word Barchart and Wordcloud - " + self.get_now_dt())                   
        word_file_name = path + '\\T_Words_FrequencyList.txt'
        if os.stat(word_file_name).st_size != 0:
            df = self.read_freq_list_file(word_file_name)
            self.plot_top_freq_list(df, 30, 'Word', exclude_top_no=0, file=path+'\\T_Words_Top30_BarChart.png', replace_existing_file=False)
            self.plot_word_cloud(df, file=path+'\\T_Words_WordCloud.png', replace_existing_file=False)
        

    #####################################
    # Method: time_series_files
    # Description: creates time frequency files    
    def time_series_files(self, path, startDate_filter=None, endDate_filter=None, arr_edges=None, arr_ht_edges=None):        
        """
        Creates timeseries frequency files. Tweet count by day and hashcount count by day.
        The data comes from the mongoDB database and is filtered based on the parameters.
                
        Parameters
        ----------                                       
        path :
            the path where the files should be saved
        
        startDate_filter : (Optional)
            filter by a certain start date
            
        endDate_filter : (Optional)
            filter by a certain end date
        
        arr_edges : (Optional)
            and array of concatenated edges that will be used to filter certain connection only.
            the method concat_edges can be used to create that array.
        
        arr_ht_edges : (Optional)
            and array of concatenated hashtag edges that will be used to filter certain ht connection only.
            the method concat_edges can be used to create that array.
    
        Examples
        --------          
            Save timeseries frequency files into path:
        
            >>> time_series_files(
            >>>    'D:\\Data\\MyFiles', 
            >>>    startDate_filter='09/20/2020 19:00:00', 
            >>>    endDate_filter='03/04/2021 00:00:00')
        """  
        
                          
        print("****** Exporting time series files - " + self.get_now_dt())           
        tweet_df = self.get_time_series_df(startDate_filter=startDate_filter, endDate_filter=endDate_filter, arr_edges=arr_edges, arr_ht_edges=arr_ht_edges)
        
        #plot time series for all tweets
        if not os.path.exists(path + '\\TS_TweetCount.png'):
            self.plot_timeseries(tweet_df, ['tweet', 'tweet_created_at'], path + '\\TS_TweetCount.png')   
        
        #plot time series for top hashtags [1-5]
        if not os.path.exists(path + '\\TS_TweetCountByHT[1-5].png'):
            self.plot_top_ht_timeseries(top_no_start=1, top_no_end=5, file = path + '\\TS_TweetCountByHT[1-5].png', 
                                        startDate_filter=startDate_filter, endDate_filter=endDate_filter, arr_edges=arr_edges, arr_ht_edges=arr_ht_edges)
            
        #plot time series for top hashtags [3-10]
        if not os.path.exists(path + '\\TS_TweetCountByHT[3-10].png'):
            self.plot_top_ht_timeseries(top_no_start=3, top_no_end=10, file = path + '\\TS_TweetCountByHT[3-10].png', 
                                        startDate_filter=startDate_filter, endDate_filter=endDate_filter, arr_edges=arr_edges, arr_ht_edges=arr_ht_edges)
    
        
    #####################################
    # Method: ht_connection_files
    # Description: creates hashags graph connections files
    def ht_connection_files(self, path, startDate_filter=None, endDate_filter=None, arr_edges=None):
                                                                       
        print("****** Exporting ht connection files - " + self.get_now_dt())        
    
        #create file with ht edges
        self.exportData('ht_edges', path + "\\" , 0, startDate_filter, endDate_filter, self.is_bot_Filter, arr_edges)                
        edge_file_path = path + "\\ht_edges.txt" 
        G = self.loadGraphFromFile(edge_file_path)
        if len(G.edges) > 0:
            if len(G.edges) > 1000:
                G = self.largest_component_no_self_loops(G)
            else:
                G.remove_edges_from(nx.selfloop_edges(G))    
                for node in list(nx.isolates(G)):
                    G.remove_node(node)
            print("HT graph # of Nodes " + str(len(G.nodes())))
            print("HT graph # of Edges " + str(len(G.edges())))
            self.graph_analysis_files(G, path, gr_prefix_nm = 'HTG_') 


        #remove top hashtags if we want to ignore the top hashtags
        if self.top_ht_to_ignore is not None:
            G2 = G.copy()
            remove_name = '[WITHOUT('
            arr_nodes = sorted(G2.degree(), key=lambda x: x[1], reverse=True)
            for ht, degree in arr_nodes[0:self.top_ht_to_ignore]:            
                remove_name = remove_name + '-' + ht
                G2.remove_node(ht)
            remove_name = remove_name + ')]'

            if len(G2.edges) > 0:
                if len(G2.edges) > 1000:
                    G2 = self.largest_component_no_self_loops(G2)
                else:
                    G2.remove_edges_from(nx.selfloop_edges(G2))    
                    for node in list(nx.isolates(G2)):
                        G2.remove_node(node)  
                print("HT graph # of Nodes " + str(len(G2.nodes())))
                print("HT graph # of Edges " + str(len(G2.edges())))
                self.graph_analysis_files(G2, path, gr_prefix_nm = 'HTG_' + remove_name + '_')                                                                


    #####################################
    # Method: graph_analysis_files
    # Description: creates graphs files
    def graph_analysis_files(self, G, path, gr_prefix_nm=''):
        """
        Plot graph analysis files for a given graph G. 
        It uses the configuration set on the setConfigs method.
                
        Parameters
        ----------               
        G : 
            undirected networkx graph created from the Twitter data                               
        path :
            the path where the files should be saved
        
        gr_prefix_nm: (Optional)
            a prefix to add to the graph name. (Default='')
            
        Examples
        --------          
        Create graph visualization files
        
            >>> graph_analysis_files(G, 'C:\\Data\\MyAnalysis\\', 'MyNameTest')
        """      
        
        if len(G.nodes()) > 0 and len(G.edges()) > 0:
        
            #plot graph
            print("\n****** Ploting graphs... *********** - " + self.get_now_dt())                

            # if not os.path.exists(path + '\\' + gr_prefix_nm + 'G_Graph.png') 
            # and not os.path.exists(path + '\\' + gr_prefix_nm + 'G_Graph(WithoutScale).png'):                        
            if ((len(G.nodes()) <= self.graph_plot_cutoff_no_nodes \
                 or len(G.edges()) <= self.graph_plot_cutoff_no_edges) \
                 and len(G.edges()) != 0) \
                 or len(G.nodes()) <= MIN_NO_OF_NODES_TO_REDUCE_GRAPH:

                if len(G.edges()) < 450:
                    v_scale = 0.01; v_k =0.7; v_iterations=100; v_node_size=2
                elif len(G.edges()) < 5000:
                    v_scale = 2; v_k = 0.6; v_iterations=200; v_node_size=0.8
                elif len(G.edges()) < 10000:
                    v_scale = 1; v_k = 0.1; v_iterations=200; v_node_size=0.6
                elif len(G.edges()) >= 10000:
                    v_scale = 1; v_k = 0.05; v_iterations=500; v_node_size=0.6

                if self.create_graph_with_node_scale_flag == 'Y':
                    G_to_plot, labels2, k = self.calculate_louvain_clustering(G)
                    self.plotSpringLayoutGraph(G_to_plot, 
                                               path + '\\' + gr_prefix_nm + 'G_Graph.png', 
                                               v_scale, 
                                               v_k, 
                                               v_iterations, 
                                               cluster_fl='Y', 
                                               v_labels=list(list(labels2)), 
                                               replace_existing_file=False)

                if self.create_graph_without_node_scale_flag == 'Y':
                    self.plotSpringLayoutGraph(G, 
                                               path + '\\' + gr_prefix_nm + 'G_Graph(WithoutScale).png',
                                               v_scale, 
                                               v_k, 
                                               v_iterations, 
                                               cluster_fl='N', 
                                               v_alpha=1, 
                                               scale_node_size_fl='N', 
                                               replace_existing_file=False)


            #plot reduced graph
            if self.create_reduced_graph_flag == 'Y':
                self.plot_graph_contracted_nodes(G, path + '\\' + gr_prefix_nm + 'G_Graph-(ReducedGraph).png')
            print("\n")
                        
            
    #####################################
    # Method: edge_files_analysis
    # Description: load graph from edge files and call methods to create all analysis 
    # files for the main graph and for the graph of each period
    def edge_files_analysis(self, output_path): 
        """
        Automated way to generate all analysis files. 
        It creates all folders, edge files, and any other files based on given settings. 
        The setting of what files are interesting or not, should be set using the setConfigs method.
        
        Parameters
        ----------                                       
        output_path :
            the path where the files should be saved        
    
        Examples
        --------          
            Create all analysis files and folder based on the configurations set on setConfigs:
        
            >>> edge_files_analysis('D:\\Data\\MyFiles') 
        """      
                
        case_ht_str = ''
        if self.type_of_graph == 'ht_conn':
            case_ht_str = 'ht_'            

        #Get the right edges file to import
        if self.is_bot_Filter is None:
            parent_path = output_path + '\\' + self.edge_prefix_str + 'All'
            edge_file_path = self.folder_path + '\\data_input_files\\' + self.edge_prefix_str + 'AllPeriods_' + case_ht_str + 'edges.txt'
            if not os.path.exists(edge_file_path): self.export_all_edges_for_input(period_arr=self.period_arr, type_of_graph=self.type_of_graph)
        elif self.is_bot_Filter == '0':
            parent_path = output_path + '\\' + self.edge_prefix_str + 'ExcludingBots'
            edge_file_path = self.folder_path + '\\data_input_files\\' + self.edge_prefix_str +'AllPeriods_ExcludingBots_' + case_ht_str + 'edges.txt'            
            if not os.path.exists(edge_file_path): self.export_all_edges_for_input(period_arr=self.period_arr, bot_filter_fl='Y', type_of_graph=self.type_of_graph)
        elif self.is_bot_Filter == '1':
            parent_path = output_path + '\\' + self.edge_prefix_str + 'Bots_Edges_Only'
            edge_file_path = self.folder_path + '\\data_input_files\\' + self.edge_prefix_str + 'AllPeriods_BotsOnly_' + case_ht_str + 'edges.txt'            
            if not os.path.exists(edge_file_path): self.export_all_edges_for_input(period_arr=self.period_arr, bot_filter_fl='Y', type_of_graph=self.type_of_graph)
            
        print(edge_file_path)
        self.create_path(output_path)
            
        # Load graph from edge file
        G = self.loadGraphFromFile(edge_file_path)
                
        # Call method to print all analysis files
        self.all_analysis_file(G, parent_path, startDate_filter=None, endDate_filter=None)                
                                       
          
        # Run analysis by period using the dates set on array period_arr
        if self.period_arr is not None:                                    
            
            # Creates a text file with the period information. 
            # This is just so that whoever is looking at these folder can know what dates we used for each period
            myFile = open(output_path + '\\PeriodsInfo.txt', 'w', encoding="utf-8")
            with myFile:
                writer = csv.writer(myFile, delimiter='\t', lineterminator='\n')
                writer.writerows(self.period_arr)
                    
            for idx, period in enumerate(self.period_arr):                    

                # Set the period information variables
                period_name = period[0]
                period_start_date = period[1]
                period_end_date = period[2]

                print("\n**********************************************************")
                print("************************** " + period_name + " ****************************\n" ) 

                # Edge file path 
                if self.is_bot_Filter is None:
                    parent_path = output_path + "\\" + self.edge_prefix_str + "All_By_Period\\" + period_name
                    edge_file_path = output_path + "\\data_input_files\\" + self.edge_prefix_str + period_name +"_" + case_ht_str + "edges.txt"                        
                    if not os.path.exists(edge_file_path): self.export_all_edges_for_input(period_arr=self.period_arr, type_of_graph=self.type_of_graph)
                elif self.is_bot_Filter  == '0':
                    parent_path = output_path + "\\" + self.edge_prefix_str + "Excluding_Bots_By_Period\\" + period_name
                    edge_file_path = output_path + "\\data_input_files\\" + self.edge_prefix_str + period_name + "_ExcludingBots_" + case_ht_str + "edges.txt"                    
                    if not os.path.exists(edge_file_path): self.export_all_edges_for_input(period_arr=self.period_arr, bot_filter_fl='Y', type_of_graph=self.type_of_graph)
                elif self.is_bot_Filter  == '1':
                    parent_path = output_path + "\\" + self.edge_prefix_str + "Bots_Edges_Only_By_Period\\" + period_name                    
                    edge_file_path = output_path + "\\data_input_files\\" + self.edge_prefix_str + period_name +"_BotsOnly_" + case_ht_str + "edges.txt"                    
                    if not os.path.exists(edge_file_path): self.export_all_edges_for_input(period_arr=self.period_arr, bot_filter_fl='Y', type_of_graph=self.type_of_graph)

                # Create new path if it doesn't exist
                self.create_path(parent_path)

                #load graph from edge file
                G = self.loadGraphFromFile(edge_file_path)                
                
                #call function to genrate all files for this graph
                self.all_analysis_file(G, parent_path, startDate_filter=period_start_date, endDate_filter=period_end_date)
                                        
        

    #####################################
    # Method: all_analysis_file
    # Description: Calls method to create all files for full dataset, for top degree nodes, and for community graphs
    def all_analysis_file(self, G, output_path, startDate_filter=None, endDate_filter=None):
                                
        #files for the main graph
        self.create_analysis_file(G, output_path, startDate_filter=startDate_filter, endDate_filter=endDate_filter)
                                        
        #files for the top nodes
        if self.create_top_nodes_files_flag == 'Y':            
            self.top_nodes_analysis(G, output_path, startDate_filter=startDate_filter, endDate_filter=endDate_filter)        
        
        #files for community nodes
        if self.create_community_files_flag == 'Y':
            self.commty_analysis_files(G, output_path, startDate_filter=startDate_filter,  endDate_filter=endDate_filter)                                

            
            
    #####################################
    # Method: create_analysis_file
    # Description: calls individual methods to create files on the settings             
    def create_analysis_file(
            self, 
            G, 
            output_path, 
            startDate_filter=None, 
            endDate_filter=None, 
            arr_edges=None):
                                
        #export file with measures
        print("****** Graph Measures - " + self.get_now_dt())
        self.print_Measures(G, fileName_to_print = output_path + "\\G_Measures-(All).txt")
        print("\n")
        
        arr_ht_edges = None
        if self.type_of_graph == 'ht_conn':
            arr_ht_edges = arr_edges
            arr_edges = None            

        if len(G.edges()) != 0:
            #get largest connected component and export file with measures
            G = self.largest_component_no_self_loops(G)
            print("****** Largest Component Graph Measures - " + self.get_now_dt())
            self.print_Measures(G, fileName_to_print = output_path + "\\G_Measures-(LargestCC).txt")
            print("\n")

            #export files with edges and degrees
            if self.create_nodes_edges_files_flag == 'Y':
                self.nodes_edges_analysis_files(G, output_path)
            
            #LDA Model
            if self.create_topic_model_files_flag == 'Y':
                self.lda_analysis_files(output_path, 
                                        startDate_filter=startDate_filter, 
                                        endDate_filter=endDate_filter, 
                                        arr_edges=arr_edges, 
                                        arr_ht_edges=arr_ht_edges)

            #export ht frequency list 
            if self.create_ht_frequency_files_flag == 'Y':           
                self.ht_analysis_files(output_path, 
                                       startDate_filter=startDate_filter, 
                                       endDate_filter=endDate_filter, 
                                       arr_edges=arr_edges, 
                                       arr_ht_edges=arr_ht_edges)
                
            #export words frequency list 
            if self.create_words_frequency_files_flag == 'Y':
                self.words_analysis_files(output_path, 
                                          startDate_filter=startDate_filter, 
                                          endDate_filter=endDate_filter, 
                                          arr_edges=arr_edges, 
                                          arr_ht_edges=arr_ht_edges)
                                 
            #plot graph
            if self.create_graphs_files_flag == 'Y':
                self.graph_analysis_files(G, output_path)
                
            #time series
            if self.create_timeseries_files_flag == 'Y':
                self.time_series_files(output_path, 
                                       startDate_filter=startDate_filter, 
                                       endDate_filter=endDate_filter, 
                                       arr_edges=arr_edges, 
                                       arr_ht_edges=arr_ht_edges) 

            #hashtag connections
            if self.create_ht_conn_files_flag == 'Y' and self.type_of_graph != 'ht_conn':
                self.ht_connection_files(output_path, 
                                         startDate_filter=startDate_filter, 
                                         endDate_filter=endDate_filter, 
                                         arr_edges=arr_edges)
            
                           
        
        
    #####################################
    # Method: top_nodes_analysis
    # Description: calls methods to create files for each of the top degree nodes
    def top_nodes_analysis(self, G, output_path, startDate_filter=None, endDate_filter=None):                                      

        # Choose which graph you want to run this for
        Graph_to_analyze = G.copy()
        
        top_degree_nodes = self.get_top_degree_nodes(Graph_to_analyze, self.top_degree_start, self.top_degree_end)

        #creates a folder to save the files for this analysis
        path = "Top_" + str(self.top_degree_start) + '-' + str(self.top_degree_end) 
        self.create_path(output_path + '\\' + path)

        i = self.top_degree_end
        # loops through the top degree nodes, creates a subgraph for them and saves the results in a folder
        for x in np.flip(top_degree_nodes, 0):
            node = x[0]

            #creates a subgraph for this node
            G_subgraph = self.create_node_subgraph(Graph_to_analyze, node)    
            G_subgraph_largestComponent = G_subgraph.copy()
            G_subgraph_largestComponent = self.largest_component_no_self_loops(G_subgraph_largestComponent)

            #creates a path to add the files for this node
            path_node = path + "\\" + str(i) + "-" + node
            self.create_path(output_path + '\\' + path_node)
            
            #get array with all edges for this top degree node                          
            if len(G_subgraph) > 1:
                arr_edges = self.concat_edges(G_subgraph)
                self.create_analysis_file(G_subgraph, 
                                          output_path + '\\' + path_node,
                                          startDate_filter=startDate_filter, 
                                          endDate_filter=endDate_filter, 
                                          arr_edges=arr_edges)
    
            i -= 1

            
    #####################################
    # Method: commty_analysis_files
    # Description: calls methods to create files for each of the communities found 
    def commty_analysis_files(self, G, output_path, startDate_filter=None, endDate_filter=None):
        
        print("\n******************************************************")
        print("******** Louvain Communities ********")
                            
        if len(G.edges()) != 0:                                                
            
            # Choose which graph you want to run this for
            Graph_to_analyze = G.copy()            
            
            #creates a folder to save the files for this analysis
            path = output_path + "\\Communities_(Louvain)"
            while os.path.exists(path):
                path = path + "+"                        
            self.create_path(path)                        

            #calculate louvain community for largest connected component
            Graph_to_analyze = self.largest_component_no_self_loops(Graph_to_analyze) 
            Graph_to_analyze, labels, k = self.calculate_louvain_clustering(Graph_to_analyze)            
            
            comm_att = 'community_louvain'

            #find the number of communities in the graph
            no_of_comm = max(nx.get_node_attributes(Graph_to_analyze, comm_att).values())+1    

            print("******************************************************")
            print("Total # of Communities " + str(no_of_comm))

            #loop through the communities print they analysis files
            for commty in range(no_of_comm):

                #find subgraphs of this community
                G_subgraph = Graph_to_analyze.subgraph([n for n,attrdict in Graph_to_analyze.node.items() if attrdict [comm_att] == commty ])                                                
                #only cares about communities with more than 1 node
                if len(G_subgraph.edges()) >= self.commty_edge_size_cutoff:

                    G_subgraph_largestComponent = G_subgraph.copy()
                    G_subgraph_largestComponent = self.largest_component_no_self_loops(G_subgraph_largestComponent)

                    #creates a path to add the files for this node
                    path_community = path + "\\Community-" + str(commty+1)
                    self.create_path(path_community)

                    print("\n")
                    print("******************************************************")
                    print("****** Printing files for community " + str(commty+1) + " ******")
                    #self.print_Measures(G_subgraph, False, False, False, False, fileName_to_print = path_community + '\\G_' + str(commty+1) + '_Measures.txt')    
                    print("\n")
                    
                    if len(G_subgraph) > 1:
                        arr_edges = self.concat_edges(G_subgraph)
                        self.create_analysis_file(G_subgraph, path_community,                                               
                                                  startDate_filter=startDate_filter, 
                                                  endDate_filter=endDate_filter,                                               
                                                  arr_edges=arr_edges)
                
                        
    #####################################
    # Method: get_time_series_df
    # Description: query data in mongoDB for timeseries analysis            
    def get_time_series_df(
            self, 
            ht_arr=None, 
            startDate_filter=None, 
            endDate_filter=None, 
            arr_edges=None, 
            arr_ht_edges=None):
        """  
        Method to query data in mongoDB for timeseries analysis given certain filters.
        It creates all folders, edge files, and any other files based on given settings. 
        The setting of what files are interesting or not, should be set using the setConfigs method.
        
        Parameters
        ----------                                       
        ht_arr :
            array of hashtags to filter the data from
        
        startDate_filter : (Optional)
            filter by a certain start date
            
        endDate_filter : (Optional)
            filter by a certain end date
        
        arr_edges : (Optional)
            and array of concatenated edges that will be used to filter certain connection only.
            the method concat_edges can be used to create that array.
        
        arr_ht_edges : (Optional)
            and array of concatenated hashtag edges that will be used to filter certain ht connection only.
            the method concat_edges can be used to create that array.
    
        Examples
        --------        
            >>> ...
        """   
        
        df = pd.DataFrame()

        if ht_arr is not None:        
            #get timeseries for each of the top hashtags    
            for i, ht in enumerate(ht_arr):
                arrData, file = self.queryData(exportType='tweet_ids_timeseries', 
                                               filepath='', 
                                               inc=0, 
                                               ht_to_filter=ht, 
                                               startDate_filter=startDate_filter, 
                                               endDate_filter=endDate_filter, 
                                               is_bot_Filter=self.is_bot_Filter, 
                                               arr_edges=arr_edges,
                                               arr_ht_edges=arr_ht_edges)
                tweet_df = pd.DataFrame(list(arrData))
                tweet_df.columns = ['tweet_created_at', ht]   
                df = pd.concat([df,tweet_df], axis=0, ignore_index=True)


        else:
            #get timeseries for all tweets
            arrData, file = self.queryData(exportType='tweet_ids_timeseries', 
                                           filepath='', inc=0,                                                     
                                           startDate_filter=startDate_filter, 
                                           endDate_filter=endDate_filter, 
                                           is_bot_Filter=self.is_bot_Filter, 
                                           arr_edges=arr_edges,
                                           arr_ht_edges=arr_ht_edges)    
            tweet_df = pd.DataFrame(list(arrData))
            tweet_df.columns = ['tweet_created_at', 'tweet']   
            df = pd.concat([df,tweet_df], axis=0, ignore_index=True)


        return df


    #####################################
    # Method: plot_top_ht_timeseries
    # Description: get top hashtags and plot their timeseries data
    def plot_top_ht_timeseries(
            self, 
            top_no_start, 
            top_no_end, 
            file, 
            startDate_filter=None, 
            endDate_filter=None, 
            arr_edges=None, 
            arr_ht_edges=None):        
        
        #get the top hashtags to plot
        ht_arr, f = self.queryData(exportType='ht_frequency_list', 
                                   filepath='', inc=0, 
                                   startDate_filter=startDate_filter, 
                                   endDate_filter= endDate_filter, 
                                   is_bot_Filter=self.is_bot_Filter, 
                                   arr_edges=arr_edges, 
                                   arr_ht_edges=arr_ht_edges,
                                   top_no_filter=top_no_end, 
                                   include_hashsymb_FL=False)
                        
        if len(ht_arr) < top_no_end:
            top_no_end = len(ht_arr)
                        
        if len(ht_arr) == 0 or top_no_start >= top_no_end:
            return ""
        
        ht_arr = np.array(ht_arr)
        ht_arr = ht_arr[top_no_start-1:top_no_end,0]
        ht_arr = list(ht_arr)
        
        #get the time series data
        df = self.get_time_series_df(ht_arr=ht_arr, 
                                     startDate_filter=startDate_filter, 
                                     endDate_filter=endDate_filter, 
                                     arr_edges=arr_edges)

        #plot timeseries graph
        arr_columns = ht_arr.copy()
        arr_columns.append('tweet_created_at')        
        self.plot_timeseries(df, arr_columns, file)


        
    #####################################
    # Method: plot_timeseries
    # Description: plot time series data
    def plot_timeseries(self, df, arr_columns, file):                      

        tweet_df = (df[arr_columns]
         .set_index('tweet_created_at')      
         .resample('D') 
         .count()         
        );

        ax = tweet_df.plot(figsize=(25,8))
        ax.set_xlabel("Date")
        ax.set_ylabel("Tweet Count")
        
        
        plt.savefig(file, dpi=200, facecolor='w', edgecolor='w')
        #plt.show()
        plt.cla()   # Clear axis
        plt.clf()   # Clear figure
        plt.close() # Close a figure window
    
    
       
    #####################################
    # Method: eda_analysis
    # Description: Save EDA files
    def eda_analysis(self):
        """  
        Method to print a summary of the initial exploratory data analysis for any dataset.        

        Examples
        --------          
            >>> # Create Exploratory Data Analysis files
            >>> eda_analysis()
            
        It includes the following metrics:
        
        + **Tweet counts**: The number of tweet document in the database, divided by the following categories.
            
                + Total Original Tweets: The number of tweet documents in the database that are original tweets.		
                + Total Replies: The number of tweet documents in the database that are replies to another tweet.		
                + Total Retweets: The number of tweet documents in the database that are retweets.		
                + Total Tweets: The total number of tweet documents in the database.
            
        + **Tweet counts by language**: The number of tweets document for each language used in the tweets.

        + **Tweet counts by month**: The number of tweets document for each month/year.

        + **Tweet counts by file**: The number of tweets document imported from each of the json files.

        + **User counts**: The number of users in the database, divided by the following categories.
            
                + tweet: Users with at least one document in the database.
                + retweet: Users that were retweeted, but are not part of previous group.
                + quote: Users that were quoted, but are not part of previous groups.
                + reply: Users that were replied to, but are not part of previous groups.
                + mention: Users that were mentioned, but are not part of previous groups.
                        
        + **All User Connections Graph**: The metrics for the graph created based on the users connecting by retweets, quotes, mentions, and replies.
            
                + # of Nodes: The total number of nodes in the graph.
                + # of Edges: The total number of edges in the graph.
                + # of Nodes of the largest connected components: The total number of nodes in the largest connected component of the graph.
                + # of Edges of the largest connected components: The total number of edges in the largest connected component of the graph.                
                + # of Disconnected Graphs: The number of sub-graphs within the main graph that are not connected to each other.                
                + # of Louvain Communities found in the largest connected component: The number of communities found in the largest connected component using the Louvain method.                
                + Degree of the top 5 most connected users: List of the top 5 users with the highest degree. Shows the user screen name and respective degrees.                
                + Average Node Degree of largest connected graph: The average degree of all nodes that are part of the largest connected component of the graph.                
                + Plot of the Louvain community distribution: A barchart showing the node count distribution of the communities found with the Louvain method.
                + Disconnected graphs distribution: A plot of a graph showing the distribution of the disconnected graphs. It shows the total number of nodes and edges for each of the disconnected graphs. 
                
        + **Mentions User Connections Graph**: The same metrics as the *All User Connections* graph, but only considering the connections made by mentions.

        + **Retweets User Connections Graph**: The same metrics as the *All User Connections* graph, but only considering the connections made by retweets.

        + **Replies User Connections Graph**: The same metrics as the *All User Connections* graph, but only considering the connections made by replies.

        + **HT Connection Graph**: The same metrics as the *All User Connections* graph, but only considering the connections made by hashtags.           

        """                 
        eda_folder =  self.folder_path  + '\\EDA'
        self.create_path(eda_folder)                    
        eda_file = open(eda_folder + '\\EDA.txt', 'w', encoding="utf-8")
                    
            
        print("**** Tweet counts ******")
        eda_file.write("**** Tweet counts ******\n")
        arr, f = self.queryData(exportType='tweetCount', filepath='', inc=0)
        for x in arr:           
            eda_file.write(str(x))
            eda_file.write("\n")
        df = pd.DataFrame(arr)
        df.columns = ['', '']
        print(df.to_string())
        print("\n")


        print("**** Tweet counts by language ******")           
        eda_file.write("\n**** Tweet counts by language ******\n")
        arr, f = self.queryData(exportType='tweetCountByLanguage', filepath='', inc=0)
        for x in arr:
            eda_file.write(str(x))
            eda_file.write("\n")
        df = pd.DataFrame(arr)
        df.columns = ['', '']
        print(df.to_string())
        print("\n")
        
        
        print("**** Tweet counts by month ******")    
        eda_file.write("\n**** Tweet counts by month ******\n")
        arr, f = self.queryData(exportType='tweetCountByMonth', filepath='', inc=0)
        for x in arr:
            eda_file.write(str(x))
            eda_file.write("\n")
        df = pd.DataFrame(arr)
        df.columns = ['', '', '']
        print(df.to_string())
        print("\n")   
        
        
        print("**** Tweet counts by file ******")    
        eda_file.write("\n**** Tweet counts by file ******\n")
        arr, f = self.queryData(exportType='tweetCountByFile', filepath='', inc=0)
        for x in arr:
            eda_file.write(str(x))
            eda_file.write("\n")
        df = pd.DataFrame(arr)
        df.columns = ['', ''] 
        print(df.to_string())
        print("\n")                  
 

        print("**** User counts ******")    
        eda_file.write("\n**** User counts ******\n")
        arr, f = self.queryData(exportType='userCount', filepath='', inc=0)
        arr.sort()
        for x in arr:
            eda_file.write(str(x))
            eda_file.write("\n")
        df = pd.DataFrame(arr)
        df.columns = ['', '', '', '']
        print(df.to_string())
        print("\n")
                    
        

            
        # Graph EDA
                        
        # Load graph from main edges file if it does not exist 
        edge_file_path = self.folder_path + '\\data_input_files\\UserConnections_AllPeriods_edges.txt'
        if not os.path.exists(edge_file_path):
            self.export_all_edges_for_input(type_of_graph = 'user_conn_all')
                    
        # types of graph
        arr_type_pre = [['UserConnections_', 'edges'], 
                        ['MentionUserConnections_','edges'], 
                        ['RetweetUserConnections_','edges'], 
                        ['ReplyUserConnections_','edges'], 
                        ['QuoteUserConnections_','edges'], 
                        ['HTConnection_', 'ht_edges']] 
        
        # Loop through the type of graphs
        for i in range(len(arr_type_pre)):                        
            
            # find the edge file name
            edge_file_path = self.folder_path + '\\data_input_files\\' + arr_type_pre[i][0] + 'AllPeriods_' + arr_type_pre[i][1] + '.txt'            
            
            # if the edge file already exists
            if os.path.exists(edge_file_path):
                
                print('\n\n*****************************************') 
                print('**** ' + arr_type_pre[i][0] + ' Graph ******') 
                
                # Construct the graph based on the edge file
                G = self.loadGraphFromFile(edge_file_path)   
                
                # if the graph is not empty
                if len(G.nodes()) > 0 and len(G.edges()) > 0:
                    # Plot distribution of the separate connected components
                    print("**** Connected Components - Distribution ******") 
                    no_of_disc_g = self.plot_disconnected_graph_distr(G, file=eda_folder + '\\' + arr_type_pre[i][0] + 'ConnectedComponents-(Graphs).png')
                    no_of_disc_g_gt50 = self.plot_disconnected_graph_distr(G, size_cutoff=50)
                    

                    #calculate louvein community clustering
                    print("**** Calculating Community Distribution of the Largest Connected Component- (Louvain) ******") 
                    G2 = self.largest_component_no_self_loops(G) 
                    G2, labels, k = self.calculate_louvain_clustering(G2)
                    self.plot_graph_att_distr(G2, 
                                              'community_louvain',
                                              title='Louvain Community Distribution for Largest Connected Component',
                                              file_name=eda_folder+'\\' + arr_type_pre[i][0] + 'community_louvain_dist.png')        

                    # Degree arrays
                    arr = np.array(sorted(G2.degree(), key=lambda x: x[1], reverse=True))                
                    #deg_mean = np.asarray(arr[:,1], dtype=np.integer).mean()
                    # get the mean node degree of the nodes
                    deg_mean = self.calculate_average_node_degree(G2)

                    print(" # of Nodes " + str(len(G.nodes())))
                    print(" # of Edges " + str(len(G.edges())))
                    print(" # of Nodes - (Largest Connected Component) " + str(len(G2.nodes())))
                    print(" # of Edges - (Largest Connected Component) " + str(len(G2.edges())))
                    print(" # of Disconnected Graphs " + str(no_of_disc_g))
                    print(" # of Disconnected Graphs with 50 or more nodes " + str(no_of_disc_g_gt50))                    
                    print(" # of Communities found in the largest connected component " + str(k))
                    if len(arr) > 1:
                        print(" Degree of top 1 most connected user " + str(arr[0]))
                    if len(arr) > 2:
                        print(" Degree of top 2 most connected user " + str(arr[1]))
                    if len(arr) > 3:
                        print(" Degree of top 3 most connected user " + str(arr[2]))
                    if len(arr) > 4:
                        print(" Degree of top 4 most connected user " + str(arr[3]))
                    if len(arr) > 5:
                        print(" Degree of top 5 most connected user " + str(arr[4]))
                    print(" Average Node Degree of largest connected graph " + str(deg_mean))
                    eda_file.write("\n")
                    eda_file.write('**** ' + arr_type_pre[i][0] + ' Graph ******') 
                    eda_file.write("\n")
                    eda_file.write("# of Nodes " + str(len(G.nodes())))
                    eda_file.write("\n")
                    eda_file.write("# of Edges " + str(len(G.edges())))
                    eda_file.write("\n")
                    eda_file.write("# of Disconnected Graphs " + str(no_of_disc_g))
                    eda_file.write("\n")
                    eda_file.write("# of Louvain Communities found in the largest connected component " + str(k))
                    eda_file.write("\n")
                    if len(arr) > 1:
                        eda_file.write("Degree of top 1 most connected user " + str(arr[0]))
                        eda_file.write("\n")
                    if len(arr) > 2:
                        eda_file.write("Degree of top 2 most connected user " + str(arr[1]))
                        eda_file.write("\n")
                    if len(arr) > 3:
                        eda_file.write("Degree of top 3 most connected user " + str(arr[2]))
                        eda_file.write("\n")
                    if len(arr) > 4:
                        eda_file.write("Degree of top 4 most connected user " + str(arr[3]))
                        eda_file.write("\n")
                    if len(arr) > 5:
                        eda_file.write("Degree of top 5 most connected user " + str(arr[4]))
                        eda_file.write("\n")
                    eda_file.write("\n")
                    eda_file.write("Average Node Degree of largest connected graph " + str(deg_mean))
                    eda_file.write("\n")
        
        
        #close file
        eda_file.close()
                
        print("*** EDA - END *** - " + self.get_now_dt())
        
        
        
    #####################################
    # Method: print_top_nodes_cluster_metrics
    # Description: calculate clustering metrics for top degree nodes
    def print_top_nodes_cluster_metrics(self, G, top_degree_end, acc_node_size_cutoff=None):                                      
        """
        Calculates clustering metrics for top degree nodes        
                
        Parameters
        ----------               
        G : 
            undirected networkx graph created from the Twitter data                               
        
        top_degree_end :
            the number of top nodes to use for calculation
        
        acc_node_size_cutoff : (Optional)
            The average clustering coefficient metric can take a long time to run, 
            so users can set a cutoff number in this parameter for the graph size 
            that will decide if that metric will be printed or not depending on the graph size. (Default=None)
              
            
        Examples
        --------          
        Create graph visualization files
        
            >>> print_cluster_metrics(
            >>> G_Community, 
            >>> G, 
            >>> top_no=3, 
            >>> acc_node_size_cutoff=None
            >>> )
        """  
        
        exec_tm = 0
        endtime = 0
        starttime = 0
        
        starttime = time.time()
        
        top_degree_nodes = self.get_top_degree_nodes(G, 1, top_degree_end)        

        i = 1
        # loops through the top degree nodes, creates a subgraph for them
        for x in top_degree_nodes:
            
            print("***** Cluster for top " + str(i) + " node")
            
            node = x[0]            
            
            #creates a subgraph for this node
            G_subgraph = self.create_node_subgraph(G, node)    
            
            starttime_met = time.time()
            # print metrics
            self.print_cluster_metrics(G_subgraph, G, top_no=3, acc_node_size_cutoff=acc_node_size_cutoff)
            endtime_met = time.time()
            exec_tm = exec_tm + (endtime_met - starttime_met)
            
            print("\n")

            i += 1
            
        endtime = time.time()
        #exec_tm_total = endtime - starttime        
        print("Execution Time:  %s seconds " % (endtime - starttime - exec_tm))
            

    #####################################
    # Method: print_commty_cluster_metrics
    # Description: calls methods to create files for each of the communities found 
    def print_commty_cluster_metrics(self, G, comm_att='community_louvain', ignore_cmmty_lt=0, acc_node_size_cutoff=None):
        """
        Calculates clustering metrics for top degree nodes        
                
        Parameters
        ----------               
        G : 
            undirected networkx graph created from the Twitter data                               
        
        comm_att : (Optional)
            Possible values: 'community_louvain' or 'spectral_clustering'. (Default='community_louvain')
        
        ignore_cmmty_lt : (Optional)
             Number used to ignore small communitites. 
             The logic will not calculate metrics for communities smaller than this number. (Default=0)

        acc_node_size_cutoff : (Optional)
            The average clustering coefficient metric can take a long time to run, 
            so users can set a cutoff number in this parameter for the graph size 
            that will decide if that metric will be printed or not depending on the graph size. (Default=None)
                         
        Examples
        --------                              
            >>> print_commty_cluster_metrics(G, 'community_louvain', '10')
        """  
        
        if len(G.edges()) != 0:

            # find the number of communities in the graph
            no_of_comm = max(nx.get_node_attributes(G, comm_att).values())+1
            print("Total # of Communities " + str(no_of_comm))            

            print("******************************************************")            
            print("*****" + comm_att + "******")
            print("\n")

            # loop through the communities print they analysis files
            no_of_comm_gt_cutoff = 0
            for commty in range(no_of_comm):
                                
                # find subgraphs of this community
                G_subgraph = G.subgraph([n for n,attrdict in G.node.items() if attrdict [comm_att] == commty ])
                
                # ignore communities that are less than ignore_cmmty_lt
                if len(G_subgraph.nodes()) >= ignore_cmmty_lt:
                    print("****Community #" + str(commty+1))
                    no_of_comm_gt_cutoff += 1
                    self.print_cluster_metrics(G_subgraph, G, top_no=3, acc_node_size_cutoff=acc_node_size_cutoff)
                    print("\n")
             
            
            print("Total # of Communities with more than " + str(ignore_cmmty_lt) + ' nodes: ' + str(no_of_comm_gt_cutoff))
            
            
            
import os
import json
import datetime
from pymongo import MongoClient
import pymongo
from pymongo.collation import Collation
from time import strptime,sleep
import datetime
import re
import nltk
from nltk.corpus import words, stopwords, wordnet
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag
from nltk.tokenize import word_tokenize
import csv
import string
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import itertools 
import requests
from requests_oauthlib import OAuth1

dictionary_words = dict.fromkeys(words.words(), None)

import pyphen
pyphen_dic = pyphen.Pyphen(lang='en')
    
stopWords = set(stopwords.words('english'))
tokenizer = RegexpTokenizer(r'\w+')

stemmer = PorterStemmer()
lemmatiser = WordNetLemmatizer()

stop = set(stopwords.words('english'))
exclude = set(string.punctuation) 
lemma = WordNetLemmatizer()

topic_doc_complete = []


class TwitterDB:
    """
    TwitterDB class
    """

    def __init__(self, mongoDB_database):
        #Inititalizing MongoDB collections        
        self.db = mongoDB_database                
        self.db.dc_bSettings = self.db.adm_dbSettings
        self.c_loadedFiles = self.db.adm_loadedFiles
        self.c_twitterSearches = self.db.adm_twitterSearches
        self.c_tweet = self.db.tweet
        self.c_focusedTweet = self.db.focusedTweet
        self.c_tweetWords = self.db.tweetWords
        self.c_tweetSentences = self.db.tweetSentences
        self.c_topicsByHashTag = self.db.topicsByHashTag
        self.c_tweetCountByFileAgg = self.db.agg_tweetCountByFile
        self.c_tweetCountByPeriodAgg = self.db.agg_tweetCountByMonth
        self.c_tweetCountByLanguageAgg = self.db.agg_tweetCountByLanguage
        self.c_tweetCountByUserAgg = self.db.agg_tweetCountByUser
        self.c_wordCountAgg = self.db.agg_wordCount
        self.c_hashTagCountAgg = self.db.agg_hashTagCount
        self.c_userLocationCountAgg = self.db.agg_userLocationCount
        self.c_loadStatus = self.db.adm_loadStatus
        self.c_htTopics = self.db.htTopics
        self.c_tweetHashTags = self.db.tweetHashTags
        self.c_tweetConnections = self.db.tweetConnections
        self.c_users = self.db.users
        self.c_tweetHTConnections = self.db.tweetHTConnections
        self.c_tweetHTConnections = self.db.tweetHTConnections
        self.c_searches = self.db.searches
        #temp collections to help with query performance
        self.c_tmpEdges = self.db.tmpEdges
        self.c_tmpEdgesTweetIds = self.db.tmpEdgesTweetIds
        self.c_tmpEdgesHTFreq = self.db.tmpEdgesHTFreq
        self.c_tmpEdgesWordFreq = self.db.tmpEdgesWordFreq
                        
        # Put fields chosen into an array of fields. 
        # These fields will be the ones used in the FocusedTweet collection             
        strFocusedTweetFields="lang;retweet_count;in_reply_to_status_id_str;in_reply_to_screen_name"
        strFocusedTweetUserFields="name;screen_name;description;location;followers_count;friends_count;statuses_count;lang;verified"        
        self.strFocusedTweetFields = strFocusedTweetFields        
        self.strFocusedTweetFieldsArr = strFocusedTweetFields.split(";")
        self.strFocusedTweetUserFieldsArr = strFocusedTweetUserFields.split(";")
            
        # Create unique index on users table to only allow one users with same user_id and screen_name. 
        # (Collation strength=2 garantees case insensitive)
        try:
            resp = self.c_users.create_index([('user_id_str', pymongo.ASCENDING), 
                                              ('screen_name', pymongo.ASCENDING) ], 
                                             unique = True, 
                                             collation=Collation(locale="en_US", strength=2))
        except Exception as e:
            print('Warning: Could not create a new index in users' + str(e))
            
        
        # Create unique index on tweet table to make sure we don't store duplicate tweets        
        try:
            resp = self.c_tweet.create_index([('id', pymongo.ASCENDING)], 
                                             unique = True)
        except:
            pass
                  


    def setFocusedDataConfigs(self, strFocusedTweetFields, strFocusedTweetUserFields):        
        """
        Twitter documents have an extensive number of fields. In order to focus only on the interesting pieces of information, this method allows you to choose which fields to keep. 
                
        Parameters
        ----------               
        strFocusedTweetFields : fields that you find interesting in the Tweet object
        
        strFocusedTweetUserFields : fields that you find interesting in the User object
                
        
        Examples
        --------          
        Setting configurations to decide which fields to keep:
        
            >>> focusedTweetFields = 'lang;retweet_count;in_reply_to_screen_name'
            >>> focusedTweetUserFields = 'name;description;location;friends_count;verified'
            >>> setFocusedDataConfigs(focusedTweetFields, focusedTweetUserFields)
            
        """
        # Put fields chosen into an array of fields. 
        # These fields will be the ones used in the FocusedTweet collection
        self.strFocusedTweetFieldsArr = strFocusedTweetFields.split(";")        
        self.strFocusedTweetUserFieldsArr = strFocusedTweetUserFields.split(";")
        
                
    #####################################
    # Method: loadDocFromFile
    # Description: This method will load tweet .json files into the DB (tweet collection)
    # It goes through all .json files in the directory and load them one by one. 
    # It also saves the files already loaded into the 'loadedFiles' collection 
    # to make sure we don't load the same file twice
    # Parameters: 
    #   -directory = the directory where the files are stored
    def loadDocFromFile(self, directory):
        """
        This method will load tweet .json files into the DB (tweet collection)
        It goes through all .json files in the directory and load them one by one. 
        It also saves the files already loaded into the 'loadedFiles' collection 
        to make sure we don't load the same file twice
        
        Parameters
        ----------                                       
        directory :
            the directory where the files are stored 
    
        Examples
        --------          
            Import data from json files into a mongoDB database:
        
            >>> loadDocFromFile(json_file_path = 'C:\\Data\\My_JSON_Files_folder')
        """    
        seq_no = 1

        starttime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print ("loading process started..." + starttime)
        
        #find the current max sequence number
        select_cTweet = self.c_tweet.aggregate( 
            [{"$group": {"_id": "seq_agg" , "count": { "$max": "$seq_no" } }}])
        for tweetCount in select_cTweet:
            seq_no = tweetCount["count"] + 1                

        #loops through the files in the dictory
        for filename in os.listdir(directory):
            if filename.endswith(".json"):                
                strpath = os.path.join(directory, filename)                               

                #find if file already loaded
                isFileLoaded = self.c_loadedFiles.count_documents({"file_path": strpath.replace("\\", "/") })        

                if isFileLoaded > 0:
                    #if the processing of that file did not finish. Deletes every record for that file so we can start over
                    select_cLoadedFiles = self.c_loadedFiles.find({ "file_path": strpath.replace("\\", "/")})                
                    if select_cLoadedFiles[0]["end_load_time"] == "loading":            
                        self.c_tweet.delete_many({"file_path": strpath.replace("\\", "/")})
                        self.c_loadedFiles.delete_many({"file_path": strpath.replace("\\", "/")})            
                        isFileLoaded=0

                #if file has already been loaded, ignores the file
                if isFileLoaded == 0:

                    #save path in loaded files collection to track which files have already been processed
                    data_loadedfiles = '{"start_load_time":"' \
                                        + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") \
                                        + '","end_load_time":"' \
                                        + "loading" \
                                        + '","file_path":"' \
                                        + strpath.replace("\\", "/") \
                                        + '"}'        
                    self.c_loadedFiles.insert_one(json.loads(data_loadedfiles))

                    #open file and goes through each document to insert tweet into DB (inserts into tweet collection)                    
                    with open(strpath, encoding="utf8") as f:
                        for line in f:        
                            data = json.loads(line)

                            #adding extra fields to document to suport future logic (processed_fl, load_time, file_path )
                            a_dict = {'processed_fl': 'N', 
                                      'seq_no': seq_no, 
                                      'seq_agg': "A", 
                                      'load_time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
                                      'file_path': strpath.replace("\\", "/")}    
                            data.update(a_dict)                                        

                            #ignores documents that are just status
                            if 'info' not in data:
                                self.c_tweet.insert_one(data)
                                seq_no = seq_no+1

                    #update end load time 
                    self.c_loadedFiles.update_one(
                        { "file_path" : strpath.replace("\\", "/") },
                        { "$set" : { "end_load_time" : datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") } });            

                continue        
            else:
                print ("Error loading into tweet collection")        

                
        try:
            resp = self.c_tweet.create_index([('seq_no', pymongo.ASCENDING)])            
        except Exception as e:
            print('Could not create index ' + str(e))    
            
        endtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print ("loading process completed " + endtime)
        
        
    
    # this method will use Twitter API to extract data and save into DB 
    # Parameters: twitterBearer = Bearer from you Twitter developer account
    #             apiName = (30day/fullarchive)
    #             devEnviroment = name of your deve enviroment
    #             query = query to select data from Twitter API
    #             dateStart = period start date
    #             dateEnd = period end date
    #             nextToken = token to start from 
    #             maxResults = maximum number of results that you want to return
    def extractDocFromAPI (self, twitterBearer, apiName, devEnviroment, query, dateStart, dateEnd, nextToken, maxResults):        
        print("Code for extractDocFromAPI. Details for this code on https://git.txstate.edu/l-n63/CS7311 ")        
                  
           
        
    #####################################
    # Method: loadCollection_UpdateStatus
    # Description: This method controls the progress the insertions into other collections. 
    # It calls other methods to load the collections
    # It keeps the progress stored in the db, so that if something fails, 
    #  we can know where to start back up.
    # The progress is stored on collections "adm_loadStatus"
    # Parameters: 
    #   -collection_name = the collections you want to load. 
    #    (Options: focusedTweet, tweetWords, tweetHashTags and tweetConnections)
    #   -inc = how many tweet records you want to load at the time. 
    #    (Large number may cause memory errors, low number may take too long to run)
    #   -type_filter = used only for users collections. 
    #    (Options: tweet, retweet, quote, reply or mention) - Default = None
    def loadCollection_UpdateStatus(self, collection_name, inc, type_filter=None):

        starttime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print ('loading process started (' + collection_name + ('-' + type_filter if type_filter is not None else '')  +  ')... ' + starttime)
            
        last_seq_no = -1
        max_seq_no = 0
        minV = 0

        #get the max sequence number from the tweet collection
        select_cTweet = self.c_tweet.aggregate( [{"$group": {"_id": "seq_agg" , "count": { "$max": "$seq_no" } } } ])
        for tweetCount in select_cTweet:
            max_seq_no = tweetCount["count"]

        #check if the process has already been run or not. This is to make sure we can restart a process from where we stopped
        if type_filter is not None: 
            hasStarted = self.c_loadStatus.count_documents({"collection_name": collection_name, "type_filter": type_filter })        
        else:
            hasStarted = self.c_loadStatus.count_documents({"collection_name": collection_name })
            
        if hasStarted > 0:
            select_cLoadStatus = self.c_loadStatus.find({"collection_name": collection_name })                
            if select_cLoadStatus[0]["status"] == "loading":
                last_seq_no = select_cLoadStatus[0]["min_seq"]-1
                if collection_name == 'focusedTweet':                    
                    self.c_focusedTweet.delete_many({ "seq_no" : { "$gte" : select_cLoadStatus[0]["min_seq"] } })
                elif collection_name == 'tweetWords':
                    self.c_tweetWords.delete_many({ "tweet_seq_no" : { "$gte" : select_cLoadStatus[0]["min_seq"] } })
                elif collection_name == 'tweetHashTags':                    
                    self.c_tweetHashTags.delete_many({ "tweet_seq_no" : { "$gte" : select_cLoadStatus[0]["min_seq"] } })
                elif collection_name == 'tweetConnections':
                    self.c_tweetConnections.delete_many({ "tweet_seq_no" : { "$gte" : select_cLoadStatus[0]["min_seq"] } })
                elif collection_name == 'tweetHTConnections':
                    self.c_tweetHTConnections.delete_many({ "tweet_seq_no" : { "$gte" : select_cLoadStatus[0]["min_seq"] } })
                                
            elif select_cLoadStatus[0]["status"] == "success":
                last_seq_no = select_cLoadStatus[0]["max_seq"] 
        else:
            if type_filter is not None: 
                data = '{"collection_name":"' + collection_name + '", "type_filter":"' + type_filter + '"}'
            else:
                data = '{"collection_name":"' + collection_name + '"}'
            doc = json.loads(data)
            self.c_loadStatus.insert_one(doc)


        # try:
        # loop through tweet sequence numbers to insert into DB. 
        # The variable "inc" will dictate how many tweet we will isert at a time int DB
        minV = last_seq_no+1
        while minV <= max_seq_no: 
            
            if type_filter is not None: 
                self.c_loadStatus.update_one(
                    {"collection_name": collection_name, "type_filter": type_filter }, 
                    { "$set" : { "min_seq" : minV, "max_seq" : minV+inc, "status" : "loading" } } )
            else:
                self.c_loadStatus.update_one(
                    {"collection_name": collection_name }, 
                    { "$set" : { "min_seq" : minV, "max_seq" : minV+inc, "status" : "loading" } } )
                
            if collection_name == 'focusedTweet':                    
                self.loadFocusedDataMinMax(minV, minV+inc)
            elif collection_name == 'tweetWords':
                self.breakTextIntoWords(minV, minV+inc)                    
            elif collection_name == 'tweetHashTags':
                self.loadTweetHashTagsMinMax(minV, minV+inc)                
            elif collection_name == 'tweetConnections':
                self.loadTweetConnectionsMinMax(minV, minV+inc)
            elif collection_name == 'tweetHTConnections':
                self.loadTweetHTConnectionsMinMax(minV, minV+inc)                
            elif collection_name == 'users':
                self.loadUsersDataMinMax(minV, minV+inc, type_filter)

            minV=minV+inc

        #if everyhting was successfull, saves status as "success"
        if type_filter is not None: 
            self.c_loadStatus.update_one(
                {"collection_name": collection_name, "type_filter": type_filter }, 
                { "$set" : { "max_seq" : max_seq_no, "status" : "success" } } )
        else:
            self.c_loadStatus.update_one(
                {"collection_name": collection_name }, 
                { "$set" : { "max_seq" : max_seq_no, "status" : "success" } } )
                                  
        endtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print ('loading process completed (' + collection_name + ')... ' + endtime)
        
    
    def cleanTweetText(self, text):
        """
        Method used to clean the tweet message. Hashtags, user screen names, 
        links, and special characters are removed.
        
        Parameters
        ----------                                       
        text :
            the string to clean
            
        Return
        --------              
        text_clean :
            the string after it's been cleaned
    
        Examples
        --------                              
            >>> cleanTweetText('re The text to Clean for @Jane!!! :) #python')
            The text to Clean for
        """    
        
        text = text.replace("\\", "").replace('\"', "").replace("\r","")
        text = text.replace("\n","").replace("\t", "").rstrip()
        text = text.lower()       

        # removing hashtahs, mentions and links from clean text    
        text_clean = text.replace("http", " http").replace("#", " #")        
        text_clean = text_clean.replace("@", " @").replace("  ", " ").strip()
        words = text_clean.split()
        text_clean = ''            
        for word in list(words):            
            if word[0:1] != '#' and word[0:1] != '@' and word[0:4] != 'http'and word[0:2] != 'rt':
                text_clean = text_clean + word + ' '                                    

        # removing apecial characters
        text_clean = text_clean.replace("\\", "").replace("@","").replace("!", "")
        text_clean = text_clean.replace("/", "").replace("*", "").replace("&amp;", "")
        text_clean = text_clean.replace("-", "").replace("~", "").replace("`", "")
        text_clean = text_clean.replace("#", "").replace("$", "").replace("…", "")
        text_clean = text_clean.replace("%", "").replace("^", "").replace("&", "")
        text_clean = text_clean.replace("(", "").replace(")", "").replace("—", "")
        text_clean = text_clean.replace("=", "").replace("+", "").replace("{", "")
        text_clean = text_clean.replace("}", "").replace("[", "").replace("“", "")
        text_clean = text_clean.replace("’", "").replace("]", "").replace("|", "")
        text_clean = text_clean.replace("'", "").replace('"', "").replace("?", "")
        text_clean = text_clean.replace(":", "").replace(";", "").replace("<", "")
        text_clean = text_clean.replace(">", "").replace(",", "").replace(".", "")
        text_clean = text_clean.replace("_", "").replace("\\\\", "")
        text_clean = text_clean.replace("  ", " ").strip()

        return text_clean

    
    #####################################
    # Method: loadFocusedData
    # Description: This method will call loadCollection_UpdateStatus to load the focusedtweet collection
    # Parameter:  
    #   -inc = how many tweet records you want to load at the time. 
    #   (Large number may cause memory errors, low number may take too long to run)
    def loadFocusedData(self, inc):
        """
        Method to load focused data into mongoDB based on the configurations set on setFocusedDataConfigs. 
        It creates collection tweetFocusedData
        
        Parameters
        ----------                                       
        inc : 
            used to determine how many tweets will be processed at a time.
            A large number may cause out of memory errors, and a low number may take a long time to run, 
            so the decision of what number to use should be made based on the hardware specification.
            the string to clean
                   
        Examples
        --------                              
            >>> loadFocusedData(50000)            
        """    
        
        
        self.loadCollection_UpdateStatus('focusedTweet', inc )
        
    
    #####################################
    # Method: loadFocusedDataMinMax
    # Description: This method will load the focusedtweet collection with the interesting information we want to study
    # It filters by a interval number of tweets. 
    # This is because loading everything at once might cause out of memory errors
    # Parameters:   
    #   -minV & maxV = the tweet seq_no interval you want to run this analysis for
    def loadFocusedDataMinMax(self, minV, maxV):     

        file_data = []

        select_cTweet = self.c_tweet.find({"seq_no":{ "$gt":minV,"$lte":maxV}})
        #select_cTweet = self.c_tweet.find({"seq_no":{ "$gt":2,"$lte":3}})

        #loop through tweets
        for tweet in select_cTweet:    
            
            #Get all the basic info about the tweet. (These will always be saved independet of configurations)    
            seq_no = tweet['seq_no']
            id_str = tweet['id_str']
            created_at = datetime.datetime.strptime(tweet['created_at'], "%a %b %d %H:%M:%S +0000 %Y")
            year =  tweet['created_at'][26:30]
            month_name =  tweet['created_at'][4:7]
            month_no =  str(strptime(month_name,'%b').tm_mon)
            day =  tweet['created_at'][8:10]
            user_id =  tweet['user']['id_str']
            
                        
            ## ***************************************************
            ## *** Getting the text information from differnt fields and different formats ****    

            # when the tweet is large, the full text is saved ion the field extended_tweet
            if 'extended_tweet' in tweet:                   
                text = tweet['extended_tweet']['full_text']
            elif 'full_text' in tweet:
                text =  tweet['full_text']
            else:
                text =  tweet['text']                   
                
            text = text.replace("\\", "").replace('\"', "").replace("\r","")    
            text = text.replace("\n","").replace("\t", "").rstrip()
            text_lower = text.lower()

            # text from the quoted text
            quote_text = ""
            if 'quoted_status' in tweet:
                if 'extended_tweet' in tweet['quoted_status']:
                    quote_text = tweet['quoted_status']['extended_tweet']['full_text']
                elif 'full_text' in tweet['quoted_status']:
                    quote_text = tweet['quoted_status']['full_text']
                else:
                    quote_text = tweet['quoted_status']['text']
                    
            quote_text = quote_text.replace("\\", "").replace('\"', "").replace("\r","")  
            quote_text = quote_text.replace("\n","").replace("\t", "").rstrip()
            quote_text = quote_text.lower()

            # text from original tweet if this is a retweet
            retweeted_text = ""
            if 'retweeted_status' in tweet:                      
                if 'extended_tweet' in tweet['retweeted_status']:
                    retweeted_text = tweet['retweeted_status']['extended_tweet']['full_text']
                elif 'full_text' in tweet['retweeted_status']:
                    retweeted_text = tweet['retweeted_status']['full_text']
                else:
                    retweeted_text = tweet['retweeted_status']['text']                       
            
            retweeted_text = retweeted_text.replace("\\", "").replace('\"', "").replace("\r","")
            retweeted_text = retweeted_text.replace("\n","").replace("\t", "").rstrip()
            retweeted_text = retweeted_text.lower()
                        
            
            text_combined = text_lower + ' ' + quote_text
            
            
            text_combined_clean = self.cleanTweetText(text_combined)
            
            '''
            # removing hashtahs, mentions and links from clean text    
            text_combined_clean = text_combined.replace("http", " http").replace("#", " #")        
            text_combined_clean = text_combined_clean.replace("@", " @").replace("  ", " ").strip()
            words = text_combined_clean.split()
            text_combined_clean = ''            
            for word in list(words):            
                if word[0:1] != '#' and word[0:1] != '@' and word[0:4] != 'http'and word[0:2] != 'rt':
                    text_combined_clean = text_combined_clean + word + ' '                                    
            
            text_combined_clean = text_combined_clean.replace("\\", "").replace("@","").replace("!", "")
            text_combined_clean = text_combined_clean.replace("/", "").replace("*", "").replace("&amp;", "")
            text_combined_clean = text_combined_clean.replace("-", "").replace("~", "").replace("`", "")
            text_combined_clean = text_combined_clean.replace("#", "").replace("$", "").replace("…", "")
            text_combined_clean = text_combined_clean.replace("%", "").replace("^", "").replace("&", "")
            text_combined_clean = text_combined_clean.replace("(", "").replace(")", "").replace("—", "")
            text_combined_clean = text_combined_clean.replace("=", "").replace("+", "").replace("{", "")
            text_combined_clean = text_combined_clean.replace("}", "").replace("[", "").replace("“", "")
            text_combined_clean = text_combined_clean.replace("’", "").replace("]", "").replace("|", "")
            text_combined_clean = text_combined_clean.replace("'", "").replace('"', "").replace("?", "")
            text_combined_clean = text_combined_clean.replace(":", "").replace(";", "").replace("<", "")
            text_combined_clean = text_combined_clean.replace(">", "").replace(",", "").replace(".", "")
            text_combined_clean = text_combined_clean.replace("_", "").replace("\\\\", "")
            text_combined_clean = text_combined_clean.replace("  ", " ").strip()
            '''
            ## ***************************************************************************
            
            
            
                                    
            ## ***************************************************************************
            ## *** Getting the hashtag information - (original tweets, and quotes)

            ht_children = []

            def addHTToList(ht, type_ht):                    

                ht_children.append({
                    'ht': ht, 'ht_lower': ht.lower(), 'type_ht' : type_ht
                })


            # get Hashtags            
            type_ht = 'original'             
            if 'extended_tweet' in tweet:
                for gt_tweet in tweet['extended_tweet']['entities']['hashtags']:
                    ht = gt_tweet['text'] 
                    addHTToList(ht,type_ht)
            else:        
                for gt_tweet in tweet['entities']['hashtags']:                    
                    ht = gt_tweet['text']
                    addHTToList(ht,type_ht)                                    

            if 'quoted_status' in tweet:                
                type_ht = 'quote'
                if 'extended_tweet' in tweet['quoted_status']:     
                    if 'entities' in tweet['quoted_status']['extended_tweet']:
                        for gt_tweet in tweet['quoted_status']['extended_tweet']['entities']['hashtags']:
                            ht = gt_tweet['text']
                            addHTToList(ht,type_ht)

                elif 'entities' in tweet['quoted_status']:
                    for gt_tweet in tweet['quoted_status']['entities']['hashtags']:
                        ht = gt_tweet['text']     
                        addHTToList(ht,type_ht)

            ## ***************************************************************************
    
                

            # creating the json doc
            data = '{"id_str":"' + id_str + \
                    '", "text":"' + text + \
                    '", "text_lower":"' + text_lower + \
                    '", "quote_text":"' + quote_text + \
                    '", "retweeted_text":"' + retweeted_text + \
                    '", "text_combined":"' + text_combined + \
                    '", "text_combined_clean":"' + text_combined_clean + \
                    '", "year":"' + year + \
                    '", "month_name":"' + month_name + \
                    '", "month_no":"' + month_no + \
                    '", "day":"' + day + \
                    '", "user_id":"' + user_id + \
                    '", "hashtags":"' + "" + '"}'
            doc = json.loads(data)
            doc['hashtags'] = ht_children
            

            
            # ***** adding other fields to collection based on the list of fields from configuration - 
            # (configuration is set on the instantiation of the class object)
            def addFieldToDoc(field_name, field_content):
                #if it is a string, clean tab and enter characters
                if isinstance(field_content,str):
                    field_content.replace("\\", "").replace('\"', "").replace("\r","")
                    field_content = field_content.replace("\n","").replace("\t", "").rstrip()

                if field_content is None:
                    field_content = "None"            

                a_dict = {field_name : field_content}    
                doc.update(a_dict)     

            # go through the list of fields from configuration and add to the document
            for i in self.strFocusedTweetFieldsArr:         
                field_name = i
                field_content = tweet[i]
                addFieldToDoc(field_name, field_content)

            #go through the list of user fields from configuration and add to the document
            for i in self.strFocusedTweetUserFieldsArr:         
                field_name = 'user_' + i
                field_content = tweet['user'][i]
                addFieldToDoc(field_name, field_content)                        
            
            # **************************
            

            # add created_at
            a_dict = {'tweet_created_at': created_at}
            doc.update(a_dict)     
                
            # add seq number to the end
            a_dict = {'seq_no': seq_no, 'seq_agg': "A"}    
            doc.update(a_dict)  

            # Add this tweet doc to the array. the array of all 
            # tweets will be used to insertMany into mongoDB 
            file_data.append(doc)

        # insert records into collection
        try:
            self.c_focusedTweet.insert_many(file_data)
        except Exception as e:
            print("Error loading focused tweet | " +str(e) )  

        
        # Create indexes in collection. This will help performance later
        try:
            resp = self.c_focusedTweet.create_index([('seq_no', pymongo.ASCENDING)])            
        except Exception as e:
            print('Could not create index in focusedTweet' + str(e))

        try:
            resp = self.c_focusedTweet.create_index([('id_str', pymongo.ASCENDING)])            
        except Exception as e:
            print('Could not create index in focusedTweet' + str(e))
            

 

    #####################################
    # Method: loadUsersData
    # Description: This method will call loadCollection_UpdateStatus to load the users collection
    # Users are store in different part of the tweet. 
    # In the tweet itself, in the retweet branch, in the quote branch, in the field in_reply_to_user and in the mention branch. 
    # Use parameter "user_type_filter" to select which type you want to load. 
    # IMPORTANT: Types reply and mention do not contain full user information
    # This method also creates a index to prevent duplicate user information. 
    # If a user already exists, it just rejects the insertion. 
    # Parameters:  
    #   -inc = how many tweet records you want to load at the time. 
    #    (Large number may cause memory errors, low number may take too long to run)
    #   -user_type_filter = the type of user you want to load - 
    #    (Options: tweet, retweet, quote, reply and mention)
    def loadUsersData(self, inc, user_type_filter):
        """
        Method to load user data into mongoDB. 
        It creates collection users
        
        Parameters
        ----------                                       
        inc : 
            used to determine how many tweets will be processed at a time.
            A large number may cause out of memory errors, and a low number may take a long time to run, 
            so the decision of what number to use should be made based on the hardware specification.
            the string to clean
        user_type_filter :
            the type of user you want to load - 
            (Options: tweet, retweet, quote, reply and mention)
                   
        Examples
        --------                              
            >>> loadUsersData(50000, 'tweet')
        """    
        
        self.loadCollection_UpdateStatus('users', inc, user_type_filter)
        

    #####################################
    # Method: loadUsersDataMinMax
    # Description: This method will load the users collection 
    # It filters by a interval number of tweets. 
    # This is because loading everything at once might cause out of memory errors
    # Parameters:   
    #   -minV & maxV = the tweet seq_no interval you want to run this analysis for
    #   -user_type_filter = the type of user you want to to load - 
    #    (Options: tweet, retweet, quote, reply and mention) 
    def loadUsersDataMinMax(self, minV, maxV, user_type_filter):
        
        file_data = []        
        
        select_cTweet = self.c_tweet.find({"seq_no":{ "$gt":minV,"$lte":maxV}})

        # add another json record to the array of records to insert
        def addToList(user_type, user_id_str, screen_name, name, location, 
                      description, created_at, protected, verified, followers_count, 
                      friends_count, listed_count, favourites_count, statuses_count):

            location_clean = ''
            description_clean = ''

            if location is not None:
                location_clean = location.replace("\\", "").replace('\"', "").replace("\r","")
                location_clean = location_clean.replace("\n","").replace("\t", "").rstrip()
            if description is not None:
                description_clean = description.replace("\\", "").replace('\"', "").replace("\r","")
                description_clean = description_clean.replace("\n","").replace("\t", "").rstrip()

            if screen_name is None:
                screen_name = user_id_str  
                
            data = '{"screen_name":"' + screen_name  + '"}'
            doc = json.loads(data)
            add_col = {'user_id_str': user_id_str}
            doc.update(add_col)
            add_col = {'name': name}
            doc.update(add_col)  
            add_col = {'user_created_at': created_at}
            doc.update(add_col)
            add_col = {'location': location}
            doc.update(add_col)
            add_col = {'location_clean': location_clean}
            doc.update(add_col)  
            add_col = {'description': description}
            doc.update(add_col)  
            add_col = {'description_clean': description_clean}
            doc.update(add_col)  
            add_col = {'protected': protected}
            doc.update(add_col)  
            add_col = {'verified': verified}
            doc.update(add_col)  
            add_col = {'followers_count': followers_count}
            doc.update(add_col)  
            add_col = {'friends_count': friends_count}
            doc.update(add_col)  
            add_col = {'listed_count': listed_count}
            doc.update(add_col)  
            add_col = {'favourites_count': favourites_count}
            doc.update(add_col)  
            add_col = {'statuses_count': statuses_count}
            doc.update(add_col)  
            add_col = {'user_type': user_type}
            doc.update(add_col)

            file_data.append(doc)
            
            
        #loop through tweets
        for tweet in select_cTweet:    

            if user_type_filter == 'tweet':                
                user_id_str = tweet['user']['id_str']
                name = tweet['user']['name']
                screen_name = tweet['user']['screen_name']
                location = tweet['user']['location']
                description = tweet['user']['description']
                protected = tweet['user']['protected']
                followers_count = tweet['user']['followers_count']
                friends_count = tweet['user']['friends_count']
                listed_count = tweet['user']['listed_count']
                created_at = tweet['user']['created_at']
                favourites_count =tweet['user']['favourites_count']
                verified = tweet['user']['verified']
                statuses_count = tweet['user']['statuses_count']        
                addToList(user_type_filter, user_id_str, screen_name, name, location, 
                          description, created_at, protected, verified, followers_count, 
                          friends_count, listed_count, favourites_count, statuses_count)


            #user from the retweet original tweet
            if user_type_filter == 'retweet':
                if 'retweeted_status' in tweet:                      
                    if 'user' in tweet['retweeted_status']:                        
                        user_id_str = tweet['retweeted_status']['user']['id_str']
                        name = tweet['retweeted_status']['user']['name']
                        screen_name = tweet['retweeted_status']['user']['screen_name']
                        location = tweet['retweeted_status']['user']['location']
                        description = tweet['retweeted_status']['user']['description']
                        protected = tweet['retweeted_status']['user']['protected']
                        followers_count = tweet['retweeted_status']['user']['followers_count']
                        friends_count = tweet['retweeted_status']['user']['friends_count']
                        listed_count = tweet['retweeted_status']['user']['listed_count']
                        created_at = tweet['retweeted_status']['user']['created_at']
                        favourites_count =tweet['retweeted_status']['user']['favourites_count']
                        verified = tweet['retweeted_status']['user']['verified']
                        statuses_count = tweet['retweeted_status']['user']['statuses_count']
                        addToList(user_type_filter, user_id_str, screen_name, name, location, 
                                  description, created_at, protected, verified, followers_count, 
                                  friends_count, listed_count, favourites_count, statuses_count)


            #user from the quoted tweet
            if user_type_filter == 'quote':
                if 'quoted_status' in tweet:                      
                    if 'user' in tweet['quoted_status']:                        
                        user_id_str = tweet['quoted_status']['user']['id_str']
                        name = tweet['quoted_status']['user']['name']
                        screen_name = tweet['quoted_status']['user']['screen_name']
                        location = tweet['quoted_status']['user']['location']
                        description = tweet['quoted_status']['user']['description']
                        protected = tweet['quoted_status']['user']['protected']
                        followers_count = tweet['quoted_status']['user']['followers_count']
                        friends_count = tweet['quoted_status']['user']['friends_count']
                        listed_count = tweet['quoted_status']['user']['listed_count']
                        created_at = tweet['quoted_status']['user']['created_at']
                        favourites_count =tweet['quoted_status']['user']['favourites_count']
                        verified = tweet['quoted_status']['user']['verified']
                        statuses_count = tweet['quoted_status']['user']['statuses_count']
                        addToList(user_type_filter, user_id_str, screen_name, name, location, 
                                  description, created_at, protected, verified, followers_count, 
                                  friends_count, listed_count, favourites_count, statuses_count)

            #in reply to user
            if user_type_filter == 'reply':
                if tweet['in_reply_to_user_id'] != None:                            
                    user_id_str = tweet['in_reply_to_user_id_str']        
                    screen_name = tweet['in_reply_to_screen_name']
                    addToList(user_type_filter, user_id_str, screen_name, name=None, location=None, 
                              description=None, created_at=None, protected=None, verified=None, 
                              followers_count=None, friends_count=None, listed_count=None, favourites_count=None, statuses_count=None)



            #find mentioned user
            if user_type_filter == 'mention':                
                if 'extended_tweet' in tweet:                
                    for gt_tweet in tweet['extended_tweet']['entities']['user_mentions']:                    
                        user_id_str = gt_tweet['id_str']        
                        screen_name = gt_tweet['screen_name']
                        addToList(user_type_filter, user_id_str, screen_name, name=None, location=None, 
                                  description=None, created_at=None, protected=None, verified=None, followers_count=None, 
                                  friends_count=None, listed_count=None, favourites_count=None, statuses_count=None)
                else:                       
                    for gt_tweet in tweet['entities']['user_mentions']:                                        
                        user_id_str = gt_tweet['id_str']        
                        screen_name = gt_tweet['screen_name']
                        addToList(user_type_filter, user_id_str, screen_name, name=None, location=None, 
                                  description=None, created_at=None, protected=None, verified=None, 
                                  followers_count=None, friends_count=None, listed_count=None, favourites_count=None, statuses_count=None)

                #find retweets mentions
                if 'retweeted_status' in tweet:
                    if 'extended_tweet' in tweet['retweeted_status']:     
                        if 'entities' in tweet['retweeted_status']['extended_tweet']:                        
                            for gt_tweet in tweet['retweeted_status']['extended_tweet']['entities']['user_mentions']:                            
                                user_id_str = gt_tweet['id_str']        
                                screen_name = gt_tweet['screen_name']
                                addToList(user_type_filter, user_id_str, screen_name, name=None, location=None, 
                                          description=None, created_at=None, protected=None, verified=None, 
                                          followers_count=None, friends_count=None, listed_count=None, favourites_count=None, statuses_count=None)

                    elif 'entities' in tweet['retweeted_status']:
                        for gt_tweet in tweet['retweeted_status']['entities']['user_mentions']:                       
                            user_id_str = gt_tweet['id_str']        
                            screen_name = gt_tweet['screen_name']
                            addToList(user_type_filter, user_id_str, screen_name, name=None, location=None, 
                                      description=None, created_at=None, protected=None, verified=None, followers_count=None, 
                                      friends_count=None, listed_count=None, favourites_count=None, statuses_count=None)

                #find quote mentions
                if 'quoted_status' in tweet:                                              
                    #find mentions in a quote
                    if 'extended_tweet' in tweet['quoted_status']:     
                        if 'entities' in tweet['quoted_status']['extended_tweet']:                        
                            for gt_tweet in tweet['quoted_status']['extended_tweet']['entities']['user_mentions']:                            
                                user_id_str = gt_tweet['id_str']        
                                screen_name = gt_tweet['screen_name']
                                addToList(user_type_filter, user_id_str, screen_name, name=None, location=None, 
                                          description=None, created_at=None, protected=None, verified=None, followers_count=None, 
                                          friends_count=None, listed_count=None, favourites_count=None, statuses_count=None)

                    elif 'entities' in tweet['quoted_status']:
                        for gt_tweet in tweet['quoted_status']['entities']['user_mentions']:
                            user_id_str = gt_tweet['id_str']        
                            screen_name = gt_tweet['screen_name']
                            addToList(user_type_filter, user_id_str, screen_name, name=None, location=None, 
                                      description=None, created_at=None, protected=None, verified=None, followers_count=None, 
                                      friends_count=None, listed_count=None, favourites_count=None, statuses_count=None)
                                
            
        # insert user into  db
        try:
            self.c_users.insert_many(file_data, ordered=False)
        except Exception as e:
            if str(type(e).__name__) == "BulkWriteError":  #igones if just failed when trying to insert duplicate users
                pass
            else:
                print('Error in insert many user - ' + str(type(e).__name__))        
        
        
        
            
    #####################################
    # Method: loadTweetHashTags
    # Description: This method will call loadCollection_UpdateStatus to load the hashtag collection    
    # Parameter:  
    #   -inc = how many tweet records you want to load at the time. 
    #    (Large number may cause memory errors, low number may take too long to run)        
    def loadTweetHashTags(self, inc):
        """
        Method to load hashthas in a separate collection in mongoDB. 
        It creates the tweetHashTags collection.
        
        Parameters
        ----------                                       
        inc : 
            used to determine how many tweets will be processed at a time.
            A large number may cause out of memory errors, and a low number may take a long time to run, 
            so the decision of what number to use should be made based on the hardware specification.
            the string to clean
                   
        Examples
        --------                              
            >>> loadTweetHashTags(50000)
        """    
        
        self.loadCollection_UpdateStatus('tweetHashTags', inc )        
        
    
    #####################################
    # Method: loadTweetHashTagsMinMax
    # Description: This method will load the hashtags associated to each tweet
    # It filters by a interval number of tweets. 
    # This is because loading everything at once might cause out of memory errors
    # Parameters:   
    #   -minV & maxV = the tweet seq_no interval you want to run this analysis for    
    def loadTweetHashTagsMinMax(self, minV, maxV):     

        file_data = []

        select_cTweet = self.c_focusedTweet.find({"seq_no":{ "$gt":minV,"$lte":maxV}})
        
        # add another json record to the array of records to insert
        def addToList(id_str, type_ht, ht, ht_lower, created_at):                                 

            #creating the json doc
            data = '{"tweet_id_str":"' + id_str + \
                    '", "type_ht":"' + type_ht + \
                    '", "ht":"' + ht + \
                    '", "ht_lower":"' + ht_lower + '"}'
            doc = json.loads(data)

            #add created_at
            a_dict = {'tweet_created_at': created_at}
            doc.update(a_dict)
            
            #add seq number to the end
            a_dict = {'tweet_seq_no': seq_no, 'seq_agg': "A"}
            doc.update(a_dict)

            # Add this tweet doc to the array. the array of all tweets 
            # will be used to insertMany into mongoDB 
            file_data.append(doc)


        #loop through tweets
        for tweet in select_cTweet:
            
            id_str = tweet['id_str']
            seq_no = tweet['seq_no']
            created_at = tweet['tweet_created_at']
                        
            #get Hashtags            
            if 'hashtags' in tweet:
                for gt_tweet in tweet['hashtags']:
                    
                    ht = gt_tweet['ht']
                    ht_lower = gt_tweet['ht_lower']                    
                    type_ht = gt_tweet['type_ht']             
                    
                    #creating the json doc
                    data = '{"tweet_id_str":"' + id_str + \
                            '", "type_ht":"' + type_ht + \
                            '", "ht":"' + ht + \
                            '", "ht_lower":"' + ht_lower + '"}'
                    doc = json.loads(data)

                    #add created_at
                    a_dict = {'tweet_created_at': created_at}
                    doc.update(a_dict)

                    #add seq number to the end
                    a_dict = {'tweet_seq_no': seq_no, 'seq_agg': "A"}
                    doc.update(a_dict)

                    # Add this tweet doc to the array. the array of all 
                    # tweets will be used to insertMany into mongoDB 
                    file_data.append(doc)
                    

        # insert hashtags into db
        try:
            self.c_tweetHashTags.insert_many(file_data)
        except:
            print("Error loading c_tweetHashTags ")
            
                                
    
    #####################################
    # Method: loadTweetConnections
    # Description: This method will call loadCollection_UpdateStatus to load the tweetConnections collection    
    # Parameter:  
    #   -inc = how many tweet records you want to load at the time. 
    #    (Large number may cause memory errors, low number may take too long to run)     
    def loadTweetConnections(self, inc):
        """
        Method to load tweet connection in a separate collection in mongoDB. 
        It creates the tweetConnections collection.
        
        Parameters
        ----------                                       
        inc : 
            used to determine how many tweets will be processed at a time.
            A large number may cause out of memory errors, and a low number may take a long time to run, 
            so the decision of what number to use should be made based on the hardware specification.
            the string to clean
                   
        Examples
        --------                              
            >>> loadTweetConnections(50000)
        """    
        
        self.loadCollection_UpdateStatus('tweetConnections', inc)
                

    #####################################
    # Method: loadTweetConnectionsMinMax
    # Description: This method will load the tweet connections (edges) associated to each tweet        
    # It filters by a interval number of tweets. 
    # This is because loading everything at once might cause out of memory errors
    # Parameters:  
    #   -minV & maxV = the tweet seq_no interval you want to run this analysis for      
    def loadTweetConnectionsMinMax(self, minV, maxV):

        file_data = []
        user_id_str_b = ''     
        desc = ''
        
        select_cTweet = self.c_tweet.find({"seq_no":{ "$gt":minV,"$lte":maxV}})
                        
        # add another json record to the array of records to insert
        def addToList(id_str, type_conn, user_id_str_a, screen_name_a, 
                      user_id_str_b, screen_name_b, desc, tweet_created_dt, 
                      retweeted_status_id=None, quoted_status_id=None, in_reply_to_status_id=None):

            if user_id_str_a is None:
                user_id_str_a = '' 
            if user_id_str_b is None:
                user_id_str_b = ''
            if retweeted_status_id is None:
                retweeted_status_id = ''
            if quoted_status_id is None: 
                quoted_status_id = ''
            if in_reply_to_status_id is None:
                in_reply_to_status_id = ''
            if screen_name_a is None:
                screen_name_a = user_id_str_a
            if screen_name_b is None:
                screen_name_b = user_id_str_b
                
             
            #to set the edge_screen_name_directed_key
            if screen_name_a > screen_name_b:
                screen_name_a_un = screen_name_a
                screen_name_b_un = screen_name_b
            else:
                screen_name_a_un = screen_name_b
                screen_name_b_un = screen_name_a
                
            #creating the json doc
            data = '{"tweet_id_str":"' + id_str + \
                    '", "type_of_connection":"' + type_conn + \
                    '", "user_id_str_a":"' + user_id_str_a + \
                    '", "screen_name_a":"' + screen_name_a + \
                    '", "user_id_str_b":"' + user_id_str_b + \
                    '", "screen_name_b":"' + screen_name_b + \
                    '", "desc":"' + desc + \
                    '", "retweeted_status_id":"' + str(retweeted_status_id) + \
                    '", "quoted_status_id":"' + str(quoted_status_id) + \
                    '", "in_reply_to_status_id":"' + str(in_reply_to_status_id) + \
                    '", "edge_screen_name_directed_key":"' + screen_name_a.lower() + '-' + screen_name_b.lower() + \
                    '", "edge_screen_name_undirected_key":"' + screen_name_a_un.lower() + '-' + screen_name_b_un.lower() + '"}'
                    
            doc = json.loads(data)            
            
            #add tweet_created_dt
            a_dict = {'tweet_created_at': tweet_created_dt}
            doc.update(a_dict)
            
            #add seq number to the end
            a_dict = {'tweet_seq_no': seq_no, 'seq_agg': "A"}
            doc.update(a_dict)  

            #add this tweet doc to the array. the array of all tweets will be used to insertMany into mongoDB 
            file_data.append(doc)
        
        
        #loop through tweets
        for tweet in select_cTweet:    

            #Get all the basic info about the tweet. 
            id_str = tweet['id_str']
            user_id_str_a = tweet['user']['id_str']
            screen_name_a = tweet['user']['screen_name']
            seq_no = tweet['seq_no']            
            tweet_created_dt = datetime.datetime.strptime(tweet['created_at'], "%a %b %d %H:%M:%S +0000 %Y")
                 
            #find replies
            type_conn = 'reply'
            desc = 'user a replied to user b'
            if tweet['in_reply_to_status_id'] is not None or tweet['in_reply_to_user_id_str'] is not None:
                in_reply_to_status_id = tweet['in_reply_to_status_id_str']
                user_id_str_b = tweet['in_reply_to_user_id_str']
                screen_name_b = tweet['in_reply_to_screen_name']                                
                addToList(id_str, type_conn, user_id_str_a, 
                          screen_name_a, user_id_str_b, screen_name_b, desc, 
                          tweet_created_dt, retweeted_status_id=None, quoted_status_id=None, 
                          in_reply_to_status_id=in_reply_to_status_id)
                                
            #find mentions
            type_conn = 'mention'
            desc = 'user a mentioned user b'
            if 'extended_tweet' in tweet:                
                for gt_tweet in tweet['extended_tweet']['entities']['user_mentions']:                    
                    user_id_str_b = gt_tweet['id_str']
                    screen_name_b = gt_tweet['screen_name']
                    addToList(id_str, type_conn, user_id_str_a, 
                              screen_name_a, user_id_str_b, screen_name_b, desc, 
                              tweet_created_dt, retweeted_status_id=None, quoted_status_id=None)
            else:                       
                for gt_tweet in tweet['entities']['user_mentions']:                                        
                    user_id_str_b = gt_tweet['id_str']
                    screen_name_b = gt_tweet['screen_name']
                    addToList(id_str, type_conn, user_id_str_a, 
                              screen_name_a, user_id_str_b, screen_name_b, desc, 
                              tweet_created_dt, retweeted_status_id=None, quoted_status_id=None)
                       
            #find retweets
            if 'retweeted_status' in tweet:
                type_conn = 'retweet'      
                desc = 'user a retweeted a tweet from user b'                
                
                retweeted_status_id = tweet['retweeted_status']['id_str']
                user_id_str_b = tweet['retweeted_status']['user']['id_str']
                screen_name_b = tweet['retweeted_status']['user']['screen_name']                
                addToList(id_str, type_conn, user_id_str_a, 
                          screen_name_a, user_id_str_b, screen_name_b, desc, 
                          tweet_created_dt, retweeted_status_id=retweeted_status_id, quoted_status_id=None)
                                 
            #find quotes
            if 'quoted_status' in tweet:                                
                type_conn = 'quote'
                desc = 'user a quoted a tweet from user b'
                
                quote_status_id = tweet['quoted_status']['id_str']
                user_id_str_b = tweet['quoted_status']['user']['id_str']
                screen_name_b = tweet['quoted_status']['user']['screen_name']                
                addToList(id_str, type_conn, user_id_str_a, 
                          screen_name_a, user_id_str_b, screen_name_b, desc, 
                          tweet_created_dt, retweeted_status_id=None, quoted_status_id=quote_status_id)                     
                    
                #find mentions in a quote
                type_conn = 'mention_quote'
                if 'extended_tweet' in tweet['quoted_status']:     
                    if 'entities' in tweet['quoted_status']['extended_tweet']:                        
                        for gt_tweet in tweet['quoted_status']['extended_tweet']['entities']['user_mentions']:                            
                            user_id_str_b = gt_tweet['id_str']
                            screen_name_b = gt_tweet['screen_name']
                            addToList(id_str, type_conn, user_id_str_a, 
                                      screen_name_a, user_id_str_b, screen_name_b, desc, 
                                      tweet_created_dt, retweeted_status_id=None, quoted_status_id=quote_status_id)
                            
                elif 'entities' in tweet['quoted_status']:
                    for gt_tweet in tweet['quoted_status']['entities']['user_mentions']:
                        user_id_str_b = gt_tweet['id_str']
                        screen_name_b = gt_tweet['screen_name']
                        addToList(id_str, type_conn, user_id_str_a, 
                                  screen_name_a, user_id_str_b, screen_name_b, desc, 
                                  tweet_created_dt, retweeted_status_id=None, quoted_status_id=quote_status_id)
            
        # insert connections(directed edges) into db
        try:
            self.c_tweetConnections.insert_many(file_data)
        except:
            print("Error loading tweetConnections ")
            

        # create indexes to improve performance
        try:
            resp = self.c_tweetConnections.create_index([('tweet_id_str', pymongo.ASCENDING)])            
        except Exception as e:
            print('Could not create index in tweetConnections' + str(e))

        try:
            resp = self.c_tweetConnections.create_index([('edge_screen_name_directed_key', pymongo.ASCENDING)])            
        except Exception as e:
            print('Could not create index in tweetConnections' + str(e))  
            
        try:
            resp = self.c_tweetConnections.create_index([('edge_screen_name_undirected_key', pymongo.ASCENDING)])            
        except Exception as e:
            print('Could not create index in tweetConnections' + str(e))             
                                
            

            
            
            
    #####################################
    # Method: loadTweetHTConnections
    # Description: This method will call loadCollection_UpdateStatus to load the tweetHTConnections collection        
    # Parameter:  
    #   -inc = how many tweet records you want to load at the time. 
    #    (Large number may cause memory errors, low number may take too long to run)     
    def loadTweetHTConnections(self, inc):
        """
        Method to load hashtag connection in a separate collection in mongoDB. 
        It creates the tweetHTConnections collection.
        
        Parameters
        ----------                                       
        inc : 
            used to determine how many tweets will be processed at a time.
            A large number may cause out of memory errors, and a low number may take a long time to run, 
            so the decision of what number to use should be made based on the hardware specification.
            the string to clean
                   
        Examples
        --------                              
            >>> loadTweetHTConnections(50000)
        """            
        
        self.loadCollection_UpdateStatus('tweetHTConnections', inc)
                

    #####################################
    # Method: loadTweetHTConnectionsMinMax
    # Description: This method will load the tweet hashtags connections (edges) associated to each hashtag for each tweet        
    # It filters by a interval number of tweets. This is because loading everything at once might cause out of memory errors
    # Parameters:  
    #   -minV & maxV = the tweet seq_no interval you want to run this analysis for      
    def loadTweetHTConnectionsMinMax(self, minV, maxV):

        file_data = []
        
        select_cTweet = self.c_focusedTweet.find({"seq_no":{ "$gt":minV,"$lte":maxV}})
                        
        #loop through tweets
        for tweet in select_cTweet:

            id_str = tweet['id_str']
            seq_no = tweet['seq_no']
            created_at = tweet['tweet_created_at']

            #get Hashtags            
            if 'hashtags' in tweet:

                #build array with all hashtags for this one tweet
                ht_arr = []
                for gt_tweet in tweet['hashtags']:
                    ht_arr.append(gt_tweet['ht_lower'])

                #loops through the combinations between the hashtags and insert one records for each combination
                for element in itertools.combinations(ht_arr, 2):   

                    if element[0] < element[1]:
                        ht_a = element[0]
                        ht_b = element[1]                
                    else:
                        ht_a = element[1]
                        ht_b = element[0]            
                    ht_key = ht_a + '-'  + ht_b

                    #creating the json doc
                    data = '{"tweet_id_str":"' + id_str + \
                            '", "ht_a":"' + ht_a + \
                            '", "ht_b":"' + ht_b + \
                            '", "ht_key":"' + ht_key + '"}'
                    doc = json.loads(data)

                    #add created_at
                    a_dict = {'tweet_created_at': created_at}
                    doc.update(a_dict)

                    #add seq number to the end
                    a_dict = {'tweet_seq_no': seq_no, 'seq_agg': "A"}
                    doc.update(a_dict)

                    #add this tweet doc to the array. the array of all tweets will be used to insertMany into mongoDB 
                    file_data.append(doc)


        #insert hashtags into db
        try:
            self.c_tweetHTConnections.insert_many(file_data)
        except:
            print("Error loading tweetHTConnections ")                            

        # create indexes to improve performance
        try:
            resp = self.c_tweetHTConnections.create_index([('tweet_id_str', pymongo.ASCENDING)])            
        except Exception as e:
            print('Could not create index in tweetHTConnections' + str(e))
            
        try:
            resp = self.c_tweetHTConnections.create_index([('ht_key', pymongo.ASCENDING)])            
        except Exception as e:
            print('Could not create index in tweetHTConnections' + str(e))  


            
    #####################################
    # Method: loadWordsData
    # Description: This method will call loadCollection_UpdateStatus to load the tweetWords collection        
    # Parameters:  
    #   -inc = how many tweet records you want to load at the time. 
    #          (Large number may cause memory errors, low number may take too long to run)    
    def loadWordsData(self, inc):
        """
        Method to load the tweet words in a separate collection in mongoDB. 
        It creates the tweetWords collection.
        
        Parameters
        ----------                                       
        inc : 
            used to determine how many tweets will be processed at a time.
            A large number may cause out of memory errors, and a low number may take a long time to run, 
            so the decision of what number to use should be made based on the hardware specification.
            the string to clean
                   
        Examples
        --------                              
            >>> loadWordsData(50000)
        """    
        
        self.loadCollection_UpdateStatus('tweetWords', inc )        
            
           
    #####################################
    # Method: breakTextIntoWords
    # Description: This method will break text from tweet into words and tag them        
    # It filters by a interval number of tweets. 
    # This is because loading everything at once might cause out of memory errors
    # Parameters:  minV & maxV = the tweet seq_no interval you want to run this analysis for    
    def breakTextIntoWords(self, minV, maxV):

        file_data = []
        seq_no = 0
        
        select_cTweetWords = self.c_tweetWords.aggregate( 
            [{"$group": {"_id": "seq_agg" , "maxSeqNo": { "$max": "$seq_no" } } } ])
        for tweetCount in select_cTweetWords:
            max_seq_no = tweetCount["maxSeqNo"] 
            seq_no = max_seq_no                                
        
        select_cFocusedTweet = self.c_focusedTweet.find({"seq_no":{ "$gt":minV,"$lte":maxV}})        
        
        
        
        #loop through tweets
        for tweet in select_cFocusedTweet:
            
            #Get all the basic info about the tweet. 
            # (These will always be saved independet of configurations)    
            id_str = tweet['id_str']
            text =  tweet['text_combined_clean']           
            year =  tweet['year']
            month_name =  tweet['month_name']
            month_no =  tweet['month_no']
            day =  tweet['day']
            user_id =  tweet['user_id']
            seq_no_tweet = tweet['seq_no']                       
            created_at = tweet['tweet_created_at']            

            try:                            
                
                for word in pos_tag(tokenizer.tokenize(text)):                                                                                                                                

                    cleanWordLw = word[0]                    
                    
                    stop_word_fl = 'F'
                    if cleanWordLw in stopWords:
                        stop_word_fl = 'T'                                            
                    
                    en_word_fl = 'T'
                    try:
                        x = dictionary_words[cleanWordLw]
                    except KeyError:
                        en_word_fl = 'F'                           
                                            
                    word_syl = pyphen_dic.inserted(cleanWordLw)
                                            
                    seq_no = seq_no+1                                                            

                    #lemmatize word
                    tag = word[1].lower()[0]                    

                    if tag == 'j':
                        tag = wordnet.ADJ
                    elif tag == 'v':
                        tag = wordnet.VERB
                    elif tag == 'n':
                        tag = wordnet.NOUN
                    elif tag == 'r':
                        tag = wordnet.ADV
                    else:
                        tag  = ''                    
                        
                    if tag in ("j", "n", "v", "r"):
                        lemm_word = lemmatiser.lemmatize(cleanWordLw, pos=tag)
                    else:
                        lemm_word = lemmatiser.lemmatize(cleanWordLw)
                                            
                    data = '{"word":"' + cleanWordLw + \
                            '","word_tag":"' + word[1]  + \
                            '","word_lemm":"' + lemm_word + \
                            '","word_syl":"' + word_syl + \
                            '","stop_word_fl":"' + stop_word_fl + \
                            '","en_word_fl":"' + en_word_fl + \
                            '","tweet_id_str":"' + id_str  + \
                            '", "text":"' + text + \
                            '", "year":"' + year + \
                            '", "month_name":"' + month_name + \
                            '", "month_no":"' + month_no + \
                            '", "day":"' + day + \
                            '", "user_id":"' + user_id + '"}'
                    
                    doc = json.loads(data)                                                                        
                    
                    #add created_at
                    a_dict = {'tweet_created_at': created_at}
                    doc.update(a_dict)  

                    a_dict = {'tweet_seq_no': seq_no_tweet, 'seq_no': seq_no, 'seq_agg': "A"}    
                    doc.update(a_dict)
                    
                    #add this tweet doc to the array. the array of all tweets will be used to insertMany into mongoDB 
                    file_data.append(doc)                                               

            except Exception as e:
                print("Error on loadWordsData. " +str(e) + " | err tweet_id: " + id_str)

        
        #insert words into db
        try:
            self.c_tweetWords.insert_many(file_data)
        except Exception as e:
            print("Error on loadWordsData | " +str(e) ) 
            
        
        # create index to improve performance
        try:
            resp = self.c_tweetWords.create_index([('tweet_seq_no', pymongo.ASCENDING)])            
        except Exception as e:
            print('Could not create index in tweetWords' + str(e))

        try:
            resp = self.c_tweetWords.create_index([('tweet_id_str', pymongo.ASCENDING)])            
        except Exception as e:
            print('Could not create index in tweetWords' + str(e))
            

                
    #####################################
    # Method: loadAggregations
    # Description: load aggregations
    # Parameters:  
    #  -aggType = the type of aggreagation you want to run - 
    #   (Options: tweetCountByFile, hashtagCount, tweetCountByLanguageAgg, 
    #             tweetCountByMonthAgg, tweetCountByUser)
    def loadAggregations(self, aggType):
        """
        Method to load addtional aggregated collection to MongoDB
        It creates the tweetWords collection.
        
        Parameters
        ----------                                       
        aggType : 
            the type of aggreagation you want to run.
            (Options: tweetCountByFile, hashtagCount, tweetCountByLanguageAgg, 
            tweetCountByMonthAgg, tweetCountByUser)
            
        Examples
        --------                              
            >>> loadAggregations('tweetCountByFile')
        """    
    
        print ("loading " + aggType + " process started....")
        
        if (aggType == 'tweetCountByFile'):
            self.tweetCountByFileAgg()
        elif (aggType == 'hashtagCount'):
            self.hashtagCountAgg()
        elif (aggType == 'tweetCountByLanguageAgg'):
            self.tweetCountByLanguageAgg()
        elif (aggType == 'tweetCountByMonthAgg'):
            self.tweetCountByPeriodAgg()
        elif (aggType == 'tweetCountByUser'):
            self.tweetCountByUser()    
            
        print ("loading " + aggType + " process completed.")


    
    #####################################
    # Method: tweetCountByFileAgg
    # Description: load aggregation on tweetCountByFileAgg collection
    def tweetCountByFileAgg(self):
    
        #delete everything from the collection because we will repopulate it
        result = self.c_tweetCountByFileAgg.delete_many({}) 
        select_cTweet = self.c_tweet.aggregate( 
            [{"$group": {"_id": {"file_path": "$file_path"}, "count": { "$sum": 1 } } } ])

        for tweetCount in select_cTweet:            
            try:        
                if tweetCount["_id"]["file_path"] is not None:
                    data = '{"file_path":"' + tweetCount["_id"]["file_path"] + \
                    '", "count":"' + str(tweetCount["count"]) + '"}'                

                    x = json.loads(data)
                    result = self.c_tweetCountByFileAgg.insert_one(x)

            except Exception as e:            
                print("Error running aggregation: tweetCountByFile | " +str(e))
                continue                 
                
                
    
    #####################################
    # Method: hashtagCountAgg
    # Description: load aggregation on hashTagCountAgg collection
    def hashtagCountAgg(self):     

        result = self.c_hashTagCountAgg.delete_many({}) 
        select_cfocusedTweet = self.c_focusedTweet.aggregate( 
            [ {"$unwind": '$hashtags'}, 
            {"$project": { "hashtags": 1, "ht": '$hashtags.ht'} },
            {"$group": { "_id": { "ht": '$hashtags.ht_lower' }, "count": { "$sum": 1 } } } ])

        for tweetCount in select_cfocusedTweet:

            try:    
                data = '{"hashtag":"' + tweetCount["_id"]["ht"] + '"}'
                x = json.loads(data)        

                a_dict = {'count': tweetCount["count"]}    
                x.update(a_dict)

                result = self.c_hashTagCountAgg.insert_one(x)

            except Exception as e:            
                print("Error running aggregation: hashtagCount | " +str(e))
                continue   
            
                    

    #####################################
    # Method: tweetCountByLanguageAgg
    # Description: load aggregation on tweetCountByLanguageAgg collection
    def tweetCountByLanguageAgg(self):

        result = self.c_tweetCountByLanguageAgg.delete_many({}) 
        select_cfocusedTweet = self.c_focusedTweet.aggregate( 
            [{"$group": {"_id": {"lang": "$lang"}, "count": { "$sum": 1 } } } ])

        for tweetCount in select_cfocusedTweet:
            try:        
                data = '{"lang":"' + tweetCount["_id"]["lang"] + \
                '", "count":"' + str(tweetCount["count"]) + '"}'                

                x = json.loads(data)
                result = self.c_tweetCountByLanguageAgg.insert_one(x)

            except Exception as e:            
                print("Error running aggregation: tweetCountByLanguageAgg | " +str(e))
                continue
                
                    
    #####################################
    # Method: tweetCountByPeriodAgg
    # Description: load aggregation on tweetCountByPeriodAgg collection 
    def tweetCountByPeriodAgg(self):

        result = self.c_tweetCountByPeriodAgg.delete_many({}) 
        select_cfocusedTweet = self.c_focusedTweet.aggregate( 
            [{"$group": {"_id": {"year": "$year", "month_no": "$month_no"}, "count": { "$sum": 1 } } } ])

        for tweetCount in select_cfocusedTweet:

            try:        
                data = '{"year":"' + tweetCount["_id"]["year"] + \
                      '","month_no":"' + tweetCount["_id"]["month_no"]  + \
                      '", "count":"' + str(tweetCount["count"]) + '"}'                

                x = json.loads(data)
                result = self.c_tweetCountByPeriodAgg.insert_one(x)

            except Exception as e:            
                print("Error running aggreagation: tweetCountByPeriodAgg | " +str(e))
                continue                                         

    
    #####################################
    # Method: tweetCountByUser
    # Description: load aggregation on tweetCountByUserAgg collection
    def tweetCountByUser(self):

        result = self.c_tweetCountByUserAgg.delete_many({})         
        select_cfocusedTweet = self.c_focusedTweet.aggregate( 
            [{"$group": {"_id": {"user_id": "$user_id", "user_screen_name" : "$user_screen_name"}, 
                         "count": { "$sum": 1 } } } ],
            allowDiskUse = True, collation=Collation(locale="en_US", strength=2))
        
        for tweetCount in select_cfocusedTweet:
            try:        
                data = '{"user_id":"' + tweetCount["_id"]["user_id"] + \
                '", "user_screen_name":"' + tweetCount["_id"]["user_screen_name"]  + \
                '", "count":"' + str(tweetCount["count"]) + '"}'                        

                x = json.loads(data)
                result = self.c_tweetCountByUserAgg.insert_one(x)

            except Exception as e:            
                print("Error running aggregation: tweetCountByUser | " +str(e))
                continue
                  

                    
    #####################################
    # Method: create_tmp_edge_collections
    # Description: This method will create temporary collections to help improve 
    # query performance when filtering data by a list of edges
    # Creating some temp collections, we can create indexes that will increase the lookup performance
    # This method was created to allow performance improvements
    # Parameters:  
    #  -arr_edges = the list of edges you want to search for - 
    #   (format "screen_name"-"screen_name")
    #  -startDate_filter & endDate_filter = if you want to filter your query by a period - (Default=None)
    #  -is_bot_Filter = if you want to filter by a connections being for a bot or not
    def create_tmp_edge_collections(self, arr_edges, arr_ht_edges, query_filter):
                                
        if arr_ht_edges is not None:
            arr_edges = arr_ht_edges                                     
        
        arr_ids = []
        self.c_tmpEdges.delete_many({})
        self.c_tmpEdgesTweetIds.delete_many({})

        # *** creating tmp collection with given edges
        file_data = []
        for x in arr_edges:
            data = '{"edge":"' + x + '"}'
            doc = json.loads(data)
            file_data.append(doc)                                               

        self.c_tmpEdges.insert_many(file_data)
        resp = self.c_tmpEdges.create_index([('edge', pymongo.ASCENDING)])  #creating index on tmp collection
        # **********************


        # *** creating tmp collection for tweet ids for the given edges            
        if arr_edges is not None:
            pipeline = [ {"$lookup":{"from":"tweetConnections",
                                     "localField": "edge",
                                     "foreignField": "edge_screen_name_undirected_key",
                                     "as":"fromItems"}},
                         {"$unwind": "$fromItems" },
                         {"$match": query_filter },
                         {"$project": { "tweet_id_str": "$fromItems.tweet_id_str"} }]
        if arr_ht_edges is not None:
            pipeline = [ {"$lookup":{"from":"tweetHTConnections",
                                     "localField": "edge",
                                     "foreignField": "ht_key",
                                     "as" : "fromItems"}},
                         {"$unwind": "$fromItems" },
                         {"$match": query_filter },
                         {"$project": { "tweet_id_str": "$fromItems.tweet_id_str"} }]
        
        select = self.c_tmpEdges.aggregate(pipeline, allowDiskUse=True)
        for x in select:                             
            arr_ids.append(x['tweet_id_str'])


        file_data = []
        arr_no_dups = list(dict.fromkeys(arr_ids)) 

        for id_str in arr_no_dups :
            data = '{"tweet_id_str":"' + id_str + '"}'
            doc = json.loads(data)
            file_data.append(doc)

        # insert data into tmp collection
        if file_data != []:
            self.c_tmpEdgesTweetIds.insert_many(file_data)
            resp = self.c_tmpEdgesTweetIds.create_index([('tweet_id_str', pymongo.ASCENDING)]) 

        # ******************************
            

    #####################################
    # Method: set_bot_flag_based_on_arr
    # Description: This method will update collections focusedTweet, users,
    # and tweetConnections to identify is a user or tweet connections are from bots.
    # The bot list is passed as parameter
    # Parameters: 
    #   -bots_list_id_str = a list of user_ids that are bots
    #   -inc = how many tweets we want to update at the time for field is_bot_connection.
    #    Default=10000 (High number might take too long to run)
    def set_bot_flag_based_on_arr(self, bots_list_id_str, inc=10000):
        """
        Method to update MongoDb collection with a flag identifieng 
        is a user is a bot or not. 
        It updates the records based on a given list of user_ids.        
        
        Parameters
        ----------      
        bots_list_id_str :
            and array with a list of Twitter user ids that are bots
            
        inc : (Optional)
            how many tweets we want to update at the time for field is_bot_connection.
            Default=10000 (High number might take too long to run)
                   
        Examples
        --------                              
            >>> arr_bots = ['123456', '1231654']
            >>> set_bot_flag_based_on_arr(arr_bots, 20000)
        """    
        
        print("updating bot flag...")
        
        # set all records to be is_bot = 0 at first
        self.c_users.update_many({}, {"$set": {"is_bot": "0"}})
        self.c_tweetConnections.update_many({}, {"$set": {"is_bot": "0"}})
        self.c_tweetHTConnections.update_many({}, {"$set": {"is_bot": "0"}})
        self.c_focusedTweet.update_many({}, {"$set": {"is_bot": "0", "is_bot_connection": "0"}})
        self.c_tweetWords.update_many({}, {"$set": {"is_bot": "0", "is_bot_connection": "0"}})        

        #updates collections based on the given list of bots user_ids
        self.c_users.update_many({'user_id_str': {'$in': bots_list_id_str}}, {'$set': {'is_bot':'1'}})
        self.c_focusedTweet.update_many({'user_id': {'$in': bots_list_id_str}}, {'$set': {'is_bot':'1'}})
        self.c_tweetWords.update_many({'user_id': {'$in': bots_list_id_str}}, {'$set': {'is_bot':'1'}})        
        self.c_tweetConnections.update_many({'user_id_str_a': {'$in': bots_list_id_str}}, {'$set': {'is_bot':'1'}})        
                
        # **** Updating the tweets that are bots or connected to bots                
        i=0; arr_bot_conn = []                
        
        #find all the ids that are connected to bots (replies, retweets, quotes or mentions)
        select = self.c_tweetConnections.find({"is_bot" : "1"})
        for x in select:
            i = i + 1
            arr_bot_conn.append(x['tweet_id_str'])
            # updating records using the $in operator can take a long time if the array is too big. That is why we do it in increments
            if i > inc:
                self.c_focusedTweet.update_many({'id_str': {'$in': arr_bot_conn}}, {'$set': {'is_bot_connection':'1'}})
                self.c_tweetWords.update_many({'id_str': {'$in': arr_bot_conn}}, {'$set': {'is_bot_connection':'1'}})
                self.c_tweetHTConnections.update_many({'id_str': {'$in': arr_bot_conn}}, {'$set': {'is_bot_connection':'1'}})
                arr_bot_conn= []; i = 0

        self.c_focusedTweet.update_many({'id_str': {'$in': arr_bot_conn}}, {'$set': {'is_bot_connection':'1'}})
        self.c_tweetWords.update_many({'id_str': {'$in': arr_bot_conn}}, {'$set': {'is_bot_connection':'1'}})
        self.c_tweetHTConnections.update_many({'id_str': {'$in': arr_bot_conn}}, {'$set': {'is_bot':'1'}})
        # **************************** 
            
        print("updating bot flag completed")
        
       
    # Method: build_filter
    # Description: Build filter for queries. 
    # This is called by method queryData to create the filter that will by used in method
    # Parameters: 
    #  -startDate_filter & endDate_filter: coming from method queryData
    #  -is_bot_Filter: coming from method queryData
    #  -ht_to_filter: coming from method queryData 
    #  -user_conn_filter: coming from method queryData
    #  -exportType: coming from method queryData
    def build_filter(
            self, 
            startDate_filter=None, 
            endDate_filter=None, 
            is_bot_Filter=None, 
            ht_to_filter=None, 
            user_conn_filter=None, 
            exportType=None):                                                      

        # set correct format for start and end dates
        if startDate_filter is not None and endDate_filter is not None:
            start_date = datetime.datetime.strptime(startDate_filter, '%m/%d/%Y %H:%M:%S')
            end_date = datetime.datetime.strptime(endDate_filter, '%m/%d/%Y %H:%M:%S')  
        
        #set the comparison operator for bots queries    
        if is_bot_Filter is not None:            
            if is_bot_Filter == '0':
                bot_filter_comp_operator = "$and"                
            elif is_bot_Filter == '1':
                bot_filter_comp_operator = "$or"
                
        #set up the query filter base on the given parameters
        date_filter = {}        
        bot_filter = {}        
        ht_filter = {}        
        conn_filter = {}
        date_filter_for_edges = {}
        bot_filter_for_edges = {}
        ht_filter_for_edges = {}
        conn_filter_edges = {}
        
        #date filter
        if startDate_filter is not None and endDate_filter is not None:
            date_filter = { "tweet_created_at" : { "$gte": start_date, "$lt": end_date } }
            date_filter_for_edges = { "fromItems.tweet_created_at" : { "$gte": start_date, "$lt": end_date } }
        #bot filter
        if is_bot_Filter is not None:
            bot_filter = { "$or": [ { "is_bot": { "$eq": str(is_bot_Filter) } } , { "is_bot_connection": { "$eq": str(is_bot_Filter) } }]}
            bot_filter_for_edges = { "fromItems.is_bot": { "$eq": str(is_bot_Filter) } }
        #ht filter
        if ht_to_filter is not None:
            ht_filter = {"hashtags.ht_lower": ht_to_filter.lower()}
            ht_filter_for_edges = {}  ##### ***need to address this later            
        if user_conn_filter is not None:
            if exportType == 'edges':
                conn_filter = {"type_of_connection": user_conn_filter.lower()}
            conn_filter_edges = {"type_of_connection": user_conn_filter.lower()}
                
        query_filter = { "$and": [ date_filter, bot_filter, ht_filter, conn_filter ]}
        query_filter_for_edges = { "$and": [ date_filter_for_edges, bot_filter_for_edges, ht_filter_for_edges, conn_filter_edges ]}
        
        return query_filter, query_filter_for_edges
                            
    
    
    #####################################
    # Method: exportData
    # Description: Exports data into \t delimited file
    def exportData(
            self, 
            exportType, 
            filepath, 
            inc, 
            startDate_filter=None, 
            endDate_filter=None, 
            is_bot_Filter=None, 
            arr_edges=None,
            arr_ht_edges=None,
            top_no_filter=None, 
            ht_to_filter=None, 
            include_hashsymb_FL='Y',  
            replace_existing_file=True, 
            user_conn_filter=None):
            
        """
        Method to export the data from MongoDb into text files based on certain filters.        
               
        Examples
        --------                              
            >>> ...
        """  
                
        
        #export edges   
        if (exportType == 'edges'):        
            file = filepath + 'edges.txt'
        #export text for topic analysis
        elif (exportType == 'text_for_topics'):
            file = filepath + 'T_tweetTextsForTopics.txt'
        #export ht frequency list
        elif (exportType == 'ht_frequency_list'):
            file = filepath + 'T_HT_FrequencyList.txt'
        #export words frequency list - (TOP 5000)
        elif (exportType == 'word_frequency_list'):                            
            file = filepath + 'T_Words_FrequencyList.txt'            
        #export text for topic analysis
        elif (exportType == 'tweet_ids_timeseries'):                                                                              
            file = filepath + 'T_tweetIdswithDates.txt'                        
        #export tweetCountByUser
        elif (exportType == 'tweetCount'):
            file = filepath + 'tweetCount.txt'                                    
        #export tweetCountByUser
        elif (exportType == 'userCount'):            
            file = filepath + 'userCount.txt'
        #export tweetCountByUser
        elif (exportType == 'tweetCountByUser'):            
            file = filepath + 'tweetCountByUser.txt'                                
        #export tweetCountByLanguage
        elif (exportType == 'tweetCountByLanguage'):
            file = filepath + '\\tweetCountByLanguage.txt'                                        
        #export tweetCountByFile
        elif (exportType == 'tweetCountByFile'):
            file = filepath + 'tweetCountByFile.txt'
        #export tweetCountByMonth
        elif (exportType == 'tweetCountByMonth'):          
            file = filepath + 'tweetCountByMonth.txt'        
        #export hashtagCount
        elif (exportType == 'hashtagCount'): 
            file = filepath + 'hashtagCount.txt'
        #export topics by hashtag
        elif (exportType == 'topicByHashtag'): 
            file = filepath + 'topicByHashtag.txt'        
        elif (exportType == 'ht_edges'): 
            file = filepath + 'ht_edges.txt'
            
        #export tweetTextAndPeriod
        #if (exportType == 'tweetTextAndPeriod'):                                
        #export tweetDetails
        #if (exportType == 'tweetDetails'):                
        #export words
        #if (exportType == 'wordsOnEachTweet'):  
        #user details on Each Tweet
        #if (exportType == 'userDetailsOnEachTweet'):        
        
        if replace_existing_file==True or not os.path.exists(file):
            arr, file = self.queryData(exportType, 
                                       filepath, inc, 
                                       startDate_filter, 
                                       endDate_filter, 
                                       is_bot_Filter, 
                                       arr_edges, 
                                       arr_ht_edges, 
                                       top_no_filter, 
                                       ht_to_filter, 
                                       user_conn_filter=user_conn_filter)
    
            #export in array into txt file
            self.exportToFile(arr, file)
        
        

    #####################################
    # Method: set_bot_flag_based_on_arr
    # Description: Exports data into \t delimited file
    # Parameters: 
    #   -exportType: (Options: edges, 
    #                          text_for_topics, 
    #                          ht_frequency_list, 
    #                          word_frequency_list 
    #                          tweetCountByUser 
    #                          tweetCountByLanguage, 
    #                          tweetCountByFile, 
    #                          tweetCountByMonth, 
    #                          hashtagCount, 
    #                          tweetTextAndPeriod, 
    #                          wordsOnEachTweet 
    #                          userDetailsOnEachTweet)    
    #   -filepath: the file path where the files will be saved  
    #   -inc: To set how many lines per files we want to save. 
    #    This is for collection that have too many records to be saved. 
    #    Memory issues can happens if this number is too big
    #    Only works when exporting for types tweetTextAndPeriod, wordsOnEachTweet,
    #    userDetailsOnEachTweet, we can set how many lines per file    
    #   -startDate_filter & endDate_filter: Date period you want to filter the tweets by.
    #    Only available for options "edges", "text_for_topics", 
    #    and "ht_frequency_list". (Defaul=None)    
    #   -is_bot_Filter: Filter tweets and connections by being bots or not.
    #    Only available for options "edges", "text_for_topics",
    #    and "ht_frequency_list". (Defaul=None)    
    #   -arr_edges: Filter tweet connections by this array of edges. 
    #    Only available for options "text_for_topics", 
    #    and "ht_frequency_list". (Defaul=None)    
    #   -top_no_filter: Filter top frequent words based on the number of this parameter.
    #    Only available for option "word_frequency_list" (Defaul=None)
    def queryData(
            self, exportType, filepath, inc, 
            startDate_filter=None, 
            endDate_filter=None, 
            is_bot_Filter=None, 
            arr_edges=None, 
            arr_ht_edges=None,
            top_no_filter=None, 
            ht_to_filter=None, 
            include_hashsymb_FL='Y', 
            user_conn_filter=None):
            
        """
        Method to query the data from MongoDb. 
        The method return an array with the data retrieved from MongoDB.
               
        Examples
        --------                              
            >>> ...
        """  
        

        arr = []
        
        # set correct format for start and end dates
        if startDate_filter is not None and endDate_filter is not None:
            start_date = datetime.datetime.strptime(startDate_filter, '%m/%d/%Y %H:%M:%S')
            end_date = datetime.datetime.strptime(endDate_filter, '%m/%d/%Y %H:%M:%S')
            
        
        #build a variable with all filter based on parameters        
        query_filter, query_filter_for_edges = self.build_filter(startDate_filter, 
                                                                 endDate_filter, 
                                                                 is_bot_Filter, 
                                                                 ht_to_filter, 
                                                                 user_conn_filter, 
                                                                 exportType)

                
        #export edges
        if (exportType == 'edges'):
                
            pipeline =  [ {"$match": query_filter },
                          {"$group": {"_id": {"screen_name_a": "$screen_name_a", 
                                              "screen_name_b": "$screen_name_b"},
                                      "count": { "$sum": 1 }}} ]
                            
            #get data from database, loop through records and insert into array
            select_edges = self.c_tweetConnections.aggregate(pipeline, 
                                                             allowDiskUse=True, 
                                                             collation=Collation(locale="en_US", strength=2))
            for x in select_edges:                
                arr.append([ x["_id"]['screen_name_a'], x["_id"]['screen_name_b'],  x['count']])
                                
            #set file path
            file = filepath + 'edges.txt'
            
            
            
        #export hashtag edges
        if (exportType == 'ht_edges'):                        
                
            #in case we don't have an array of edges to filter by
            if arr_edges is None:
                pipeline =  [ {"$match": query_filter },
                              {"$group": {"_id": {"ht_a": "$ht_a", "ht_b": "$ht_b"},
                                          "count": { "$sum": 1 }}} 
                            ]              
                select_edges = self.c_tweetHTConnections.aggregate(pipeline, allowDiskUse=True)

            else:                
                
                #create temp collection for edges                
                self.create_tmp_edge_collections(arr_edges, arr_ht_edges, query_filter_for_edges)                                
                
                #create temp collection for ht
                self.c_tmpEdgesHTFreq.delete_many({})
                pipeline = [ {"$lookup":{
                                       "from": "tweetHTConnections",
                                       "localField": "tweet_id_str",
                                       "foreignField": "tweet_id_str",
                                       "as" : "tweetHTConnections"}},
                              {"$unwind": "$tweetHTConnections"},
                              {"$group": {"_id": {"ht_a": "$tweetHTConnections.ht_a", "ht_b": "$tweetHTConnections.ht_b"},
                                          "count": { "$sum": 1 }}}
                           ]
                select_edges = self.c_tmpEdgesTweetIds.aggregate(pipeline, allowDiskUse=True)
                            
            #get data from database, loop through records and insert into array            
            for x in select_edges:                                
                arr.append([x["_id"]['ht_a'], x["_id"]['ht_b'],  x['count']])
                                
            #set file path
            file = filepath + 'ht_edges.txt'



        #export text for topic analysis
        if (exportType == 'text_for_topics'):                                                                  
                
            #in case we don't have an array of edges to filter by
            if arr_edges is None and arr_ht_edges is None:
                         
                select_texts = self.c_focusedTweet.find(query_filter, { "text_combined_clean": 1} )                                

            #in case we have an array of edges to filter by
            else:
                
                self.create_tmp_edge_collections(arr_edges, arr_ht_edges, query_filter_for_edges)
                
                pipeline = [ {"$lookup":{
                                       "from": "focusedTweet",
                                       "localField": "tweet_id_str",
                                       "foreignField": "id_str",
                                       "as" : "focusedTweet"}},    
                              {"$unwind": "$focusedTweet" },                              
                              {"$project": { "text_combined_clean": "$focusedTweet.text_combined_clean" }}]
                
                select_texts = self.c_tmpEdgesTweetIds.aggregate(pipeline, allowDiskUse=True)
                                


            #get data from database, loop through records and insert into array
            for x in select_texts:
                arr.append([x['text_combined_clean']])
            
            #set file path
            file = filepath + 'T_tweetTextsForTopics.txt'


                        
        #export ht frequency list
        if (exportType == 'ht_frequency_list'):                
                                    
            #in case we don't have an array of edges to filter by
            if arr_edges is None and arr_ht_edges is None:
                
                pipeline = [  {"$match": query_filter },
                              { "$unwind": '$hashtags' },                                                                    
                              {"$group": { "_id": { "ht": '$hashtags.ht' }, "count": { "$sum": 1 } } }]

                select_ht = self.c_focusedTweet.aggregate(pipeline, allowDiskUse=True, collation=Collation(locale="en_US", strength=2))


            #in case we have an array of edges to filter by
            else:
                
                
                #*************************************************************************************
                # Creating a temporary collection with all hashtags for each tweet for the given edges
                # This is possible without creating temp collections, 
                #   but it was done this way to improve performance. 
                # Running with a different Collation can take a LONG time - 
                #  (We need to run with Collation strength=2 to get canse insensitive counts )
                
                #create temp collection for edges                
                self.create_tmp_edge_collections(arr_edges, arr_ht_edges, query_filter_for_edges)                                
                
                #create temp collection for ht
                self.c_tmpEdgesHTFreq.delete_many({})
                pipeline = [ {"$lookup":{
                                       "from": "focusedTweet",
                                       "localField": "tweet_id_str",
                                       "foreignField": "id_str",
                                       "as" : "focusedTweet"}},
                              {"$unwind": "$focusedTweet" },
                              {"$unwind": '$focusedTweet.hashtags' },
                              {"$project": { "ht": '$focusedTweet.hashtags.ht', "tweet_id_str": '$tweet_id_str'  } }]
                file_data = []
                select_ht = self.c_tmpEdgesTweetIds.aggregate(pipeline, allowDiskUse=True)
                for x in select_ht:    
                    data = '{"tweet_id_str":"' + x['tweet_id_str'] + \
                            '", "ht":"' + x['ht'] + '"}'
                    doc = json.loads(data)
                    file_data.append(doc)
                if file_data != []:
                    self.c_tmpEdgesHTFreq.insert_many(file_data)
                #**************************************************************************************
                
                
                #getting counts for each hashtag
                pipeline = [  {"$group": { "_id": { "ht": '$ht' }, "count": { "$sum": 1 } } }]                                
                select_ht = self.c_tmpEdgesHTFreq.aggregate(pipeline, allowDiskUse=True, collation=Collation(locale="en_US", strength=2))
                
                

            hash_symbol = "#"
            if include_hashsymb_FL==False:
                hash_symbol=""

            #get data from database, loop through records and insert into array
            for x in select_ht:
                arr.append([hash_symbol + x['_id']['ht'], x['count']])

            #sort array in count descending order
            def sortSecond(val): 
                return val[1] 
            arr.sort(key=sortSecond,reverse=True) 

            if top_no_filter != None:
                arr = arr[:top_no_filter]
                
                
                                    
            #set file path
            file = filepath + 'T_HT_FrequencyList.txt'
            
            
        #export words frequency list - (TOP 5000)
        if (exportType == 'word_frequency_list'):                
                                                       
            # This variable will get set to True for options where we want to create 
            #  a separate tmp collection to save the words. 
            #  (This was done this way to allow some performance improvements)
            bln_GetWords_From_Text = False             
            
            #in case we don't have an array of edges to filter by
            if arr_edges is None and arr_ht_edges is None:
                                                                                                     
                #if we are filtering by period and by is_bot
                if startDate_filter is not None and endDate_filter is not None and is_bot_Filter is not None:
                    bln_GetWords_From_Text = True
                    select_texts = self.c_focusedTweet.find(query_filter, { "text_combined_clean": 1, "id_str": 1} )
                                        

                #if we are filtering by period only
                elif startDate_filter is not None and endDate_filter is not None:
                    pipeline = [{"$match": {"$and": 
                                            [{"tweet_created_at" : {"$gte": start_date, "$lt": end_date}},
                                             {"stop_word_fl" : {"$eq": "F"} } ]}},
                                {"$group": {"_id": {"word": '$word'}, "count": {"$sum": 1}}}]
                    select_word = self.c_tweetWords.aggregate(pipeline, allowDiskUse=True)
                    

                #if we are filtering by is_bot
                elif is_bot_Filter is not None:  #wrong                                      
                    bln_GetWords_From_Text = True
                    select_texts = self.c_focusedTweet.find(query_filter, 
                                                            { "text_combined_clean": 1, "id_str": 1 })
                

                #if there is no filter
                else:                        
                    pipeline = [{"$match": {"stop_word_fl" :  { "$eq": "F" }}}, 
                                {"$group": {"_id": {"word": '$word'}, "count": {"$sum": 1}}}]
                    
                    select_word = self.c_tweetWords.aggregate(pipeline, allowDiskUse=True)
                                                                                                    

            #in case we have an array of edges to filter by
            else:                
                
                #**************************************************************************************
                # Creating a temporary collection with all hashtags for each tweet for the given edges
                # This is possible without creating temp collections, but it was done this way to improve performance. 
                # Running with a different Collation can take a LONG time - 
                # (We need to run with Collation strength=2 to get canse insensitive counts )
                                
                #create temp collection for edges                
                self.create_tmp_edge_collections(arr_edges, arr_ht_edges, query_filter_for_edges)
                
                pipeline = [ {"$lookup":{
                                       "from": "focusedTweet",
                                       "localField": "tweet_id_str",
                                       "foreignField": "id_str",
                                       "as" : "focusedTweet"}},
                              {"$unwind": "$focusedTweet" },
                              {"$project": {"id_str": "$tweet_id_str", 
                                            "text_combined_clean": "$focusedTweet.text_combined_clean" }}]
                select_texts = self.c_tmpEdgesTweetIds.aggregate(pipeline, allowDiskUse=True)
                
                bln_GetWords_From_Text = True
                


            # If we want to create a tmp collection to save the words after spliting the words from text. 
            # (This was done this way to allow some performance improvements)
            # this option is being used when we are filtering by is_bot or by edges
            if bln_GetWords_From_Text == True:
                self.c_tmpEdgesWordFreq.delete_many({})                
                file_data = []                
                for x in select_texts:                         
                    for word in pos_tag(tokenizer.tokenize(x['text_combined_clean'])):
                        if word[0] not in stopWords:
                            data = '{"tweet_id_str":"' + x['id_str'] + \
                                    '", "word":"' + word[0] + '"}'
                            doc = json.loads(data)
                            file_data.append(doc)
                                            
                if file_data != []:
                    self.c_tmpEdgesWordFreq.insert_many(file_data)
                #**************************************************************************************                

                #getting counts for each word
                pipeline = [  {"$group": { "_id": { "word": '$word' }, "count": { "$sum": 1 } } }]                
                select_word = self.c_tmpEdgesWordFreq.aggregate(pipeline, allowDiskUse=True)
                
                

            #get data from database, loop through records and insert into array            
            for x in select_word:
                arr.append([x['_id']['word'], x['count']])

            #sort array in count descending order
            def sortSecond(val): 
                return val[1] 
            arr.sort(key=sortSecond,reverse=True) 
            
            arr = arr[:top_no_filter]
            
   
            #set file path
            file = filepath + 'T_Words_FrequencyList.txt'            
            
          
        
        #export text for topic analysis
        if (exportType == 'tweet_ids_timeseries'):                                                                  
                
            #in case we don't have an array of edges to filter by
            if arr_edges is None and arr_ht_edges is None:
                         
                select_ids = self.c_focusedTweet.find(query_filter, { "id_str": 1, "tweet_created_at": 1} )                                

            #in case we have an array of edges to filter by
            else:
                
                self.create_tmp_edge_collections(arr_edges, arr_ht_edges, query_filter_for_edges)                                
                
                if ht_to_filter is None:                
                    pipeline = [ {"$lookup":{
                                           "from": "focusedTweet",
                                           "localField": "tweet_id_str",
                                           "foreignField": "id_str",
                                           "as" : "focusedTweet"}},    
                                  {"$unwind": "$focusedTweet" },   
                                  {"$project": {"id_str": "$focusedTweet.id_str", 
                                                "tweet_created_at": "$focusedTweet.tweet_created_at" }}]
                else:
                    pipeline = [ {"$lookup":{
                                           "from": "focusedTweet",
                                           "localField": "tweet_id_str",
                                           "foreignField": "id_str",
                                           "as" : "focusedTweet"}},    
                                  {"$unwind": "$focusedTweet" },   
                                  {"$match": {"focusedTweet.hashtags.ht_lower": ht_to_filter.lower()} },
                                  {"$project": {"id_str": "$focusedTweet.id_str", 
                                                "tweet_created_at": "$focusedTweet.tweet_created_at" }}]
                    
                select_ids = self.c_tmpEdgesTweetIds.aggregate(pipeline, allowDiskUse=True)


            #get data from database, loop through records and insert into array
            for x in select_ids:
                arr.append([x['tweet_created_at'], x['id_str']]) 
                
            
            #set file path
            file = filepath + 'T_tweetIdswithDates.txt'
            
            
        #export tweetCountByUser
        if (exportType == 'tweetCount'):
            
            total_tweets = 0    
            total_retweets = 0
            total_replies = 0

            select_cTweet = self.c_focusedTweet.aggregate([{"$match" : {"retweeted_text" : {"$ne": ""} }}, 
                                                           {"$group": {"_id": {"seq_agg": "$seq_agg"}, 
                                                                       "count": { "$sum": 1 } } } ])
            for tweetCount in select_cTweet:   
                total_retweets = tweetCount["count"]     


            select_cTweet = self.c_focusedTweet.aggregate([{"$group": {"_id": {"seq_agg": "$seq_agg"}, 
                                                                       "count": { "$sum": 1 } } } ])
            for tweetCount in select_cTweet:            
                total_tweets = tweetCount["count"]


            select_cTweet = self.c_focusedTweet.aggregate([{"$match" : {"in_reply_to_screen_name" : {"$ne": "None"} }}, 
                                                           {"$group": {"_id": {"seq_agg": "$seq_agg"}, 
                                                                       "count": { "$sum": 1 } } } ])
            for tweetCount in select_cTweet:            
                total_replies = tweetCount["count"]

            arr.append([ 'Total Original Tweets', str(total_tweets-total_retweets-total_replies)])
            arr.append([ 'Total Replies', str(total_replies)])
            arr.append([ 'Total Retweets', str(total_retweets)])
            arr.append([ 'Total Tweets', str(total_tweets)])

            #set file path
            file = filepath + 'tweetCount.txt'
            
            
            
        #export tweetCountByUser
        if (exportType == 'userCount'):
                                   
            tweet_user_count = 0
            reply_user_count = 0
            quote_user_count = 0
            retweet_user_count = 0

            select_cTweet = self.c_users.aggregate( [{"$group": {"_id": {"user_type": "$user_type"}, "count": { "$sum": 1 } } } ])
            for tweetCount in select_cTweet:                   
                if tweetCount["_id"]["user_type"] == 'tweet':
                    arr.append(['1', tweetCount["_id"]["user_type"], 'Users with at least one document in this db', str(tweetCount["count"]) ])                                
                elif tweetCount["_id"]["user_type"] == 'retweet':
                    arr.append([ '2', tweetCount["_id"]["user_type"], 'Users that were retweeted, but are not part of previous group', str(tweetCount["count"]) ])
                elif tweetCount["_id"]["user_type"] == 'quote':
                    arr.append([ '3', tweetCount["_id"]["user_type"], 'Users that were quoted, but are not part of previous groups', str(tweetCount["count"]) ])                
                elif tweetCount["_id"]["user_type"] == 'reply':
                    arr.append([ '4', tweetCount["_id"]["user_type"], 'Users that were replied to, but are not part of previous groups', str(tweetCount["count"]) ])
                elif tweetCount["_id"]["user_type"] == 'mention':
                    arr.append([ '5', tweetCount["_id"]["user_type"], 'Users that were mentioned, but are not part of previous groups', str(tweetCount["count"]) ])
                else:
                    arr.append([ '6', tweetCount["_id"]["user_type"], '', str(tweetCount["count"]) ])    
            
            #set file path
            file = filepath + 'userCount.txt'
            
        

        #export tweetCountByUser
        if (exportType == 'tweetCountByUser'):

            #set header of txt file
            arr.append([ 'user_id', 'user_screen_name', 'count'])

            #get data from database and loop through records and insert into array
            select_tweetCountByUser = self.c_tweetCountByUserAgg.find()        
            for x in select_tweetCountByUser:
                arr.append([ x['user_id'], x['user_screen_name'],  x['count']])        

            #set file path
            file = filepath + 'tweetCountByUser.txt'
            
            

        
        #export tweetCountByLanguage
        if (exportType == 'tweetCountByLanguage'):

            #set header of txt file
            arr.append([ 'lang', 'count'])

            #get data from database and loop through records and insert into array
            select_tweetCountByLang = self.c_tweetCountByLanguageAgg.find()        
            for x in select_tweetCountByLang:
                arr.append([ x['lang'],  x['count']])

            #set file path
            file = filepath + '\\tweetCountByLanguage.txt'
            
                    

        
        #export tweetCountByFile
        if (exportType == 'tweetCountByFile'):

            #set header of txt file
            arr.append([ 'file_path', 'count'])

            #get data from database and loop through records and insert into array
            select_tweetCountByFile = self.c_tweetCountByFileAgg.find()        
            for x in select_tweetCountByFile:
                arr.append([ x['file_path'],  x['count']])        

            #set file path
            file = filepath + 'tweetCountByFile.txt'



        #export tweetCountByMonth
        if (exportType == 'tweetCountByMonth'):

            #set header of txt file
            arr.append([ 'year', 'month_no', 'count'])   

            #get data from database and loop through records and insert into array
            select_tCountByPeriod = self.c_tweetCountByPeriodAgg.find()        
            for x in select_tCountByPeriod:
                arr.append([ x['year'], x['month_no'], x['count']])         

            #set file path
            file = filepath + 'tweetCountByMonth.txt'        



        #export hashtagCount
        if (exportType == 'hashtagCount'): 

            #set header of txt file
            arr.append([ 'hashtag', 'count'])            

            #get data from database and loop through records and insert into array
            select_hashtagCountByDay = self.c_hashTagCountAgg.find()        
            for x in select_hashtagCountByDay:
                arr.append([ x['hashtag'],  x['count']])

            #set file path
            file = filepath + 'hashtagCount.txt'


            
        #export topics by hashtag
        if (exportType == 'topicByHashtag'): 

            #set header of txt file
            arr.append([ 'ht', 'ht_count', 'lib', 'model', 'no_words', 'topic_no', 'topic'])       

            #get data from database and loop through records and insert into array
            select_cHTTopics = self.c_htTopics.find()        
            for x in select_cHTTopics:
                arr.append([ x['ht'], x['ht_count'],  x['lib'],  x['model'],  
                             x['no_tweets'],  x['topic_no'],  x['topic']])

            #set file path
            file = filepath + 'topicByHashtag.txt'


        #export tweetTextAndPeriod
        if (exportType == 'tweetTextAndPeriod'):

            i = 0                

            #get data from database and loop through records and insert into array
            select_focusedTweet = self.c_focusedTweet.find() 
            for x in select_focusedTweet:

                if (i % inc == 0 and i != 0):                                                
                    self.exportToFile(arr, file) #export in array into txt file

                if (i==0 or i % inc==0):
                    arr = []
                    file = filepath + 'tweetTextAndPeriod_' + str(i) + '.txt' #set file path
                    arr.append([ 'text', 'text_lower', 'year', 'month_no', 'day', 'user_id'])

                arr.append([ x['text'], x['text_lower'], x['year'],  
                             x['month_no'],  x['day'],  x['user_id']])

                i = i +1
                
                
        #export tweetDetails
        if (exportType == 'tweetDetails'):

            i = 0                

            #get data from database and loop through records and insert into array
            select_focusedTweet = self.c_focusedTweet.find() 
            for x in select_focusedTweet:

                if (i % inc == 0 and i != 0):                                                
                    self.exportToFile(arr, file) #export in array into txt file


                if (i==0 or i % inc==0):                
                    arr = []
                    file = filepath + 'tweetTextAndPeriod_' + str(i) + '.txt' #set file path
                    arr.append([ 'text', 'text_lower', 'year', 'month_no', 'day', 'user_id'])

                arr.append([ x['text'], x['text_lower'], x['year'],
                             x['month_no'],  x['day'],  x['user_id']])

                i = i +1   
                


        #export words
        if (exportType == 'wordsOnEachTweet'):  

            i = 0                

            #get data from database
            select_tweetWords = self.c_tweetWords.find()
            for x in select_tweetWords:

                if (i % inc == 0 and i != 0):                                                
                    self.exportToFile(arr, file) #export in array into txt file                

                if (i==0 or i % inc==0):                
                    arr = []
                    file = filepath + 'wordsOnEachTweet_' + str(i)  + '.txt' #set file path
                    arr.append(['word_orig', 'word', 'word_lower', 'word_tag', 'word_lemm', 
                                'id_str', 'text', 'seq_no_tweet', 'seq_no'])


                arr.append([ x['word_orig'],  x['word'],  x['word_lower'],  x['word_tag'],  
                             x['word_lemm'],  x['id_str'],  x['text'],  x['seq_no_tweet'],  x['seq_no']])

                i = i +1



        #user details on Each Tweet
        if (exportType == 'userDetailsOnEachTweet'):

            i = 0                

            #get data from database
            select_Tweet = self.c_tweet.find()
            for tweet in select_Tweet:

                if (i % inc == 0 and i != 0):                                                
                    self.exportToFile(arr, file) #export in array into txt file                

                if (i==0 or i % inc==0):
                    arr = []
                    file = filepath + 'userDetailsOnEachTweet_' + str(i)  + '.txt' #set file path
                    arr.append(['id_str', 'user_id', 'user_location', 'user_name', 
                                'user_screen_name', 'user_description', 'user_verified', 
                                'user_followers_count', 'user_friends_count', 
                                'user_statuses_count', 'user_created_at', 'user_time_zone', 
                                'user_lang', 'user_geo_enabled'])


                #get relevant information from tweet
                id_str = tweet['id_str'] 
                user_id = tweet['user']['id_str']
                user_location = tweet['user']['location']
                user_name = tweet['user']['name']
                user_screen_name = tweet['user']['screen_name']
                user_description = tweet['user']['description']                                
                user_verified = tweet['user']['verified']
                user_followers_count = tweet['user']['followers_count']
                user_friends_count = tweet['user']['friends_count']
                user_statuses_count = tweet['user']['statuses_count']
                user_created_at = tweet['user']['created_at']
                user_time_zone = tweet['user']['time_zone']
                user_lang = tweet['user']['lang']        
                user_geo_enabled = tweet['user']['geo_enabled']        

                if user_description is not None:            
                    user_description = user_description.replace("|", "").strip().replace("\n", "").replace("\r", "")

                if user_location is not None:            
                    user_location = user_location.replace("|", "").strip().replace("\n", "").replace("\r", "")

                if user_name is not None:        
                    user_name = user_name.replace("|", "").strip().replace("\n", "").replace("\r", "")

                if user_screen_name is not None: 
                    user_screen_name = user_screen_name.replace("|", "").strip().replace("\n", "").replace("\r", "")


                arr.append([id_str, user_id, user_location, user_name, user_screen_name, 
                            user_description, user_verified, user_followers_count, 
                            user_friends_count, user_statuses_count, 
                            user_created_at, user_time_zone, user_lang, user_geo_enabled])  

                i = i +1    

        #export in array into txt file
        #self.exportToFile(arr, file)
        return arr, file
    
    
    
    #####################################
    # Method: exportToFile
    # Description: Method used to export an array to a t\ delimited file
    # Parameters: arrData = the array with the data you want to export
    # file = the path and name of the file you want to export
    def exportToFile(self, arrData, file):         
        
        myFile = open(file, 'w', encoding="utf-8")
        with myFile:
            writer = csv.writer(myFile, delimiter='\t', lineterminator='\n')
            writer.writerows(arrData)
                                    
        
    
    
    ######### Topic Analysis ###############################################
    # *This was just an initital analysis. refer to pyTwitterTopics for more.
        
    #####################################
    # Method: get_docs
    # Description: create one array with all tweets of one hashtag for topic analysis
    def get_docs(self, ht, max_doc_ctn):    
        
        ctn=0
        doc = ""
        topic_doc_complete.append(doc)
        
        select_cTweet = self.c_focusedTweet.find({"hashtags.ht_lower" : ht }) 
        #loop through tweets
        for tweet in select_cTweet:     
            if ctn < max_doc_ctn:
                doc = tweet['text_lower']
                topic_doc_complete.append(doc)
            ctn=ctn+1    
            
    
    #####################################
    # Method: clean_1
    # Description: clean documents for topic analysis
    def clean_1(self, doc): 
        stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
        punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
        normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
        return normalized
    
    
    #topic analysis using gensim model 
    def gensim_model(self, num_topics_lda, num_topics_lsi, ht, tc):

        import gensim
        from gensim import corpora

        doc_clean = [self.clean_1(doc).split() for doc in topic_doc_complete]   

        # Creating the term dictionary of our courpus, where every unique term is assigned an index. dictionary = corpora.Dictionary(doc_clean)
        dictionary = corpora.Dictionary(doc_clean)            

        # Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
        doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

        # Creating the object for LDA model using gensim library
        Lda = gensim.models.ldamodel.LdaModel    

        # Build the LDA model
        lda_model = gensim.models.LdaModel(corpus=doc_term_matrix, num_topics=num_topics_lda, id2word=dictionary)    

        # Build the LSI model
        lsi_model = gensim.models.LsiModel(corpus=doc_term_matrix, num_topics=num_topics_lsi, id2word=dictionary)    


        file_data = []     
        for idx in range(num_topics_lda):                
            topic = idx+1
            strtopic = str(topic)

            data = '{"ht":"' + ht + \
                    '", "ht_count":"' + str(tc) + \
                    '", "lib":"' + "gensim" + \
                    '", "model":"' + "lda" + \
                    '", "no_tweets":"' + str(tc) + \
                    '", "topic_no":"' + strtopic + \
                    '", "topic":"' + str(lda_model.print_topic(idx, num_topics_lda)).replace('"', "-") + '"}'

            x = json.loads(data)
            file_data.append(x)



        for idx in range(num_topics_lsi):        
            data = '{"ht":"' + ht + \
                '", "ht_count":"' + str(tc) + \
                '", "lib":"' + "gensim" + \
                '", "model":"' + "lsi" + \
                '", "no_tweets":"' + str(tc) + \
                '", "topic_no":"' + str(idx+1) +\
                '", "topic":"' + str(lsi_model.print_topic(idx, num_topics_lsi)).replace('"', "-") + '"}'

            x = json.loads(data)
            file_data.append(x)


        self.c_htTopics.insert_many(file_data)        
        

            
    #topic analysis using sklearn model 
    def skl_model(self, num_topics_lda, num_topics_lsi, num_topics_nmf, ht, tc):    

        vectorizer = CountVectorizer(min_df=0.009, max_df=0.97, stop_words='english', lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')

        data_vectorized = vectorizer.fit_transform(topic_doc_complete)

        # Build a Latent Dirichlet Allocation Model
        lda_model = LatentDirichletAllocation(n_components=num_topics_lda, max_iter=5,learning_method='online',learning_offset=50.,random_state=0)
        lda_Z = lda_model.fit_transform(data_vectorized)

        # Build a Non-Negative Matrix Factorization Model
        nmf_model = NMF(num_topics_nmf)
        nmf_Z = nmf_model.fit_transform(data_vectorized)

        # Build a Latent Semantic Indexing Model
        lsi_model = TruncatedSVD(1)
        lsi_Z = lsi_model.fit_transform(data_vectorized)


        file_data = []

        for idx, topic in enumerate(lda_model.components_):  
            topic = str([( str(topic[i]) + "*" + vectorizer.get_feature_names()[i] + " + " )
                            for i in topic.argsort()[:-num_topics_lda - 1:-1]]).replace("[", "").replace("]", "").replace("'", "").replace(",", "")

            data = '{"ht":"' + ht + \
                '", "ht_count":"' + tc + \
                '", "lib":"' + "sklearn" + \
                '", "model":"' + "lda" + \
                '", "no_tweets":"' + str(tc) + \
                '", "topic_no":"' + str(idx+1) +\
                '", "topic":"' + topic + '"}'

            x = json.loads(data)
            file_data.append(x)



        for idx, topic in enumerate(lsi_model.components_):  
            topic = str([( str(topic[i]) + "*" + vectorizer.get_feature_names()[i] + " + " )
                            for i in topic.argsort()[:-num_topics_lsi - 1:-1]]).replace("[", "").replace("]", "").replace("'", "").replace(",", "")

            data = '{"ht":"' + ht + \
                '", "ht_count":"' + tc + \
                '", "lib":"' + "sklearn" + \
                '", "model":"' + "lsi" + \
                '", "no_tweets":"' + str(tc) + \
                '", "topic_no":"' + str(idx+1) +\
                '", "topic":"' + topic + '"}'

            x = json.loads(data)
            file_data.append(x)




        for idx, topic in enumerate(nmf_model.components_):  
            topic = str([( str(topic[i]) + "*" + vectorizer.get_feature_names()[i] + " + ")
                            for i in topic.argsort()[:-num_topics_nmf - 1:-1]]).replace("[", "").replace("]", "").replace("'", "").replace(",", "")

            data = '{"ht":"' + ht + \
                '", "ht_count":"' + tc + \
                '", "lib":"' + "sklearn" + \
                '", "model":"' + "nmf" + \
                '", "no_tweets":"' + str(tc) + \
                '", "topic_no":"' + str(idx+1) +\
                '", "topic":"' + topic + '"}'

            x = json.loads(data)
            file_data.append(x)        


        self.c_htTopics.insert_many(file_data)            
        
        

    #find topics for each hashtag  
    def findTopics(self, num_topics_lda, num_topics_lsi, num_topics_nmf, max_no_tweets_perHT, model):

        starttime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print ("loading process started...." + starttime)
        
        #find all hashtag and their count
        select_cHashtagCount = self.c_hashTagCountAgg.find().sort("count", -1)
                

        try:
            #loop through hashtags
            for tweet in  select_cHashtagCount: 

                ht = tweet['hashtag']
                count = tweet['count']

                if ht != "metoo" and count > 500:                                

                    #get all tweets for that hashtag
                    topic_doc_complete.clear()
                    self.get_docs(ht, max_no_tweets_perHT) 

                    #run topic models
                    try:
                        if model == "gensim":
                            self.gensim_model(num_topics_lda, num_topics_lsi, ht, str(count))        
                        elif model == "sklearn":
                            self.skl_model(num_topics_lda, num_topics_lsi, num_topics_nmf, ht, str(count))                 
                    except Exception as e:                                      
                        print("Error finding topics for hashtag " + ht + ", using model " + model +". Err msg: " + str(e)) 
                        continue  
        
        except Exception as e:
            print("Error finding topics. Err msg: " + str(e))             
                    

        endtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print ("loading process completed. " + endtime)
        
     
    # search7dayapi
    def search7dayapi(
            self, 
            consumer_key, 
            consumer_secret, 
            access_token, 
            access_token_secret, 
            query, 
            result_type= 'mixed', 
            max_count='100', 
            lang='en'):
        """
        Send requests to the 7-Day search API and save data into MongoDB
                
        Parameters
        ----------               
        consumer_key : 
            User's consumer key               
            
        consumer_secret :
            User's consumer secret
        
        access_token : 
            User's access token

        access_token_secret : 
            User's access token secret

        query :
            The query that will be used to filter the tweets
            
        result_type : 
            Options: recent, popular, or mixed
            
        max_count : 
            The number of tweets to be returned at a time
            
        lang :     
            Language to filter the tweets
            
        Returns
        -------
        response 
            the response received from Twitter, which will contain either the tweets retrieved from the search, or the error message if any      
    
        Examples
        --------                  
            >>> # send request to 7-day search API
            >>> response = myAnalysisObj.search7dayapi(
            >>>     consumer_key = '[key]',
            >>>     consumer_secret = '[secret]',
            >>>     access_token = '[token]',
            >>>     access_token_secret = '[token_secret]',
            >>>     query = 'austintexas OR atx OR austintx OR atxlife',
            >>>     result_type = 'mixed',
            >>>     max_count = '100',
            >>>     lang = 'en')    
        """       
        
        
        aut = OAuth1(consumer_key, consumer_secret, access_token, access_token_secret)
        
        endpoint = 'https://api.twitter.com/1.1/search/tweets.json?q=' + query + '&count=' + max_count + '&lang=' + lang + '&include_entities=true&tweet_mode=extended&result_type=' + result_type + ''
        
        response = requests.get(endpoint, auth=aut).json()        
        
        # if there was an error, print error and end method
        if 'error' in response:            
            print (response)
            return ''
        
        tweets = json.loads(json.dumps(response, indent = 2))
        
        if 'search_metadata' in tweets:
            search = tweets['search_metadata']
        else:
            search = {}
            
        # insert tweets into DB
        self.insertTweetToDBFromAPI(tweets, 'statuses', search, '7day')
            
        return response
    
    
    
    # searchPremiumAPI
    def searchPremiumAPI(self, 
            twitter_bearer, 
            api_name, 
            dev_environment,
            query, 
            date_start, 
            date_end, 
            next_token = None, 
            max_count='100'):
            
        """
        Send requests to the Premium search API and save data into MongoDB
                
        Parameters
        ----------               
        twitter_bearer : 
            bearer authentication token created from the consumer_key and consumer_secret
        
        api_name :
            the options are either 30day or FullArchive
        
        dev_environment :
            the name of the environment created on the Twitter's developer account
        
        query :
            the query that will be used to filter the tweets
            
        date_start : 
            the start date that will be used to filter the tweets.
        
        date_end :            
            the end date that will be used to filter the tweets.
            
        next_token :             
            then token that points to the previous search done with the same query.
            
        max_count : 
            the number of tweets to be returned at a time
            
            
        Returns
        -------
        response 
            the response received from Twitter, which will contain either the tweets retrieved from the search, or the error message if any      
        
        next_token
            token value that can be used for the next request, that way it is possible to avoid searches for the same records
    
        Examples
        --------                  
            >>> # send request to premium API
            >>> response, next_token = myAnalysisObj.searchPremiumAPI(
            >>>     twitter_bearer = '[bearer]',
            >>>     api_name = '30day',
            >>>     dev_environment = 'myDevEnv.json',
            >>>     query = '(coronavirus OR COVID19) lang:en',
            >>>     date_start = '202002150000',
            >>>     date_end = '202002160000',
            >>>     next_token = None,
            >>>     max_count = '100'
            >>> )   
        """              
        
        headers = {"Authorization":"Bearer " + twitter_bearer + "", "Content-Type": "application/json"}  
        endpoint = "https://api.twitter.com/1.1/tweets/search/" + api_name + "/" + dev_environment
      
        if next_token is None:
            data = '{"query":"' + query + '","fromDate":"' + date_start + '","toDate":"' + date_end + '", "maxResults":"' + max_count + '"}' 
        else:
            data = '{"query":"' + query + '","fromDate":"' + date_start + '","toDate":"' + date_end +'", "next":"' + next_token + '", "maxResults":"' + max_count + '"}'
            
        response = requests.post(endpoint,data=data,headers=headers).json()
        
        # if there was an error, print error and end method
        if 'error' in response:            
            print (response)
            return ''
    
        # load tweets into a json variable
        tweets = json.loads(json.dumps(response, indent = 2)) 
                
        #Get "next"token
        if 'next' in tweets:
            next_token = tweets['next']
        else:
            next_token = ""            

        # save what information was used for this search
        search = json.loads(data)
        a_dict = {'next': next_token}                
        search.update(a_dict)                
            
        # insert tweets into DB
        self.insertTweetToDBFromAPI(tweets, 'results', search, api_name)
            
        return response, next_token

            
    # insertTweetToDB
    def insertTweetToDBFromAPI(self, tweets, parent_field, search, api):
         
        seq_no = 0
        select_cTweet = self.c_tweet.aggregate( [{"$group": {"_id": "seq_agg" , "count": { "$max": "$seq_no" } } } ])
        for tweetCount in select_cTweet:
            seq_no = tweetCount["count"]
    
        if parent_field in tweets:
            
            #Insert into searches table
            a_dict = {'search_time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'api': api}                
            search.update(a_dict)            
            result = self.c_searches.insert_one(search)

            #Insert into tblTweet table
            for tweet in tweets[parent_field]:

                seq_no = seq_no + 1

                #adding extra fields to document to suport future logic (processed_fl, load_time, file_path )
                a_dict = {'processed_fl': 'N', 'seq_no': seq_no, 'seq_agg': "A", 'load_time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}                
                tweet.update(a_dict)

                try:
                    result = self.c_tweet.insert_one(tweet)
                except:
                    result = "" 
                    
                    

    # create python script and .bat file for scheduled processing
    def create_bat_file_apisearch(self, 
            mongoDBServer,
            mongoDBName,
            file_path,
            python_path,
            consumer_key,
            consumer_secret,
            access_token,
            access_token_secret,
            query,
            result_type='mixed', 
            max_count='100',
            lang='en'):
            
        """
        The method will create two files, one python script containing the code necessary to make the requests, 
        and a *.bat* file that can be used to schedule the call of the python script.
                
        Parameters
        ----------  
        mongoDBServer :
            the mongoDB server that will be used to save the tweets
        
        mongoDBName :
            the mongoDB name that will be used to save the tweets
        
        file_path :
            the folder path where the files will be saved
        
        python_path :        
            the path where the python.exe is installed
            
        consumer_key : 
            User's consumer key               
            
        consumer_secret :
            User's consumer secret
        
        access_token : 
            User's access token

        access_token_secret : 
            User's access token secret

        query :
            The query that will be used to filter the tweets
            
        result_type : 
            Options: recent, popular, or mixed
            
        max_count : 
            The number of tweets to be returned at a time
            
        lang :     
            Language to filter the tweets                  
            
        Examples
        --------                  
            >>> create python script and .bat file for scheduled processing
            >>> create_bat_file_apisearch( 
            >>>         mongoDBServer='mongodb://localhost:27017',
            >>>         mongoDBName='myDBName',
            >>>         file_path='C:\\Data\\myScriptsFolder\\MyScriptName.py',
            >>>         python_path='C:\\Users\\Me\Anaconda3\envs\myEnv\python.exe',
            >>>         consumer_key = '[key]',
            >>>         consumer_secret = '[secret]',
            >>>          access_token = '[token]',
            >>>         access_token_secret = '[token_secret]',
            >>>         query = 'austintexas OR atx OR austintx OR atxlife',
            >>>         result_type = 'mixed',
            >>>         max_count = '100',
            >>>         lang = 'en')
        """    
        
    
        # create path is does not exist
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))
                        
        f = open(file_path, "w")        
        
        f.write("import json\n")
        f.write("import requests\n")
        f.write("from pymongo import MongoClient\n")
        f.write("from requests_oauthlib import OAuth1\n")
        f.write("import datetime\n")
        f.write("\n")

        f.write("mongoDBServer = '" + mongoDBServer + "'\n")
        f.write("client = MongoClient(mongoDBServer)\n")
        f.write("db = client." + mongoDBName + "\n")
        f.write("\n")
        
        f.write("# Create unique index on tweet table to make sure we don't store duplicate tweets\n")
        f.write("try:\n")
        f.write("    resp = self.c_tweet.create_index([('id', pymongo.ASCENDING)],unique = True)\n")
        f.write("except:\n")
        f.write("    pass\n")
        f.write("\n")        
        
        f.write("aut = OAuth1(" + "'" + consumer_key + "'" + ","  + "'" + consumer_secret  + "'" + ","  + "'" + access_token + "'" + ","  + "'" + access_token_secret  + "'" + ")\n")
        f.write("\n")
        
        endpoint = 'https://api.twitter.com/1.1/search/tweets.json?q=' + query + '&count=' + max_count + '&lang=' + lang + '&include_entities=true&tweet_mode=extended&result_type=' + result_type + ''
        f.write("endpoint = " + "'" + endpoint + "'\n") 
        f.write("\n")
        
        f.write("response = requests.get(endpoint, auth=aut).json()")
        f.write("\n")
        
        f.write("# if there was an error, print error and end method\n")
        f.write("if 'error' in response:\n")
        f.write("    print (response)\n")        
        f.write("    \n")
        f.write("tweets = json.loads(json.dumps(response, indent = 2))\n")
        f.write("\n")
        f.write("search = tweets['search_metadata']\n")
        f.write("\n")       
        f.write("seq_no = 0\n")
        f.write("select_cTweet = db.tweet.aggregate( [{'$group': {'_id': 'seq_agg' , 'count': { '$max': '$seq_no' } } } ])\n")
        f.write("for tweetCount in select_cTweet:\n")
        f.write("    seq_no = tweetCount['count']\n")
        f.write("    \n")
        f.write("if 'statuses' in tweets:\n")
        f.write("        \n")
        f.write("    #Insert into searches table\n")
        f.write("    a_dict = {'search_time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'api': '7day'}\n")                
        f.write("    search.update(a_dict)\n")   
        f.write("    result = db.searches.insert_one(search)\n")
        f.write("    \n")
        f.write("    #Insert into tblTweet table\n")
        f.write("    for tweet in tweets['statuses']:\n")
        f.write("    \n")
        f.write("        seq_no = seq_no + 1\n")
        f.write("        \n")
        f.write("        #adding extra fields to document to suport future logic (processed_fl, load_time, file_path )\n")
        f.write("        a_dict = {'processed_fl': 'N', 'seq_no': seq_no, 'seq_agg': 'A', 'load_time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("        tweet.update(a_dict)\n")
        f.write("        \n")
        f.write("        try:\n")
        f.write("            result = db.tweet.insert_one(tweet)\n")
        f.write("        except:\n")
        f.write("            result = ''\n")
                    
        f.close()
        
        print(python_path)
        fbat = open(os.path.dirname(file_path) + '\\twitter_request_script.bat', "w")        
        fbat.write('start ' + python_path + ' "' + file_path + '"')               
        fbat.close()



import os
import csv
import datetime
import networkx as nx
import numpy as np
import numpy.linalg as la
import community as community_louvain
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from networkx import algorithms
from networkx.algorithms import distance_measures
from networkx.algorithms.components import is_connected
from networkx.algorithms.dominance  import immediate_dominators
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn import metrics
import math

import scipy as sp
from scipy.sparse import csgraph
import scipy.cluster.vq as vq
import scipy.sparse.linalg as SLA
    
import pandas as pd
import seaborn as sns
    
    
class TwitterGraphs:   

    def __init__(self, folder_path):
        
        #creates path if does not exists
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
    
        self.folder_path = folder_path        
        self.graph_details_file = folder_path + "\\log_graph_plots.txt"
        
        f = open(self.graph_details_file, "a")
        f.write('graph_name\t scale\t k\t iteration\t kmeans_k\t starttime\t endtime\t'  + '\n')
        f.close()
            

    #####################################
    # Method: loadGraphFromFile
    # Description: method receives nodes and edges files and returns an nwtworkx 
    # Parameters: 
    #   -nodes_file = file path and file name for netwrok nodes
    #   -edge_file = file path and file name for netwrok edges with weight
    def loadGraphFromFile(self, edge_file):
        
        G = nx.Graph()
        G = nx.read_edgelist(edge_file, data=(('weight',float),))

        return G

        
    # function to plot network graph
    # Parameters: G - NetworkX graph
    #             v_graph_name - The name of your graph
    #             v_scale - Scale factor for positions. The nodes are positioned in a box of size [0,scale] x [0,scale].
    #             v_k - Optimal distance between nodes - Increase this value to move nodes farther apart.
    #             v_iterations - Number of iterations of spring-force relaxation
    #             cluster_fl - determines if you are sending labels for clusters or not
    #             v_labels - cluster labels
    #             kmeans_k - k_means k used for clustering
    #             node_color - node color, default '#A0CBE2'
    #             edge_color - edge color, default '#A79894'
    #             width - with, default 0.05
    #             node_size - node size, default 0.6
    #             font_size - font size, default 1
    #             dpi - size of the image in dpi, default 800
    # More details at: https://networkx.github.io/documentation/networkx-1.9/reference/generated/networkx.drawing.layout.spring_layout.html
    '''
    def plotSpringLayoutGraph(self, G, v_graph_name, v_scale, v_k, v_iterations, 
                              cluster_fl='N', v_labels=None, kmeans_k='', v_node_color='#A0CBE2', v_edge_color='#A79894', 
                              v_width=0.05, v_node_size=0.6, v_font_size=1, v_dpi=900):
    '''
    
        

    #####################################
    # Method: plot_graph_att_distr
    # Description: Plot distribution of nodes based on graph attribute (e.g. communitiy)
    def plot_graph_att_distr(self, G, att, title='Community Counts', xlabel='Community ID', ylabel='Count', file_name=None, replace_existing_file=True):
        #create dataframe based on the given attribute
        df = pd.DataFrame.from_dict(nx.get_node_attributes(G, att),  orient='index')
        df.columns = [att] 
        df.index.rename('node' , inplace=True)
        sns.distplot(df[att], kde=False, bins=100)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        #if file name was give, save file in default folder
        if file_name != None:
            if replace_existing_file==True or not os.path.exists(file_name):
                plt.savefig(file_name)
            
        plt.show()
        plt.cla()   # Clear axis
        plt.clf()   # Clear figure
        plt.close() # Close a figure window
        

        
    #####################################
    # Method: plot_disconnected_graph_distr
    # Description: plot the distribution of disconnected graphs
    def plot_disconnected_graph_distr(self, G, file=None, replace_existing_file=True, size_cutoff=None):
    
        sub_conn_graphs = sorted(list(nx.connected_component_subgraphs(G)), key = len, reverse=True)
    
        if size_cutoff is not None:
            sub_conn_graphs2 = sub_conn_graphs.copy()
            sub_conn_graphs = []

            for x in sub_conn_graphs2:
                if len(x.nodes()) > size_cutoff:
                    sub_conn_graphs.append(x)
                
                
        x = []
        y = []
        for i, a in enumerate(sub_conn_graphs):
            x.append(len(a.nodes()))
            y.append(len(a.edges()))

        fig, axs = plt.subplots(2, 3,figsize=(16,10))
        
        try:
            axs[0, 0].plot(x, y, 'ro'); axs[0, 0].set_title('All subgraphs')
            x.pop(0); y.pop(0); axs[0, 1].plot(x, y, 'ro'); axs[0, 1].set_title('Excluding top1')
            x.pop(0); y.pop(0); axs[0, 2].plot(x, y, 'ro'); axs[0, 2].set_title('Excluding top2')
            x.pop(0); y.pop(0); axs[1, 0].plot(x, y, 'ro'); axs[1, 0].set_title('Excluding top3')
            x.pop(0); y.pop(0); axs[1, 1].plot(x, y, 'ro'); axs[1, 1].set_title('Excluding top4')
            x.pop(0); y.pop(0); axs[1, 2].plot(x, y, 'ro'); axs[1, 2].set_title('Excluding top5')
        except Exception as e:            
                print("Warning: could not plot all 6  - " +str(e))
                pass 
        
        for ax in axs.flat:
            ax.set(xlabel='Nodes', ylabel='Edges')
            
        #if file name was give, save file in default folder
        if file != None:
            if replace_existing_file==True or not os.path.exists(file):
                plt.savefig(file)
            
        plt.show()
        plt.cla()   # Clear axis
        plt.clf()   # Clear figure
        plt.close() # Close a figure window
        
        return len(sub_conn_graphs)

    
    #####################################
    # Method: contract_nodes_commty_per
    # Description: reduce graph based on a percentage given for each community found
    def contract_nodes_commty_per(
            self,
            G, 
            perc, 
            comm_att='community_louvain', 
            enforce_ext_nodes_conn_fl ='N', 
            commty_already_calculated='N'):

        G_to_contract = G.copy()        
        all_nodes = []

        #if we need to calculate the communities because the original report doesn't have the community labels
        if commty_already_calculated == 'N':
            G_to_contract, labels, k = self.calculate_louvain_clustering(G_to_contract)


        #find the number of communities in the graph
        no_of_comm = max(nx.get_node_attributes(G_to_contract, comm_att).values())+1    

        #loop through the communities and get the top nodes for each communities based on the given percentage
        for commty in range(no_of_comm):

            #find subgraphs of this community
            com_sub_graph = G_to_contract.subgraph([n for n,attrdict in G_to_contract.node.items() if attrdict [comm_att] == commty ])
            arr_nodes = np.array(sorted(com_sub_graph.degree(), key=lambda x: x[1], reverse=True))

            #get the comunity size and calculate how many top nodes we want to use based on the given percentage
            comm_size = len(com_sub_graph)        
            top_nodes = math.ceil(comm_size*(1-(perc/100)))
            if top_nodes == 1: top_nodes=top_nodes+1
            arr_top_nodes = arr_nodes[:top_nodes,0]

            if enforce_ext_nodes_conn_fl == 'Y':
                #create subgraph including external edges
                G_W_Ext_Edges = G_to_analyze.edge_subgraph(G_to_analyze.edges(com_sub_graph.nodes()))    
                #find the nodes in this community with external edges
                G_edges_Rem = G_W_Ext_Edges.copy()
                G_edges_Rem.remove_edges_from(com_sub_graph.edges())
                nodes_w_ext_edges = G_edges_Rem.edge_subgraph(G_edges_Rem.edges()).nodes()                            
                arr_top_nodes = np.concatenate((arr_top_nodes, nodes_w_ext_edges))

            all_nodes = np.concatenate((all_nodes, arr_top_nodes))

        #Create graph with only the contracted nodes
        G_Contracted = G_to_contract.subgraph(all_nodes)
        G_Contracted = self.largest_component_no_self_loops(G_Contracted)

        return G_Contracted


    #####################################
    # Method: draw_scaled_labels
    # Description: draw labels in the graphs
    def draw_labels_for_node(self, G, nodes, size, pos):  
        labels = {}        
        for node in G.nodes():
            if node in nodes:
                #set the node name as the key and the label as its value 
                labels[node] = node

        nx.draw_networkx_labels(G, pos, labels, font_size=size)


    #####################################
    # Method: draw_scaled_labels
    # Description: draw labels in the graph scaled by node degree
    def draw_scaled_labels(self, G, pos, default_font_size, font_size_multiplier):
        #get array of nodes in sorted order
        arr = np.array(sorted(G.degree(), key=lambda x: x[1], reverse=True))    
        #get the value of the highest degree. We will use this as reference to calculate the fonte sizes
        top_value = int(arr[:1,1])

        nodes_with_same_font_size = []
        for node, degree in arr:    

            #calculate the font size for this node
            size_scale = (int(degree) * 1) / top_value
            new_font_size = size_scale*font_size_multiplier

            #if the calculate font size is greater than the parameter give, print that label. If not add to the array of nodes with the default size.
            if new_font_size > default_font_size:
                self.draw_labels_for_node(G, node, new_font_size, pos)
            else:
                nodes_with_same_font_size.append(node)

        #Print labels for all nodes with the default size.
        self.draw_labels_for_node(G, nodes_with_same_font_size, default_font_size, pos)


    #####################################
    # Method: plotSpringLayoutGraph
    # Description: plot graph
    def plotSpringLayoutGraph(
            self, 
            G,  
            v_graph_name,  
            v_scale,  
            v_k,  
            v_iterations, 
            cluster_fl='N',  
            v_labels=None,  
            kmeans_k='',  
            v_node_color='#A0CBE2',  
            v_edge_color='#A79894', 
            v_width=0.05,  
            v_node_size=0.6,  
            v_font_size=0.4,  
            v_dpi=900,  
            v_alpha=0.6,  
            v_linewidths=0.6,
            scale_node_size_fl='Y',  
            draw_in_mult_steps_fl='N',  
            node_size_multiplier=6,  
            font_size_multiplier=7, 
            replace_existing_file=True):


        if replace_existing_file==True or not os.path.exists(v_graph_name):
            
            v_with_labels = True
            if scale_node_size_fl == 'Y':
                d = dict(G.degree)
                v_node_size = [(v * node_size_multiplier)/10 for v in d.values()]
                v_with_labels = False

            #node_color=v_labels,
            #node_color=v_node_color,
            #draw graph
            pos=nx.spring_layout(G, scale=v_scale, k=v_k, iterations=v_iterations)   #G is my graph
            if cluster_fl == 'N':                    
                nx.draw(G, pos,                 
                        width=v_width,                          
                        edge_color=v_edge_color,
                        node_color=v_node_color,
                        edge_cmap=plt.cm.Blues, 
                        with_labels=v_with_labels, 
                        node_size=v_node_size, 
                        font_size=v_font_size,                  
                        linewidths=v_linewidths,
                        alpha=v_alpha)
            else:         
                nx.draw(G, pos,
                        node_color=v_labels,
                        edge_color=v_edge_color,
                        width=v_width,
                        cmap=plt.cm.viridis,
                        edge_cmap=plt.cm.Purples,
                        with_labels=v_with_labels,
                        node_size=v_node_size,
                        font_size=v_font_size,
                        linewidths=v_linewidths,
                        alpha=v_alpha)


            # draw labels - logic to print labels in nodes in case we 
            # want to change the font size to match the scale of the node size
            if scale_node_size_fl == 'Y':
                self.draw_scaled_labels(G, pos, v_font_size, font_size_multiplier)           


            plt.savefig(v_graph_name, dpi=v_dpi, facecolor='w', edgecolor='w')

            plt.show()
            plt.cla()   # Clear axis
            plt.clf()   # Clear figure
            plt.close() # Close a figure window
   


        
    #####################################
    # Method: largest_component_no_self_loops
    # Description: remove self loops nodes, isolate nodes and exclude smaller components
    def largest_component_no_self_loops(self, G):           
                
        G2 = G.copy()
        
        G2.remove_edges_from(nx.selfloop_edges(G2))    
        for node in list(nx.isolates(G2)):
            G2.remove_node(node)
         
        graphs = sorted(list(nx.connected_component_subgraphs(G2)), key = len, reverse=True)
        
        #G = max(nx.connected_components(G), key=len)
        if len(graphs) > 0:
            return graphs[0]
        else:
            return G2
        
    
    #####################################
    # Method: export_nodes_edges_to_file
    # Description: export nodes and edges of a graph into a file
    def export_nodes_edges_to_file(self, G, node_f_name, edge_f_name):
                        
        nx.write_edgelist(G,  edge_f_name)               
        np.savetxt(node_f_name, np.array(sorted(G.degree(), key=lambda x: x[1], reverse=True)), fmt="%s", encoding="utf-8")
        
        
    #####################################
    # Method: create_node_subgraph
    # Description: creates a subgraph for one node. 
    # subgraph contains all nodes connected to that node and their edges to each other
    def create_node_subgraph(self, G, node):
        G_subgraph_edges = nx.Graph()
        G_subgraph_edges = G.edge_subgraph(G.edges(node))
        G_subgraph = G.subgraph(G_subgraph_edges.nodes())
        
        return G_subgraph
        
    #####################################
    # Method: get_top_degree_nodes
    # Description: returns a array of the top degree nodes based on parameter passed by user    
    def get_top_degree_nodes(self, G, top_degree_start, top_degree_end):
        
        return np.array(sorted(G.degree(), key=lambda x: x[1], reverse=True))[top_degree_start-1:top_degree_end]
    
    
    #####################################
    # Method: calculate_spectral_clustering_labels
    # Description: calculate cluster labels for a graph
    def calculate_spectral_clustering_labels(self, G, k, affinity = 'precomputed', n_init=100):
        #adj_mat = nx.to_numpy_matrix(G)
        adj_mat = nx.to_scipy_sparse_matrix(G)        
        sc = SpectralClustering(k, affinity=affinity, n_init=n_init)
        sc.fit(adj_mat)
               
        return  sc.labels_
    
    
    #####################################
    # Method: calculate_spectral_clustering
    # Description: calculate cluster labels for a graph
    def calculate_spectral_clustering(self, G, k=None, affinity = 'precomputed', n_init=100):
        
        #calculate adjacent matrix
        adj_mat = nx.to_scipy_sparse_matrix(G)  
        
        #get number of clusters if None was given
        if k == None:
            nb_clusters, eigenvalues, eigenvectors = self.eigenDecomposition(adj_mat)            
            k = nb_clusters[0]
              
        #calculate spectral clustering labels
        sc = SpectralClustering(k, affinity=affinity, n_init=n_init)
        sc.fit(adj_mat)
        
        #update graph with the communitites
        dic = dict(zip(G.nodes(), sc.labels_))
        nx.set_node_attributes(G, dic, 'community_spectral')        
               
        return G, list(sc.labels_), k
    
    
    #####################################
    # Method: calculate_louvain_clustering
    # Description: calculate cluster labels for a graph using community_louvain
    def calculate_louvain_clustering(self, G):
        
        # compute the best partition
        partition = community_louvain.best_partition(G)

        #get number of clusters
        partition_arr = list(partition.values())
        partition_arr_no_dups = list(dict.fromkeys(partition_arr)) 
        k = len(partition_arr_no_dups) #number of clusters

        #update graph with the communitites
        dic = dict(zip(G.nodes(), partition.values()))
        nx.set_node_attributes(G, dic, 'community_louvain')
               
        return G, partition.values(), k
    
    
    #####################################
    # Method: calculate_separability
    # Description: calculates the separability score for a community
    # Parameters:
    #   -G_Community: the subgraph with of nodes that belong to the same commty
    #   -G_All: The entire graph
    #   -dens: separability score
    def calculate_separability(self, G_Community, G_All):            
        
        # #of edges for that community - (internal nodes)
        ms = len(G_Community.edges(G_Community.nodes()))
        
        # #of edges edges pointing outside of the community - (external nodes)        
        cs = len(G_All.edges(G_Community.nodes())) - ms
        
        # ratio between internal and external nodes
        sep = ms/cs
        
        return sep
    
    
    #####################################
    # Method: calculate_density
    # Description: calculates the density score for a community
    # Parameters:
    #   -G_Community: the subgraph with of nodes that belong to the same commty    
    # Returns:
    #   -dens: density score
    def calculate_density(self, G_Community):            
        
        # #of edges for that community 
        ms = len(G_Community.edges())
        
        # #of nodes for that community 
        ns = ms = len(G_Community.nodes())
        
        # fraction of the edges that appear between the nodes in G_Community
        dens = ms / (ns * (ns-1) / 2)
        
        return dens
    
    
    #####################################
    # Method: calculate_average_clustering_coef
    # Description: calculates the average clustering coefficient of a graph
    # Parameters:
    #   -G_Community: the subgraph with of nodes that belong to the same commty    
    # Returns:
    #   -acc: the average clustering coefficient
    def calculate_average_clustering_coef(self, G_Community):
        
        # calculates the average clustering coefficient number
        acc = nx.average_clustering(G_Community)

        return acc
    
    
    #####################################
    # Method: calculate_cliques
    # Description: calculates the clique number of the graph and 
    # the number of maximal cliques in the graph.
    # Parameters:
    #   -G_Community: the subgraph with of nodes that belong to the same commty    
    # Returns:
    #   -gcn: the clique number of the graph
    #   -nofc: the number of maximal cliques in the graph
    def calculate_cliques(self, G):
        
        gcn = nx.graph_clique_number(G)
        nofc = nx.graph_number_of_cliques(G)

        return gcn, nofc
    
    
    #####################################
    # Method: calculate_power_nodes_score
    # Description: calculates power nodes score    
    # This is to calculate how many of the total nodes 
    # in graph are connected to a few top degree nodes
    # Parameters:
    #   -G: the graph to analyze
    #   -top_no: top number of nodes you want to analyze
    # Returns:
    #   -pns: power nodes score    
    #         1 means that all other nodes in the graph 
    #         are connected to the top nodes
    def calculate_power_nodes_score(self, G, top_no=3):

        # number of nodes of the original graph
        no_of_nodes = len(G.nodes())                

        # get the top 3 high degree nodes
        arr_nodes = []
        for x in list(self.get_top_degree_nodes(G, 1, top_no)):
            arr_nodes.append(x[0])
    
        # creates a subgrpah of all nodes conencted to the top nodes
        sub_graph = self.create_node_subgraph(G, arr_nodes)
        
        # number of nodes of the sub-graph of top nodes
        no_of_nodes_sub_graph = len(sub_graph.nodes()) 
        
        # calculates the ratio between the two.        
        pns = no_of_nodes_sub_graph / no_of_nodes

        return pns
    
    #####################################
    # Method: calculate_average_node_degree
    # Description: calculates the average of the degree of all nodes        
    # Parameters:
    #   -G: the graph to analyze    
    # Returns:
    #   -deg_mean: the mean 
    def calculate_average_node_degree(self, G):
        
        arr = np.array(sorted(G.degree(), key=lambda x: x[1], reverse=True))                
        deg_mean = np.asarray(arr[:,1], dtype=np.integer).mean()
        
        return deg_mean

    
    #####################################
    # Method: print_cluster_metrics
    # Description: print cluster graphs metrics
    def print_cluster_metrics(self, G_Community, G_All, top_no=3, acc_node_size_cutoff=None):
    
        if acc_node_size_cutoff is None:
            acc_node_size_cutoff = len(G_Community.nodes())
            
        print("# of Nodes: " + str(len(G_Community.nodes())))
        print("# of Edges: " + str(len(G_Community.edges())))
        
        deg_mean = self.calculate_average_node_degree(G_Community)
        print("Average Node Degree: " + str(deg_mean)  + " - " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        sep = self.calculate_separability(G_Community, G_All)
        print("Separability: " + str(sep)+ " - " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        den = self.calculate_density(G_Community)
        print("Density: " + str(den)+ " - " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                            
        if acc_node_size_cutoff > len(G_Community.nodes()):
            acc = self.calculate_average_clustering_coef(G_Community)
            print("Average Clustering Coefficient: " + str(acc) + " - " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        else:
            print("Average Clustering Coefficient: " + " (**more nodes than the cutoff number)")
                
        gcn, nofc = self.calculate_cliques(G_Community)
        print("Clique number: " + str(gcn))
        print("Number of maximal cliques: " + str(nofc) + " - " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                
        pns = self.calculate_power_nodes_score(G_Community, top_no)
        print("Power Nodes Score: " + str(pns) + " - " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        


    #####################################
    # Method: eigenDecomposition
    # Description: This method performs the eigen decomposition on a given affinity matrix
    # Re-used code from https://github.com/ciortanmadalina/high_noise_clustering
    # Parameters: 
    #   -af_matrix = Affinity matrix
    #    -bln_plot = flag to determine if we should plot the sorted eigen values for visual inspection or not
    #    -topK = number of suggestions as the optimal number of of clusters
    # Returns: 
    #   -nb_clusters = the optimal number of clusters by eigengap heuristic
    #   -eigenvalues = all eigen values
    #   -eigenvectors = all eigen vectors
    def eigenDecomposition(self, af_matrix, bln_plot = False, topK = 5):       
        
        #construct the laplacian of the matrix
        L = csgraph.laplacian(af_matrix, normed=True)
        n_components = af_matrix.shape[0]

        # LM parameter : Eigenvalues with largest magnitude (eigs, eigsh), that is, largest eigenvalues in 
        # the euclidean norm of complex numbers.
        #eigenvalues, eigenvectors = sp.sparse.linalg.eigs(L)
        eigenvalues, eigenvectors = SLA.eigsh(L, which = 'LM')

        if bln_plot:
            plt.title('Largest eigen values of input matrix')
            plt.scatter(np.arange(len(eigenvalues)), eigenvalues)
            plt.grid()

        # Identify the optimal number of clusters as the index corresponding
        # to the larger gap between eigen values
        index_largest_gap = np.argsort(np.diff(eigenvalues))[::-1][:topK]
        nb_clusters = index_largest_gap + 1

        return nb_clusters, eigenvalues, eigenvectors

    


    
        
    #####################################
    # Method: remove_edges
    # Description: removes edges of nodes with less than the given degree. 
    # (both nodes in the edge must be less than the given degree for us to remove the edge)
    def remove_edges(self, G, min_degree_no):
        
        G2 = G.copy()        
        count = 0
        for edge in list(G2.edges()):
            degree_node_from = G2.degree(edge[0])
            degree_node_to = G2.degree(edge[1])

            if degree_node_from < min_degree_no and degree_node_to < min_degree_no:
                count = count +1
                G2.remove_edge(edge[0], edge[1])

        print(str(count) + ' edges removed')        
        return G2    
    
    

    #####################################
    # Method: remove_edges_eithernode
    # Description: removes edges of nodes with less than the given degree. 
    # (both nodes in the edge must be less than the given degree for us to remove the edge)
    def remove_edges_eithernode(self, G, min_degree_no):
        
        G2 = G.copy()
        count = 0
        for edge in list(G2.edges()):
            degree_node_from = G2.degree(edge[0])
            degree_node_to = G2.degree(edge[1])

            if degree_node_from < min_degree_no or degree_node_to < min_degree_no:
                count = count +1
                G2.remove_edge(edge[0], edge[1])

        print(str(count) + ' edges removed')
        return G2       
    
    
    
    #####################################
    # Method: contract_nodes_degree1
    # Description: Contract nodes degree 1 in groups of the given number
    def contract_nodes_degree1(self, G, n_to_group):    
        
        G2 = G.copy()    
        degree_to_contract = 1                 
        
        for node_degree in list(sorted(G2.degree, key=lambda x: x[1], reverse=True)):    
            
            try:
                D = nx.descendants(G2, node_degree[0])
                D.add(node_degree[0])
                this_node_subgraph = G2.subgraph(D)


                ##################### degree1
                nodes_degree1 = [node for node, degree in list(this_node_subgraph.degree()) if degree == degree_to_contract]
                subgraph_degree1 = this_node_subgraph.subgraph(nodes_degree1)

                j = 0
                n = int(n_to_group/(degree_to_contract))

                for node in list(subgraph_degree1):                           
                    if j==0 or j%n==0:
                        first_node = node
                    else:        
                        G2 = nx.contracted_nodes(G2, first_node, node, self_loops=True)

                    j=j+1                                                                             

            except Exception as e:        
                continue
          

        return G2    
    
        
    #####################################
    # Method: print_Measures
    # Description: print Graph measures to the screen and to a file 
    def print_Measures(
            self, 
            G, 
            blnCalculateDimater=False, 
            blnCalculateRadius = False, 
            blnCalculateExtremaBounding=False, 
            blnCalculateCenterNodes=False, 
            fileName_to_print = None):
        
        
        #verify if graph is connected or not
        try:
            blnGraphConnected =  is_connected(G)
        except:
            blnGraphConnected = False
            
        
        no_nodes =  str(len(G.nodes()))
        no_edges = str(len(G.edges()))
        print("# Nodes: " + no_nodes)
        print("# Edges: " + no_edges)
        
        
        #Calculate and print Diameter 
        if blnCalculateDimater == True:
            if blnGraphConnected == True:
                diameter_value = str(distance_measures.diameter(G))
                print("Diameter: " + diameter_value)
            else:
                diameter_value = "Not possible to calculate diameter. Graph must be connected"
                print(diameter_value)
                
        #Calculate and print Radius 
        if blnCalculateRadius == True:
            if blnGraphConnected == True:
                radius_value = str(distance_measures.radius(G))
                print("Radius: " + radius_value)
            else:
                radius_value = "Not possible to calculate radius. Graph must be connected"
                print(radius_value)        
        
        #Calculate and print Extrema bounding 
        if blnCalculateExtremaBounding == True:
            if blnGraphConnected == True:
                extrema_bounding_value = str(distance_measures.extrema_bounding(G))
                print("Extrema bounding: " + extrema_bounding_value) 
            else:
                extrema_bounding_value = "Not possible to calculate Extrema bounding. Graph must be connected"
                print(extrema_bounding_value)
        
        #Calculate and print Centers
        if blnCalculateCenterNodes == True:
            str_centers_nodes=""
            if blnGraphConnected == True:
                centers_nodes = distance_measures.center(G)    
                str_centers_nodes = str(sorted(G.degree(centers_nodes), key=lambda x: x[1], reverse=True))
                print("Centers with their degree: " + str_centers_nodes) 
            else:
                centers_nodes = "Not possible to calculate Centers. Graph must be connected"
                print(centers_nodes)


        # if file name is passed in the parameters, we save the measures into a file
        if fileName_to_print != None:
            #creates path if does not exists
            
            if not os.path.exists(os.path.dirname(fileName_to_print)):
                os.makedirs(os.path.dirname(fileName_to_print))
                        
            f = open(fileName_to_print, "w")
            f.write("# Nodes: " + no_nodes + "\n")
            f.write("# Edges: " + no_edges + "\n")
            
            if blnCalculateDimater == True:
                f.write("Diameter: " + diameter_value + "\n")
            if blnCalculateRadius == True:
                f.write("Radius: " + radius_value + "\n")
            #if blnCalculateBaryCenter == True:
            #    f.write("Bary Center: " + barycenter_node + "\n")        
            if blnCalculateExtremaBounding == True:
                f.write("Extrema bounding: " + extrema_bounding_value + "\n")
            if blnCalculateCenterNodes == True:
                f.write("Centers with their degree: " + str_centers_nodes + "\n")
                
            f.close()



import os
import json
import datetime
import csv
import string
import gensim
from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel

import nltk
from nltk.corpus import words, stopwords, wordnet
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag
from nltk.tokenize import word_tokenize

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import collections

import numpy as np
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

#dictionary_words = dict.fromkeys(words.words(), None)
   
#stopWords = set(stopwords.words('english'))
#tokenizer = RegexpTokenizer(r'\w+')

#stemmer = PorterStemmer()
#lemmatiser = WordNetLemmatizer()

stop = set(stopwords.words('english'))
stop.add ('u')
stop.add ('e')
exclude = set(string.punctuation) 
lemma = WordNetLemmatizer()

#topic_doc_complete = []
#lda_model = ""


class TwitterTopics:   

    def __init__(self, folder_path, mongoDB_database=None):
                    
        self.folder_path = folder_path        
        self.lda_model = object()
        self.lsi_model = object()
        self.doc_term_matrix = object()
        self.dictionary = object()
        self.lda_coh_u_mass = 0
        self.lda_coh_c_v = 0
        self.lsi_coh_u_mass = 0
        self.lsi_coh_c_v = 0          
        
        self.db = mongoDB_database
        if mongoDB_database is not None:
            self.c_topics = self.db.topics
        else:
            self.c_topics = None
                
        
    def __del__(self):
        self.folder_path = None        
        self.lda_model = None
        self.lsi_model = None
        self.doc_term_matrix = None
        self.dictionary = None
        self.lda_coh_u_mass = None
        self.lda_coh_c_v = None
        self.lsi_coh_u_mass = None
        self.lsi_coh_c_v = None
                

    def get_coh_u_mass(self):
        return self.lda_coh_u_mass, self.lsi_coh_u_mass
    
    def get_coh_c_v(self):
        return self.lda_coh_c_v, self.lda_coh_c_v
    

    #create one array with all tweets of one hashtag for topic analysis
    def get_docs_from_file(self, file_path):

        docs = []
        
        with open(file_path, 'r', encoding='utf8', errors='ignore') as f:                            
            for line in f:        
                docs.append(line)

        f.close()
        return docs


    #clean documents for topic analysis
    def clean_docs(self, doc, delete_numbers=True, delete_stop_words=True, lemmatize_words=True): 
        
        doc_clean = doc
        
        if delete_numbers ==True:
            doc_clean = doc.replace('1', '').replace('2', '').replace('3', '').replace('4', '').replace('5', '').replace('6', '').replace('7', '').replace('8', '').replace('9', '').replace('0', '')
            
        if delete_stop_words == True:
            doc_clean = " ".join([i for i in doc_clean.lower().split() if i not in stop])
        
        doc_clean = ''.join(ch for ch in doc_clean if ch not in exclude)
        
        if lemmatize_words == True:
            doc_clean = " ".join(lemma.lemmatize(word) for word in doc_clean.split())

        return doc_clean
        

    #train model
    def train_model(self, topic_docs, num_topics, model_name, blnSaveinDB=False, blnSaveTrainedModelFiles=False, txtFileName=None,
                    model_type='both', lda_num_of_iterations=150, delete_stop_words=True, lemmatize_words=True, delete_numbers=True):
        
        #starttime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        #print("Executing train_model... Started at: " + starttime )        

        doc_clean = [self.clean_docs(doc, delete_numbers, delete_stop_words, lemmatize_words).split() for doc in topic_docs]

        # Creating the term dictionary of our corpus, where every unique term is assigned an index. dictionary = corpora.Dictionary(doc_clean)
        self.dictionary = corpora.Dictionary(doc_clean)

        # Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
        self.doc_term_matrix = [self.dictionary.doc2bow(doc) for doc in doc_clean]

        # Creating the object for LDA model using gensim library
        Lda = gensim.models.ldamodel.LdaModel

        
        file_data = []        
        
        if model_type in ('lda', 'both'):
            # Build the LDA model
            self.lda_model = gensim.models.LdaModel(corpus=self.doc_term_matrix, num_topics=num_topics, id2word=self.dictionary, iterations=lda_num_of_iterations)                            
            #get LDA coherence
            self.lda_coh_u_mass = CoherenceModel(model=self.lda_model, corpus=self.doc_term_matrix, dictionary=self.dictionary, coherence='u_mass') 
            self.lda_coh_c_v = CoherenceModel(model=self.lda_model, texts=doc_clean, dictionary=self.dictionary, coherence='c_v')
            
            #create json file with lda results
            for idx in range(num_topics):                
                topic = idx+1
                strtopic = str(topic)
                data = '{"model_name":"' + model_name + \
                        '", "model_type":"' + 'lda' + \
                        '", "timestamp":"' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + \
                        '", "no_tweets":"' + str(len(topic_docs)) + \
                        '", "coh_u_mass":"' + str(self.lda_coh_u_mass.get_coherence()) + \
                        '", "coh_c_v":"' + str(self.lda_coh_c_v.get_coherence()) + \
                        '", "topic_no":"' + strtopic + \
                        '", "topic":"' + str(self.lda_model.print_topic(idx, num_topics)).replace('"', "-") + '"}'
                x = json.loads(data)
                file_data.append(x)
            
                
        if model_type in ('lsi', 'both'):
            # Build the LSI model
            self.lsi_model = gensim.models.LsiModel(corpus=self.doc_term_matrix, num_topics=num_topics, id2word=self.dictionary)    
            #get LSI coherence
            self.lsi_coh_u_mass = CoherenceModel(model=self.lsi_model, corpus=self.doc_term_matrix, dictionary=self.dictionary, coherence='u_mass') 
            self.lsi_coh_c_v = CoherenceModel(model=self.lsi_model, texts=doc_clean, dictionary=self.dictionary, coherence='c_v')
        
            #create json file with lsi results
            for idx in range(num_topics):
                topic = idx+1
                strtopic = str(topic)
                data = '{"model_name":"' + model_name + \
                        '", "model_type":"' + 'lsi' + \
                        '", "timestamp":"' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + \
                        '", "no_tweets":"' + str(len(topic_docs)) + \
                        '", "coh_u_mass":"' + str(self.lsi_coh_u_mass.get_coherence()) + \
                        '", "coh_c_v":"' + str(self.lsi_coh_c_v.get_coherence()) + \
                        '", "topic_no":"' + strtopic + \
                        '", "topic":"' + str(self.lsi_model.print_topic(idx, num_topics)).replace('"', "-") + '"}'
                x = json.loads(data)
                file_data.append(x)


        # Save if mongoDB collection is asked
        if blnSaveinDB == True:        
            if self.db  is not None:
                self.c_topics.insert_many(file_data)                
            else:
                print("Can't save topics in db. No mongoDB connection was set up.")
                    
        # Save results in a text file
        if txtFileName is not None:
            with open(txtFileName, 'w', encoding="utf-8") as outfile:
                json.dump(file_data, outfile)
    

            
            
        # Save models into file
        if blnSaveTrainedModelFiles == True:
            
            #creates path if does not exists
            if not os.path.exists(self.folder_path + "/trained_models/"):
                os.makedirs(self.folder_path + "/trained_models/")
            
            self.lda_model.save(self.folder_path + "/trained_models/" + model_name + "_lda_model.model")
            self.dictionary.save(self.folder_path + "/trained_models/" + model_name + "_dictionary.dict")
        
        
        #endtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        #print("Finished executing train_model. Ended at: " + endtime)
    

    #train model from file
    def train_model_from_file(self, file_path, num_topics, model_name, blnSaveinDB=False, blnSaveTrainedModelFiles=False, txtFileName=None,
                    model_type='both', lda_num_of_iterations=150, delete_stop_words=True, lemmatize_words=True, delete_numbers=True):        
        docs = self.get_docs_from_file(file_path)
        self.train_model(docs, num_topics, model_name, blnSaveinDB, blnSaveTrainedModelFiles, txtFileName, model_type, lda_num_of_iterations, delete_stop_words, lemmatize_words, delete_numbers)        
    
    
    
        
    
    #plot graph with lda topics
    def plot_topics(self, file_name, no_of_topics, model_type = 'lda', fig_size_x = 17, fig_size_y=15, replace_existing_file=True):
        
        
        if replace_existing_file==True or not os.path.exists(file_name):
                
            fig_size_y = 7*(no_of_topics/2)        
            fiz=plt.figure(figsize=(fig_size_x, fig_size_y))

            for i in range(no_of_topics):
                if model_type == 'lda':
                    df=pd.DataFrame(self.lda_model.show_topic(i), columns=['term','prob']).set_index('term')        
                elif model_type == 'lsi':
                    df=pd.DataFrame(self.lsi_model.show_topic(i), columns=['term','prob']).set_index('term')        

                no_rows = int(no_of_topics/2)+no_of_topics%2            
                plt.subplot(no_rows,2,i+1)
                plt.title('topic '+str(i+1))
                sns.barplot(x='prob', y=df.index, data=df, label='Cities', palette='Reds_d')
                plt.xlabel('probability')

            #save the file 
            plt.savefig(file_name, dpi=200, facecolor='w', edgecolor='w')

            #plt.show()
            plt.cla()   # Clear axis
            plt.clf()   # Clear figure
            plt.close() # Close a figure window
                

    
    # read a frequency list into a pandas objects
    # file format word\tfrequency
    def read_freq_list_file(self, file_path, delimiter='\t'):
        #df = pd.read_csv(file_path, encoding = "ISO-8859-1", header=None, sep=delimiter, lineterminator='\n')        
        df = pd.read_csv(file_path, encoding = "utf-8", header=None, sep=delimiter, lineterminator='\n')
        
        df.columns = ['word', 'freq']
        
        return df
    
    
    #plot a bar graph with the top frequency list
    def plot_top_freq_list(self, fr_list, top_no, ylabel, exclude_top_no=0, file=None, replace_existing_file= True):
                        
        if exclude_top_no != 0:
            fr_list = fr_list.iloc[exclude_top_no:]
        
        fr_list = fr_list.nlargest(top_no,'freq')                            
        
        
        if len(fr_list) < top_no:
            for i in range( int((top_no-len(fr_list)) / 2.5)):
                
                data = [['', 0], ['', 0] ]    
                df2 = pd.DataFrame(data, columns = ['word', 'freq'])  
                fr_list = fr_list.append(df2)            

        
        fr_list_gr = fr_list.groupby("word")
            
        plt.figure(figsize=(12, len(fr_list)/2.5))                
        fr_list_gr.max().sort_values(by="freq",ascending=True)["freq"].plot.barh()
        plt.xticks(rotation=50)
        plt.xlabel("Frequency")
        plt.ylabel(ylabel)
        if file != None:
            if replace_existing_file==True or not os.path.exists(file):
                plt.savefig(file, dpi=300, bbox_inches='tight')
                
        #plt.show()
        plt.cla()   # Clear axis
        plt.clf()   # Clear figure
        plt.close() # Close a figure window
                
        
    
    #plot a word cloudfor a frequency list
    def plot_word_cloud(self, fr_list, file=None, replace_existing_file=True):                
        
        wordcount = collections.defaultdict(int)

        for index, row in fr_list.iterrows():            
            wordcount[row['word']] = row['freq']
    
        try:
            wordcloud = WordCloud(width=2000, height=1300, max_words=1000, background_color="white").generate_from_frequencies(wordcount)
        except:
            wordcloud = WordCloud(width=2000, height=1300, background_color="white").generate_from_frequencies(wordcount)            

        # Display the generated image:
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        
        if file is not None:
            if replace_existing_file==True or not os.path.exists(file):
                plt.savefig(str(file), dpi=300)
                        
        #plt.show()            
        plt.cla()   # Clear axis
        plt.clf()   # Clear figure
        plt.close() # Close a figure window
                        

    
    #load existing model from file
    #predict topic of a new tweet based on model
    
    
    
       

        

