import json
import re
import numpy as np
import tkinter.messagebox as message
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from datetime import datetime, timedelta
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors

from keyword_explorer.Apps.AppBase import AppBase
from keyword_explorer.OpenAI.OpenAIComms import OpenAIComms
from keyword_explorer.TwitterV2.TweetKeywords import TweetKeywords
from keyword_explorer.tkUtils.CanvasFrame import CanvasFrame
from keyword_explorer.tkUtils.TopicComboExt import TopicComboExt
from keyword_explorer.tkUtils.DataField import DataField
from keyword_explorer.tkUtils.LabeledParam import LabeledParam
from keyword_explorer.tkUtils.SelectParam import SelectParam
from keyword_explorer.tkUtils.Buttons import Buttons
from keyword_explorer.tkUtils.ToolTip import ToolTip
from keyword_explorer.tkUtils.MoveableNode import MovableNode
from keyword_explorer.tkUtils.Checkboxes import Checkboxes, DIR
from keyword_explorer.utils.MySqlInterface import MySqlInterface
from keyword_explorer.utils.ManifoldReduction import ManifoldReduction, EmbeddedText
from keyword_explorer.utils.CorporaGenerator import CorporaGenerator
from keyword_explorer.tkUtils.TextField import TextField

from typing import Dict, List, Any

class SubsampleInfo:
    experiment_id:int
    tweet_row:int
    query_row:int
    keyword:str

    def __init__(self, experiment_id:int, tweet_row:int, query_row:int, keyword:str):
        self.experiment_id = experiment_id
        self.tweet_row = tweet_row
        self.query_row = query_row
        self.keyword = keyword

    def to_db(self, msi:MySqlInterface):
        sql = "insert into table_subsampled (experiment_id, tweet_row, query_row, keyword) VALUES (%s, %s, %s, %s)"
        vals = (self.experiment_id, self.tweet_row, self.query_row, self.keyword)
        msi.write_sql_values_get_row(sql, vals)

class EmbeddingsExplorer(AppBase):
    oai: OpenAIComms
    msi: MySqlInterface
    mr: ManifoldReduction
    cg: CorporaGenerator
    canvas_frame: CanvasFrame
    engine_combo: TopicComboExt
    keyword_combo: TopicComboExt
    graph_keyword_combo: TopicComboExt
    experiment_combo: TopicComboExt
    keyword_count_field: DataField
    exclude_cluster_field: DataField
    pca_dim_param: LabeledParam
    eps_param: LabeledParam
    min_samples_param: LabeledParam
    perplexity_param: LabeledParam
    rows_param: LabeledParam
    subsampleParam:SelectParam
    tweet_option_checkboxes:Checkboxes
    author_option_checkboxes:Checkboxes
    generation_options:Checkboxes
    corpora_action_buttons:Buttons
    speech_action_buttons:Buttons
    canvas_command_buttons:Buttons
    db_buttons:Buttons
    speech_text_field:TextField
    experiment_id: int
    speech_df:pd.DataFrame
    subsample_list:List

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("EmbeddingsExplorer")

    def setup_app(self):
        self.app_name = "EmbeddingsExplorer"
        self.app_version = "3.23.23"
        self.geom = (640, 620)
        self.oai = OpenAIComms()
        self.tkws = TweetKeywords()
        self.msi = MySqlInterface(user_name="root", db_name="twitter_v2")
        self.mr = ManifoldReduction()
        self.cg = CorporaGenerator(self.msi)
        self.subsample_list = []

        if not self.oai.key_exists():
            message.showwarning("Key Error", "Could not find Environment key 'OPENAI_KEY'")

        if not self.tkws.key_exists():
            message.showwarning("Key Error", "Could not find Environment key 'BEARER_TOKEN_2'")
        self.experiment_id = -1
        self.speech_df = pd.DataFrame()


    def build_app_view(self, row: int, text_width: int, label_width: int) -> int:
        experiments = ["exp_1", "exp_2", "exp_3"]
        keywords = ["foo", "bar", "bas"]
        print("build_app_view")

        combo_width = 30
        self.experiment_combo = TopicComboExt(self, row, "Experiment:", self.dp, entry_width=20, combo_width=combo_width)
        self.experiment_combo.set_combo_list(experiments)
        self.experiment_combo.set_callback(self.keyword_callback)
        row = self.experiment_combo.get_next_row()
        ToolTip(self.experiment_combo.tk_combo, "Select the experiment you want to explore here")
        self.keyword_combo = TopicComboExt(self, row, "Keyword:", self.dp, entry_width=20, combo_width=combo_width)
        self.keyword_combo.set_combo_list(keywords)
        ToolTip(self.keyword_combo.tk_combo, "Select the keyword for the experiment. 'all_keywords' gets everything")
        b = self.keyword_combo.add_button("Num Entries:", command=lambda: self.get_keyword_entries_callback(
            self.keyword_combo.get_text()))
        ToolTip(b, "Query the DB to see how many entries there are\nResults go in 'Num Rows:'")
        row = self.keyword_combo.get_next_row()

        s = ttk.Style()
        s.configure('TNotebook.Tab', font=self.default_font)

        # Add the tabs
        tab_control = ttk.Notebook(self)
        tab_control.grid(column=0, row=row, columnspan=2, sticky="nsew")
        get_store_tab = ttk.Frame(tab_control)
        tab_control.add(get_store_tab, text='Get/Store')
        self.build_get_store_tab(get_store_tab)

        canvas_tab = ttk.Frame(tab_control)
        tab_control.add(canvas_tab, text='Canvas')
        self.build_graph_tab(canvas_tab)

        speech_tab = ttk.Frame(tab_control)
        tab_control.add(speech_tab, text='Speech')
        self.build_speech_tab(speech_tab, text_width, label_width)

        corpora_tab = ttk.Frame(tab_control)
        tab_control.add(corpora_tab, text='Corpora')
        self.build_create_corpora_tab(corpora_tab)
        row += 1

        return row

    def build_create_corpora_tab(self, tab: ttk.Frame):
        label_width = 20
        row = 0
        self.tweet_option_checkboxes = Checkboxes(tab, row, "Tweet meta wrapping:", label_width=label_width, border=True)
        # cb = self.tweet_option_checkboxes.add_checkbox("Randomize", self.randomize_callback, dir=DIR.ROW)
        # ToolTip(cb, "Randomly select the starting time for each day so that a full pull won't go into tomorrow")
        cb = self.tweet_option_checkboxes.add_checkbox("Created at", lambda : self.set_corpora_flag_callback("tweet_created_at_flag"), dir=DIR.ROW)
        ToolTip(cb.cb, "Adds the date and time the tweet was created to the tweet meta-wrapping")
        cb = self.tweet_option_checkboxes.add_checkbox("Language", lambda: self.set_corpora_flag_callback("language_flag"), dir=DIR.ROW)
        ToolTip(cb.cb, "Adds the tweet language to the tweet meta-wrapping")
        cb = self.tweet_option_checkboxes.add_checkbox("Keyword", lambda: self.set_corpora_flag_callback("keyword_flag"), dir=DIR.ROW)
        ToolTip(cb.cb, "Adds the keyword to the tweet meta-wrapping")
        cb = self.tweet_option_checkboxes.add_checkbox("Exclude threaded tweets", lambda: self.set_corpora_flag_callback("exclude_thread_flag"), dir=DIR.ROW)
        ToolTip(cb.cb, "Excludes tweets that are in threads connected\nto a tweet containing the keyword")
        row = self.tweet_option_checkboxes.get_next_row()
        self.author_option_checkboxes = Checkboxes(tab, row, "Author meta wrapping:", label_width=label_width, border=True)
        cb = self.author_option_checkboxes.add_checkbox("Name", lambda: self.set_corpora_flag_callback("name_flag"), dir=DIR.ROW)
        ToolTip(cb.cb, "Adds the tweet author's name to the tweet meta-wrapping")
        cb = self.author_option_checkboxes.add_checkbox("Username", lambda: self.set_corpora_flag_callback("username_flag"), dir=DIR.ROW)
        ToolTip(cb.cb, "Adds the tweet author's username to the tweet meta-wrapping")
        cb = self.author_option_checkboxes.add_checkbox("Location", lambda: self.set_corpora_flag_callback("location_flag"), dir=DIR.ROW)
        ToolTip(cb.cb, "Adds the tweet author's location to the tweet meta-wrapping")
        cb = self.author_option_checkboxes.add_checkbox("Description", lambda: self.set_corpora_flag_callback("description_flag"), dir=DIR.ROW)
        ToolTip(cb.cb, "Adds the tweet author's self-description to the tweet meta-wrapping")
        row = self.author_option_checkboxes.get_next_row()
        self.generation_options = Checkboxes(tab, row, "Corpora Generation:", label_width=label_width, border=True)
        cb = self.generation_options.add_checkbox("Wrapping before text (default is after)", lambda: self.set_corpora_flag_callback("wrap_after_text_flag"), dir=DIR.ROW)
        ToolTip(cb.cb, "Check to place the meta-wrapping before the tweet text")
        cb = self.generation_options.add_checkbox("Single file (default is separate)", lambda: self.set_corpora_flag_callback("single_file_flag"), dir=DIR.ROW)
        ToolTip(cb.cb, "Generate a single test/train set with all keywords in it\nrather than a separate test/trin file for each keyword")
        cb = self.generation_options.add_checkbox("Percent OFF (default is ON)", lambda: self.set_corpora_flag_callback("percent_on_flag"), dir=DIR.ROW)
        ToolTip(cb.cb, "Disable the generation of percent values in the meta-wrapping to determine learning accuracy (10%, 20%, 30%, 40%")
        cb = self.generation_options.add_checkbox("Include excluded clusters", lambda: self.set_corpora_flag_callback("excluded_culsters_flag"), dir=DIR.ROW)
        ToolTip(cb.cb, "Include all clusters that were flagged for exclusion")
        row = self.generation_options.get_next_row()
        self.corpora_action_buttons = Buttons(tab, row, "Corpora")
        b = self.corpora_action_buttons.add_button("Set folder", self.cg.set_folder)
        ToolTip(b, "Set the folder for all test/train files to be written to")
        b = self.corpora_action_buttons.add_button("Generate", lambda: self.cg.write_files(self.experiment_id, self.keyword_combo.get_text()))
        ToolTip(b, "Write all test/train files to the selected directory")


    def build_get_store_tab(self, tab: ttk.Frame):
        engine_list = self.oai.list_models(keep_list = ["embed"])
        row = 0
        self.engine_combo = TopicComboExt(tab, row, "Engine:", self.dp, entry_width=25, combo_width=25)
        self.engine_combo.set_combo_list(engine_list)
        self.engine_combo.set_text(engine_list[0])
        self.engine_combo.tk_combo.current(0)
        row = self.engine_combo.get_next_row()
        ToolTip(self.engine_combo.tk_combo, "Select the embedding engine.\nAda is the cheapest, Davinci is the best")
        self.keyword_count_field = DataField(tab, row, "Num rows")
        b = self.keyword_count_field.add_button("Get Embeddings", self.get_oai_embeddings_callback)
        row = self.keyword_count_field.get_next_row()
        ToolTip(b, "Get embeddings for each tweet using the\nselected engine and store them with the\ntweets in the DB")
        self.db_buttons = Buttons(tab, row, "Update DB")
        b = self.db_buttons.add_button("Reduced+Clusters", self.store_reduced_and_clustering_callback, -1)
        ToolTip(b, "Add cluster ids and reduced embeddings (from the Canvas tab)\n to the db along with each tweet")
        b = self.db_buttons.add_button("Clusters", self.store_clustering_callback, -1)
        ToolTip(b, "Add just the cluster ids (from the Canvas tab)\n to the db along with each tweet")
        b = self.db_buttons.add_button("Topic Names", self.implement_me, -1)
        ToolTip(b, "Not implemented. Will guess at topic names using GPT-3")
        b = self.db_buttons.add_button("Users", self.store_user_callback, -1)
        ToolTip(b, "Fora each tweet's user id, create an\nentry in the db for that user using\n the twitter API")
        row = self.db_buttons.get_next_row()

    def build_param_row(self, parent:tk.Frame, row:int) -> int:
        f = tk.Frame(parent)
        f.grid(row=row, column=0, columnspan=2, sticky="nsew", padx=1, pady=1)
        self.pca_dim_param = LabeledParam(f, 0, "PCA Dim:")
        self.pca_dim_param.set_text('10')
        ToolTip(self.pca_dim_param.tk_entry, "The number of dimensions that the PCA\nwill reduce the original vectors to")
        self.eps_param = LabeledParam(f, 2, "EPS:")
        self.eps_param.set_text('8')
        ToolTip(self.eps_param.tk_entry, "DBSCAN: Specifies how close points should be to each other to be considered a part of a \ncluster. It means that if the distance between two points is lower or equal to \nthis value (eps), these points are considered neighbors.")
        self.min_samples_param = LabeledParam(f, 4, "Min Samples:")
        self.min_samples_param.set_text('5')
        ToolTip(self.min_samples_param.tk_entry, "DBSCAN: The minimum number of points to form a dense region. For \nexample, if we set the minPoints parameter as 5, then we need at least 5 points \nto form a dense region.")
        self.perplexity_param = LabeledParam(f, 6, "Perplex:")
        self.perplexity_param.set_text('80')
        ToolTip(self.perplexity_param.tk_entry, "T-SNE: The size of the neighborhood around each point that \nthe embedding attempts to preserve")
        self.rows_param = LabeledParam(f, 8, "Limit:")
        self.rows_param.set_text('1000')
        ToolTip(self.rows_param.tk_entry, "The number of rows that the full list will be subsampled down to\nfor performance")
        self.subsampleParam = SelectParam(f, 10, "Subsample", None)
        return row + 1

    def build_graph_tab(self, tab: ttk.Frame):
        row = 0
        row = self.build_param_row(tab, row)
        f = tk.Frame(tab)
        # add "select clusters" field and "export corpus" button
        self.canvas_command_buttons = Buttons(tab, row, "Commands", label_width=10)
        b = self.canvas_command_buttons.add_button("Retreive", self.retreive_tweet_data_callback, -1)
        ToolTip(b, "Get the high-dimensional embeddings from the DB")
        b = self.canvas_command_buttons.add_button("Reduce", self.reduce_dimensions_callback, -1)
        ToolTip(b, "Reduce a  2 dimensions with PCS and TSNE")
        b = self.canvas_command_buttons.add_button("Cluster", self.cluster_callback, -1)
        ToolTip(b, "Compute clusters on reduced data")
        b = self.canvas_command_buttons.add_button("Plot", self.plot_callback, -1)
        ToolTip(b, "Plot the clustered points using PyPlot")
        b = self.canvas_command_buttons.add_button("Explore", self.explore_callback, -1)
        ToolTip(b, "Interactive graph of a subsample of points")
        b = self.canvas_command_buttons.add_button("Topics", self.label_clusters_callback, -1)
        ToolTip(b, "Use GPT to guess at topic names for clusters\n(not implemented)")
        row = self.canvas_command_buttons.get_next_row()

        f = tk.Frame(tab)
        f.grid(row=row, column=0, columnspan=2, sticky="nsew", padx=1, pady=1)
        self.canvas_frame = CanvasFrame(f, 0, "Interactive embedding explorer (Mouse wheel to zoom)", self.dp, width=550, height=250)
        self.canvas_frame.set_select_callback_fn(self.selected_node_callback)

        row += 1
        self.exclude_cluster_field = DataField(tab, row, "Exclude Cluster:")
        b = self.exclude_cluster_field.add_button("Exclude", self.exclude_cluster_callback)
        ToolTip(b, "Add cluster ID to the exclude table for this experiment")
        row = self.exclude_cluster_field.get_next_row()

    def build_speech_tab(self, frm: ttk.Frame, text_width: int, label_width: int):
        row = 0
        self.speech_text_field = TextField(frm, row, 'Speech:', text_width, height=11, label_width=label_width)
        ToolTip(self.speech_text_field.tk_text, "Speech categories are displayed here")
        row = self.speech_text_field.get_next_row()

        self.speech_action_buttons = Buttons(frm, row, "Actions")
        b = self.speech_action_buttons.add_button("Retrieve", self.retreive_tweet_data_callback)
        ToolTip(b, "Load speech data if available")
        b = self.speech_action_buttons.add_button("Plot", self.plot_speech_data)
        ToolTip(b, "Plot speech classes if available")
        b = self.speech_action_buttons.add_button("Save", self.save_speech_data)
        ToolTip(b, "Save raw chart data to spreadsheet")
        row = self.speech_action_buttons.get_next_row()

    def color_excluded_clusters(self):
        keyword = self.keyword_combo.get_text()
        sql = "select * from table_exclude where experiment_id = %s and keyword = %s"
        vals = (self.experiment_id, keyword)
        if keyword == 'all_keywords':
            sql = "select * from table_exclude where experiment_id = %s"
            vals = (self.experiment_id, )
        result = self.msi.read_data(sql, vals)
        cluster_list = []
        d:Dict
        for d in result:
            cluster_list.append(d['cluster_id'])

        et:EmbeddedText
        for et in self.mr.embedding_list:
            if et.cluster_id in cluster_list and et.mnode != None:
                et.mnode.set_color("black")

    def safe_dict(self, d:Dict, name:str, default:Any) -> Any:
        if name in d:
            return d[name]
        return default

    def set_corpora_flag_callback(self, var_name:str):
        print("set_corpora_flag_callback({})".format(var_name))
        val = self.cg.set_by_name(var_name)

    def store_reduced_and_clustering_callback(self):
        print("store_reduced_and_clustering_callback")

        et:EmbeddedText
        rows = 0
        for et in self.mr.embedding_list:
            ra = np.array(et.reduced)
            sql = "update table_tweet set cluster_id = %s, cluster_name = %s, reduced = %s where row_id = %s;"
            vals = (int(et.cluster_id), et.cluster_name, ra.dumps(), int(et.row_id))
            # print("store_reduced_and_clustering_callback\n\t{}\n\t{}".format(sql, vals))
            self.msi.write_sql_values_get_row(sql, vals)
            rows += 1


        # get the embedding model entry
        sql = "select * from table_embedding_params where experiment_id = %s AND keyword = %s"
        vals = (self.experiment_id, self.keyword_combo.get_text())
        results = self.msi.read_data(sql, vals)
        # insert if there is no experiment_id that matches, otherwise update
        if len(results) == 0:
            sql = "insert into table_embedding_params (experiment_id, keyword, model, PCA_dim, EPS, min_samples, perplexity) values (%s, %s, %s, %s, %s, %s, %s)"
            vals = (self.experiment_id, self.keyword_combo.get_text(), self.engine_combo.get_text(), self.pca_dim_param.get_as_int(), self.eps_param.get_as_float(),
                    self.min_samples_param.get_as_int(), self.perplexity_param.get_as_int())
            self.msi.write_sql_values_get_row(sql, vals)
        else:
            sql = "update table_embedding_params set PCA_dim = %s, EPS = %s, min_samples = %s, perplexity = %s where experiment_id = %s and keyword = %s"
            vals = (self.pca_dim_param.get_as_int(), self.eps_param.get_as_float(), self.min_samples_param.get_as_int(), self.perplexity_param.get_as_int(),
                    self.experiment_id, self.keyword_combo.get_text())
            self.msi.write_sql_values_get_row(sql, vals)
        message.showinfo("DB Write", "Wrote {} rows of reduced and cluster data".format(rows))


    def store_clustering_callback(self):
        print("store_clustering_callback")
        et:EmbeddedText
        rows = 0
        for et in self.mr.embedding_list:
            sql = "update table_tweet set cluster_id = %s, cluster_name = %s where row_id = %s;"
            vals = (et.cluster_id, et.cluster_name, et.row_id)
            self.msi.write_sql_values_get_row(sql, vals)
            rows += 1
        message.showinfo("DB Write", "Wrote {} rows of cluster data".format(rows))

    def store_user_callback(self):
        keyword = self.keyword_combo.get_text()

        if self.experiment_id == -1 or len(keyword) < 2:
            message.showwarning("DB Error", "get_db_embeddings_callback(): Please set database and/or keyword")
            return

        sql = "select distinct author_id from keyword_tweet_view where experiment_id = %s order by author_id"
        vals = (self.experiment_id,)
        if keyword != 'all_keywords':
            sql = "select distinct author_id from keyword_tweet_view where experiment_id = %s and keyword = %s order by author_id"
            vals = (self.experiment_id, keyword)
        results = self.msi.read_data(sql, vals)
        d:Dict
        count = 0
        l = []
        for d in results:
            l.append(d['author_id'])
            count += 1
            if count == 99:
                self.tkws.run_user_query(l, self.msi)
                s = ",".join(map(str, l))
                print("[{}]: {}".format(len(l), s))
                count = 0
                l = []
        if len(l) > 0:
            self.tkws.run_user_query(l, self.msi)
            s = ",".join(map(str, l))
            print("[{}]: {}".format(len(l), s))

        # self.tkws.run_user_query([155,22186596,758514997995589632,1289933022901370880], self.msi)
        print("store_user_callback(): complete")

    def retreive_tweet_data_callback(self):
        print("TweetEmbedExplorer.retreive_tweet_data_callback()")
        keyword = self.keyword_combo.get_text()

        if self.experiment_id == -1 or len(keyword) < 2:
            message.showwarning("DB Error", "get_db_embeddings_callback(): Please set database and/or keyword")
            return

        tweet_result = []
        # if the subsample box is checked, retrieve or select a subsample list of tweets.
        subsample_exists = False
        if self.subsampleParam.get_value():
            num_rows = self.rows_param.get_as_int()
            print("\tChecking if {} subsamples exists".format(num_rows))
            if self.keyword_combo.get_text() == 'all_keywords':
                sql = "select * from subsample_tweet_view where experiment_id = %s"
                vals = (self.experiment_id,)
            else:
                sql = "select * from subsample_tweet_view where experiment_id = %s and keyword = %s"
                vals = (self.experiment_id,self.keyword_combo.get_text())
            tweet_result = self.msi.read_data(sql, vals)
            print("\tSubsample test: Found {} rows".format(len(tweet_result)))
            if len(tweet_result) >= num_rows:
                print("\tLoading existing samples")
                # we have found existing subsamples, so we don't have to make any
                subsample_exists = True
            else:
                print("\tNot enough samples found {} found vs. {} needed. Clearing subsample list for new manifold reduction".format(len(tweet_result), num_rows))
                # clear out the subsample list
                tweet_result = []
                if self.keyword_combo.get_text() == 'all_keywords':
                    sql = "delete from table_subsampled where experiment_id = %s"
                    vals = (self.experiment_id,)
                else:
                    sql = "delete from table_subsampled where experiment_id = %s and keyword = %s"
                    vals = (self.experiment_id,self.keyword_combo.get_text())
                self.msi.write_sql_values_get_row(sql, vals)


            # If no subsamples exist, get some from the main list
            if subsample_exists == False:
                print("\tCreating a subsample of data")
                # get the min and max query_id of queries for a experiment/keyword combo
                if self.keyword_combo.get_text() == 'all_keywords':
                    sql = "select MIN(id) as min_query, max(id)as max_query from table_query where experiment_id = %s"
                    vals = (self.experiment_id,)
                else:
                    sql = "select MIN(id) as min_query, max(id)as max_query from table_query where experiment_id = %s and keyword = %s"
                    vals = (self.experiment_id,self.keyword_combo.get_text())

                results = self.msi.read_data(sql, vals)
                if len(results) == 0:
                    message.showwarning("DB Error", "get_db_embeddings_callback(): Got zero query ids")
                    return

                # get the number of tweets
                min_query = results[0]['min_query']
                max_query = results[0]['max_query']
                print("\tExperiment {}: query range = {:,} - {:,}".format(self.experiment_id, min_query, max_query))
                sql = "CALL get_random_tweets(%s, %s, %s)"
                num_rows = self.rows_param.get_as_int()
                vals = (min_query, max_query, num_rows)
                print("\tLoading {} NEW subsampled rows from DB".format(num_rows))
                tweet_result = self.msi.read_data(sql, vals)
                print("\tGot {} tweets back".format(len(tweet_result)))
                if len(results) == 0:
                    message.showwarning("DB Error", "get_db_embeddings_callback(): Got zero subsampled tweets")
                    return

                # we have created subsamples, so set the flag to True and store them in the DB
                subsample_exists = True
                # Load up the subsampled_list and save
                si:SubsampleInfo
                d:Dict
                self.subsample_list = []
                for d in tweet_result:
                    si = SubsampleInfo(self.experiment_id, d['row_id'], d['query_id'], self.keyword_combo.get_text())
                    si.to_db(self.msi)
                    self.subsample_list.append(si)

        if subsample_exists == False:
            print("\tLoading full data from DB")
            query = 'select tweet_row, tweet_id, embedding, moderation, cluster_id, cluster_name, reduced from keyword_tweet_view where experiment_id = %s'
            values = (self.experiment_id,)
            if keyword != 'all_keywords':
                query = 'select tweet_row, tweet_id, text, embedding, moderation, cluster_id, cluster_name, reduced from keyword_tweet_view where experiment_id = %s and keyword = %s'
                values = (self.experiment_id, keyword)
            tweet_result = self.msi.read_data(query, values, True)
            print("\t Loaded {} records from DB".format(len(tweet_result)))
        row_dict:Dict

        print("\tLoading embedding parameters")
        sql = "select * from table_embedding_params where experiment_id = %s and keyword = %s"
        vals = (self.experiment_id, self.keyword_combo.get_text())
        results = self.msi.read_data(sql, vals)
        for d in results:
            print("\t{}".format(d))
        if len(results) > 0:
            row_dict = results[0]
            self.pca_dim_param.set_text(self.safe_dict(row_dict, 'PCA_dim', self.pca_dim_param.get_as_int()))
            self.eps_param.set_text(self.safe_dict(row_dict, 'EPS', self.eps_param.get_as_float()))
            self.min_samples_param.set_text(self.safe_dict(row_dict, 'min_samples', self.min_samples_param.get_as_int()))
            self.perplexity_param.set_text(self.safe_dict(row_dict, 'perplexity', self.perplexity_param.get_as_int()))

        print("\tClearing ManifoldReduction")
        self.mr.clear()
        self.canvas_frame.clear_Nodes()
        print("\tLoading {} rows".format(len(tweet_result)))
        count = 0
        et:EmbeddedText
        moderation_list = []
        for row_dict in tweet_result:
            if 'tweet_row' in row_dict:
                row_id = row_dict['tweet_row']
            else:
                row_id = row_dict['row_id']
            #row_id = self.safe_dict(row_dict, 'tweet_row', row_dict['row_id'])
            et = self.mr.load_row(row_id, row_dict['embedding'], None, None)
            et.text = self.safe_dict(row_dict, 'text', "unset")
            reduced = self.safe_dict(row_dict, 'reduced', None)
            cluster_id = self.safe_dict(row_dict, 'cluster_id', None)
            cluster_name = self.safe_dict(row_dict, 'cluster_name', None)
            et.set_optional(reduced, cluster_id, cluster_name)
            mod = self.safe_dict(row_dict, 'moderation', None)
            if mod != None:
                jmod = json.loads(mod)
                jmod['text'] = et.text
                moderation_list.append(jmod)
            if count % 1000 == 0:
                self.dp.dprint("loaded {} of {} records".format(count, len(tweet_result)))
            count += 1

        if len(moderation_list) > 0:
            self.speech_df = pd.DataFrame(moderation_list)
            stats = self.speech_df.agg(['mean', 'std']).T
            self.speech_text_field.clear()
            self.speech_text_field.set_text(stats.to_string())
        else:
            self.speech_df = pd.DataFrame()

        self.mr.calc_xy_range()


        for i in range(10):
            et = self.mr.embedding_list[i]
            print(et.to_string())

        message.showinfo("get_db_embeddings_callback", "Finished loading {} rows".format(len(self.mr.embedding_list)))

        print("\tFinished loading")

    def reduce_dimensions_callback(self):
        pca_dim = self.pca_dim_param.get_as_int()
        perplexity = self.perplexity_param.get_as_int()
        self.dp.dprint("Reducing: PCA dim = {}  perplexity = {}".format(pca_dim, perplexity))
        self.mr.calc_embeding(perplexity=perplexity, pca_components=pca_dim)
        print("\tFinished dimension reduction")
        message.showinfo("reduce_dimensions_callback", "Reduced to {} dimensions".format(pca_dim))

    def cluster_callback(self):
        print("Clustering")
        eps = self.eps_param.get_as_float()
        min_samples = self.min_samples_param.get_as_int()
        self.mr.dbscan(eps=eps, min_samples=min_samples)
        self.dp.dprint("Finished clustering")

    def plot_callback(self):
        print("Plotting")
        title = self.keyword_combo.get_text()
        if title == 'all_keywords':
            title = self.experiment_combo.get_text()
        perplexity = self.perplexity_param.get_as_int()
        eps = self.eps_param.get_as_int()
        min_samples = self.min_samples_param.get_as_int()
        pca_dim = self.pca_dim_param.get_as_int()
        self.mr.plot("{}\ndim: {}, eps: {}, min_sample: {}, perplex = {}".format(
            title, pca_dim, eps, min_samples, perplexity))
        plt.show()

    def explore_callback(self):
        print("Exploring")
        et:EmbeddedText
        n:MovableNode
        color_list = list(mcolors.TABLEAU_COLORS.values())
        num_nodes = len(self.mr.embedding_list)
        self.dp.dprint("Explore: num_nodes = {}".format(num_nodes))
        if num_nodes == 0:
            return
        step = int(num_nodes / self.rows_param.get_as_int())
        print("\tstep = {}".format(step))
        #calculate the x, y scalar
        x_dist = self.mr.max_x - self.mr.min_x
        y_dist = self.mr.max_y - self.mr.min_y
        x_scale = self.canvas_frame.virtual_canvas_size / x_dist
        y_scale = self.canvas_frame.virtual_canvas_size / y_dist
        for i in range(0, num_nodes, step):
            et = self.mr.embedding_list[i]
            c = self.mr.get_cluster_color(et.cluster_id, color_list)
            x = et.reduced[0] * x_scale
            y = et.reduced[1] * y_scale
            if et.mnode == None:
                n = self.canvas_frame.create_MoveableNode(et.text, x=x, y=y, color=c, size = 2, show_name=False)
                et.mnode = n
            else:
                et.mnode.set_color(c)
        self.color_excluded_clusters()
        self.dp.dprint("Finished creating points")

    def selected_node_callback(self, node_id:int, msg:str):
        print("node_id = {}, msg = {}".format(node_id, msg))
        et:EmbeddedText
        for et in self.mr.embedding_list:
            if et.mnode != None:
                mn = et.mnode
                if mn.id == node_id:
                    self.exclude_cluster_field.clear()
                    self.exclude_cluster_field.set_text(str(et.cluster_id))
                    break

    def exclude_cluster_callback(self):
        print("exclude_cluster_callback")
        to_exclude = self.exclude_cluster_field.get_text()
        if to_exclude.isdigit():
            cluster_id = int(to_exclude)
            keyword = self.keyword_combo.get_text()
            experiment_id = self.experiment_id
            sql = "SELECT COUNT(*) FROM table_exclude where experiment_id = %s and cluster_id = %s and keyword = %s"
            vals = (experiment_id, cluster_id, keyword)
            result = self.msi.read_data(sql, vals)
            d:Dict = result[0]
            print(d)
            if d['COUNT(*)'] == 0:
                sql = "INSERT INTO table_exclude (experiment_id, cluster_id, keyword) VALUES (%s, %s, %s)"
                self.msi.write_sql_values_get_row(sql, vals, True)
                self.color_excluded_clusters()


    def label_clusters_callback(self):
        pass

    def get_oai_embeddings_callback(self, api_limit = 500, db_limit = 10000, debug = False):
        print("get_oai_embeddings_callback")
        if debug:
            api_limit = 10
            db_limit = 100
        keyword = self.keyword_combo.get_text()

        if self.experiment_id == -1 or len(keyword) < 2:
            message.showwarning("DB Error", "get_oai_embeddings_callback(): Please set database and/or keyword")
            return

        # do a big pull because pulls take a long time
        get_remaining_sql = "select tweet_row, text from keyword_tweet_view where experiment_id = %s and embedding is NULL limit %s"
        get_remaining_values = (self.experiment_id, db_limit)
        if keyword != 'all_keywords':
            get_remaining_sql = "select tweet_row, text from keyword_tweet_view where experiment_id = %s and keyword = %s and embedding is NULL limit %s"
            get_remaining_values = (self.experiment_id, keyword, db_limit)

        engine = self.engine_combo.get_text()
        print("get_embeddings_callback() Experiment id = {}, Keyword = {}, Engine = {}".format(self.experiment_id, keyword, engine))
        results = self.msi.read_data(get_remaining_sql, get_remaining_values)
        count = len(results)
        print("\tGetting embeddings and moderations for {} rows".format(len(results)))
        total = 0
        while len(results) > 0:
            d:Dict
            #chunk up the returned list to send off to the api
            for i in range(0, len(results), api_limit):
                # create a list of text
                s_list = []
                sub_results = []
                last_r = min(i+api_limit, len(results))
                for d in results[i:last_r]:
                    s_list.append(d['text'])
                    sub_results.append(d)
                print("First ({}/{}) = {}".format(i, sub_results[0]['tweet_row'], s_list[0][:40].strip()))
                print("Last ({}/{}) = {}".format(last_r, sub_results[-1]['tweet_row'], s_list[-1][:40].strip()))

                # send that list to get embeddings
                print("\tGetting embeddings for {} rows".format(len(sub_results)))
                embd_list = self.oai.get_embedding_list(s_list, engine)
                print("\tGetting moderations for {} rows".format(len(sub_results)))
                mod_list = self.oai.get_moderation_vals(s_list)
                row_dict:Dict
                print("\tUpdating DB")
                for j in range(len(sub_results)):
                    rd = sub_results[j]
                    tweet_row = rd['tweet_row']
                    d = embd_list[j]
                    embedding = d['embedding']
                    embd_s = np.array(embedding)
                    d = mod_list[j]
                    mods = d['category_scores']
                    mods_s = json.dumps(mods)
                    # print("row_id: {} text: {} embed: {}, mods = {}".format(tweet_row, tweet, embedding, mods_s))
                    sql = "update table_tweet set embedding = %s, moderation = %s where row_id = %s"
                    values = (embd_s.dumps(), mods_s, tweet_row)
                    if debug:
                        print("\tTweetEmbedExplorer.get_oai_embeddings_callback() Writing {}/{} ({})".format(i, j, i*api_limit + j))
                    else:
                        # print("\t[{}] row: {} embd: {}, mod: {}".format(total, tweet_row, embd_s, mods_s))
                        self.msi.write_sql_values_get_row(sql, values)
                        total += 1
                print("\tUpdated DB with {}/{} entries".format(len(sub_results), total+1))

            print("\tEmbedded {} records".format(count))
            if debug:
                print("\tbreaking early")
                break
            print("\tLooking for more NULL rows")
            results = self.msi.read_data(get_remaining_sql, get_remaining_values)
            print("\tGetting embeddings and moderations for {} rows".format(len(results)))
            count += len(results)
        print("TweetEmbedExplorer.get_oai_embeddings_callback(): Finished! Embedded a total of {} records".format(count))


    def get_keyword_entries_callback(self, keyword: str):
        print("get_keyword_entries: keyword = {}, experiment_id = {}".format(keyword, self.experiment_id))
        query = "select count(*) from keyword_tweet_view where experiment_id = %s"
        values = (self.experiment_id)
        if keyword != 'all_keywords':
            query = "select count(*) from keyword_tweet_view where experiment_id = %s and keyword = %s"
            values = (self.experiment_id, keyword)
        result:Dict = self.msi.read_data(query, values)[0]
        count = result['count(*)']

        self.keyword_count_field.set_text(count)
        self.rows_param.set_text(count)

    def keyword_callback(self, event:tk.Event):
        print("keyword_callback: event = {}".format(event))
        num_regex = re.compile(r"\d+")
        s = self.experiment_combo.tk_combo.get()
        self.experiment_combo.set_text(s)
        self.experiment_id = num_regex.findall(s)[0]
        print("keyword_callback: experiment_id = {}".format(self.experiment_id))
        query = "select distinct keyword from table_query where experiment_id = %s"
        values = (self.experiment_id,)
        result = self.msi.read_data(query, values)
        l = ['all_keywords']
        row_dict:Dict
        for row_dict in result:
            l.append(row_dict['keyword'])
        self.keyword_combo.set_combo_list(l)

    def plot_speech_data(self):
        plot_df = self.speech_df
        if 'text' in self.speech_df:
            plot_df = self.speech_df.drop(["text"], axis=1)
        print("plot_speech_data")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.boxplot(plot_df.values)
        ax.set_xticklabels(plot_df.columns)
        ax.set_xlabel('Variable')
        ax.set_ylabel('Value')
        title = self.keyword_combo.get_text()
        if title == 'all_keywords':
            title = self.experiment_combo.get_text()
        ax.set_title('[{}] moderated speech - {:,} results'.format(self.keyword_combo.get_text(), len(plot_df.index)))
        plt.show()

    def save_speech_data(self):
        print("save_speech_data")
        default = "{}_{}.xlsx".format(self.experiment_field.get_text(), self.keyword_combo.get_text())
        filename = filedialog.asksaveasfilename(filetypes=(("Excel files", "*.xlsx"),("All Files", "*.*")), title="Save Excel File", initialfile=default)
        if filename:
            print("saving to {}".format(filename))
            with pd.ExcelWriter(filename) as writer:
                self.speech_df.to_excel(writer, index=False)

    def setup(self):
        # set up the canvas
        self.canvas_frame.setup(debug=False, show_names=False)

        # set up the selections that come from the db
        l = []
        row_dict:Dict
        query = "select * from table_experiment"
        result = self.msi.read_data(query)
        for row_dict in result:
            s = "{}: {}".format(row_dict['id'], row_dict['keywords'])
            l.append(s)
        self.experiment_combo.set_combo_list(l)


def main():
    app = EmbeddingsExplorer()
    app.setup()
    app.mainloop()


if __name__ == "__main__":
    main()
