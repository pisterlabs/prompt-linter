'''
Load - Loads an existing experiment
Model Selection - Various models of the GPT-3 or a local model
Prompt - "Once upon a time there was." or potentially much longer, paragraph-sized so lots of room. Also UTF-8 to handle other languages
Parameters - number of tokens, etc.
Run - sends the prompt to the GPT and gets the text response. A checkbox indicates if there should be automatic embedding
Clear embeddings
Get Embeddings
Extend - uses the existing prompt and response as a prompt
Cluster - happens on line boundaries. There should be an editable regex for that. Same sort of PCA/T-SNE as embedding explorer, which means there needs to be parameter tweaking. Clustering will have to be re-run multiple times, though I hope the embedding step is run once. To avoid the complexity of the interactive plotting, I think I'll just label the clusters (https://stackoverflow.com/questions/44998205/labeling-points-in-matplotlib-scatterplot)
Query - 1) Run cluster queries on the DB. Select the cluster ID and the number of responses. 2) Get the number of responses per cluster
Save - stores the text on a sentence-by sentence bases with clustering info
Generate Graph - runs through each narrative in an experiment to produce a directed graph of nodes. The output is all the narratives threaded together. Used as an input to Gephi
'''
import os
import re
import getpass
import math
import numpy as np
import tkinter as tk
from tkinter import ttk
import tkinter.messagebox as message
from datetime import datetime
from tkinter import filedialog
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors

import pandas as pd

from keyword_explorer.Apps.AppBase import AppBase
from keyword_explorer.tkUtils.Buttons import Buttons
from keyword_explorer.tkUtils.ToolTip import ToolTip
from keyword_explorer.tkUtils.GPT3GeneratorFrame import GPT3GeneratorSettings, GPT3GeneratorFrame
from keyword_explorer.tkUtils.GPT3EmbeddingFrame import GPT3EmbeddingSettings, GPT3EmbeddingFrame
from keyword_explorer.tkUtils.ListField import ListField
from keyword_explorer.tkUtils.TextField import TextField
from keyword_explorer.tkUtils.DataField import DataField
from keyword_explorer.tkUtils.TopicComboExt import TopicComboExt
from keyword_explorer.tkUtils.LabeledParam import LabeledParam
from keyword_explorer.OpenAI.OpenAIComms import OpenAIComms
from keyword_explorer.utils.MySqlInterface import MySqlInterface
from keyword_explorer.utils.ManifoldReduction import ManifoldReduction, EmbeddedText, ClusterInfo
from keyword_explorer.utils.SharedObjects import SharedObjects
from keyword_explorer.utils.NetworkxGraphing import NetworkxGraphing

from typing import List, Dict

class NarrativeExplorer2(AppBase):
    oai: OpenAIComms
    msi: MySqlInterface
    mr: ManifoldReduction
    so:SharedObjects
    generator_frame: GPT3GeneratorFrame
    embedding_frame: GPT3EmbeddingFrame
    experiment_combo: TopicComboExt
    recurse_depth_list:ListField
    style_list:ListField
    new_experiment_button:Buttons
    embed_state_text_field:TextField
    embedded_field:DataField
    reduced_field:DataField
    experiment_id:int
    run_id:int

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("NarrativeExplorer")
        self.text_width = 60
        self.label_width = 15

        dt = datetime.now()
        experiment_str = "{}_{}_{}".format(self.app_name, getpass.getuser(), dt.strftime("%H:%M:%S"))
        self.experiment_field.set_text(experiment_str)
        self.load_experiment_list()

        self.so.add_object("experiment_field", self.experiment_field, DataField)
        self.so.add_object("clusters_field", self.clusters_field, DataField)
        self.so.add_object("reduced_field", self.reduced_field, DataField)
        self.so.print_contents()
        # self.test_data_callback()

    def setup_app(self):
        self.app_name = "NarrativeExplorer2"
        self.app_version = "3.30.2023"
        self.geom = (870, 670)
        self.oai = OpenAIComms()
        self.msi = MySqlInterface(user_name="root", db_name="narrative_maps")
        self.mr = ManifoldReduction()
        self.so = SharedObjects()

        if not self.oai.key_exists():
            message.showwarning("Key Error", "Could not find Environment key 'OPENAI_KEY'")

        self.experiment_id = -1
        self.run_id = -1

    def build_app_view(self, row: int, text_width: int, label_width: int) -> int:
        print("build_app_view")
        self.generator_frame = GPT3GeneratorFrame(self.oai, self.dp, self.so)
        self.embedding_frame = GPT3EmbeddingFrame(self.oai, self.mr, self.dp, self.so)
        lf = tk.LabelFrame(self, text="GPT")
        lf.grid(row=row, column=0, columnspan = 2, sticky="nsew", padx=5, pady=2)
        self.build_gpt(lf, text_width, label_width)

        lf = tk.LabelFrame(self, text="Params")
        lf.grid(row=row, column=2, sticky="nsew", padx=5, pady=2)
        self.build_params(lf, int(text_width/3), int(label_width*.75))

        return row + 1

    def build_menus(self):
        print("building menus")
        self.option_add('*tearOff', tk.FALSE)
        menubar = tk.Menu(self)
        self['menu'] = menubar
        menu_file = tk.Menu(menubar)
        menubar.add_cascade(menu=menu_file, label='File')
        menu_file.add_command(label='Load params', command=self.load_params_callback)
        menu_file.add_command(label='Save params', command=self.save_params_callback)
        menu_file.add_command(label='Load IDs', command=self.load_ids_callback)
        menu_file.add_command(label='Test data', command=self.generator_frame.test_data_callback)
        menu_file.add_command(label='Exit', command=self.terminate)

    def load_experiment_list(self):
        experiments = []
        results = self.msi.read_data("select name from table_experiment")
        for r in results:
            experiments.append(r['name'])
        self.experiment_combo.set_combo_list(experiments)

    def build_gpt(self, lf:tk.LabelFrame, text_width:int, label_width:int):
        row = 0
        self.experiment_combo = TopicComboExt(lf, row, "Saved Experiments:", self.dp, entry_width=20, combo_width=20)
        self.experiment_combo.set_callback(self.load_experiment_callback)
        row = self.experiment_combo.get_next_row()
        self.new_experiment_button = Buttons(lf, row, "Experiments")
        b = self.new_experiment_button.add_button("Create", self.create_experiment_callback)
        ToolTip(b, "Create a new, named experiment")
        b = self.new_experiment_button.add_button("Load", self.load_experiment_callback)
        ToolTip(b, "Load an existing experiment")
        b = self.new_experiment_button.add_button("Update", self.update_experiment_callback)
        ToolTip(b, "Update an existing experiment")
        row = self.new_experiment_button.get_next_row()

        s = ttk.Style()
        s.configure('TNotebook.Tab', font=self.default_font)

        # Add the tabs
        tab_control = ttk.Notebook(lf)
        tab_control.grid(column=0, row=row, columnspan=2, sticky="nsew")
        gpt_tab = ttk.Frame(tab_control)
        tab_control.add(gpt_tab, text='Generate')
        self.build_generator_tab(gpt_tab, text_width, label_width)

        embed_tab = ttk.Frame(tab_control)
        tab_control.add(embed_tab, text='Embedding')
        self.build_embed_tab(embed_tab, text_width, label_width)

        row += 1
        return row

    def build_params(self, lf:tk.LabelFrame, text_width:int, label_width:int):
        row = 0
        self.runs_field = DataField(lf, row, 'Runs:', text_width, label_width=label_width)
        row = self.runs_field.get_next_row()
        self.parsed_field = DataField(lf, row, 'Parsed:', text_width, label_width=label_width)
        row = self.parsed_field.get_next_row()
        self.embedded_field = DataField(lf, row, 'Embeds:', text_width, label_width=label_width)
        row = self.embedded_field.get_next_row()
        self.reduced_field = DataField(lf, row, 'Reduced:', text_width, label_width=label_width)
        row = self.reduced_field.get_next_row()
        self.clusters_field = DataField(lf, row, 'Clusters:', text_width, label_width=label_width)
        row = self.clusters_field.get_next_row()
        self.style_list = ListField(lf, row, "Style", width=text_width, label_width=label_width, static_list=True)
        self.style_list.set_text(text='Story, List, Sequence')
        self.style_list.set_callback(self.set_style_callback)
        ToolTip(self.style_list.tk_list, "Creates a narrative or recursive context for generation - Not implemented")
        row = self.style_list.get_next_row()
        self.recurse_depth_list = ListField(lf, row, "Depth\n(2)", width=text_width, label_width=label_width, static_list=True)
        self.recurse_depth_list.set_text(text='2, 3, 4, 5')
        self.recurse_depth_list.set_callback(self.set_recurse_depth_callback)
        ToolTip(self.recurse_depth_list.tk_list, "The  recursion depth for lists - Not implemented")
        row = self.recurse_depth_list.get_next_row()

    def build_generator_tab(self, tab: ttk.Frame, text_width:int, label_width:int):
        self.generator_frame.build_frame(tab, text_width, label_width)
        self.generator_frame.add_button("Save", self.save_text_list_callback, "Manually saves the result to the database")
        self.generator_frame.add_button("Automate", self.automate_callback, "Automatically runs probes, parses, and stores the results\n the number of times in the 'Run Count' field")

    def build_embed_tab(self, tab: ttk.Frame, text_width:int, label_width:int):
        self.embedding_frame.build_frame(tab, text_width, label_width)
        self.embedding_frame.add_button("GPT embed", self.get_oai_embeddings_callback, "Get source embeddings from the GPT")
        self.embedding_frame.add_button("Retreive", self.get_db_embeddings_callback, "Get the high-dimensional embeddings from the DB")
        self.embedding_frame.add_button("Export", self.export_network_callback, "Plot and Export the graph to GML")

    def build_embed_params(self, parent:tk.Frame, row:int) -> int:
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
        return row + 1

    def count_parsed(self, experiment_id):
        sql = "select * from parsed_view where experiment_id = %s"
        vals = (experiment_id,)
        results = self.msi.read_data(sql, vals)
        run_set = set()
        parsed_set = set()
        embed_set = set()
        reduced_set = set()
        clusters_set = set()
        d:Dict
        for d in results:
            if d['run_index'] != None:
                run_set.add(d['run_index'])
            if d['embedding'] != None:
                embed_set.add(d['embedding'])
            if d['parsed_text'] != None:
                parsed_set.add(d['parsed_text'])
            if d['mapped'] != None:
                reduced_set.add(d['mapped'])
            if d['cluster_id'] != None:
                clusters_set.add(d['cluster_id'])

        self.runs_field.set_text(len(run_set))
        self.parsed_field.set_text(len(parsed_set))
        self.embedded_field.set_text(len(embed_set))
        self.reduced_field.set_text(len(reduced_set))
        self.clusters_field.set_text(len(clusters_set))

    def get_list(self, to_parse:str, regex_str:str = ",") -> List:
        rlist = re.split(regex_str, to_parse)
        to_return = []
        for t in rlist:
            if t != None:
                to_return.append(t.strip())
        to_return = [x for x in to_return if x] # filter out the blanks
        return to_return

    def set_style_callback(self, event:tk.Event = None):
        style_str = self.style_list.get_selected()
        self.style_list.set_label("Style\n({})".format(style_str))
        if style_str == "Story":
            print("Set story regex")
        elif style_str == "List":
            print("Set list regex")
        elif style_str == "Sequence":
            print("Set Sequence regex")

    def set_recurse_depth_callback(self, event:tk.Event = None):
        depth_str = self.recurse_depth_list.get_selected()
        self.recurse_depth_list.set_label("Depth\n({})".format(depth_str))

    def create_experiment_callback(self):
        print("create_experiment_callback")
        cur_date = datetime.now()
        experiment_name = self.experiment_field.get_text()

        if self.app_name in experiment_name:
            result = tk.messagebox.askyesno("Warning!", "You are about to creat an experiment\nwith the default name{}\nProceed?".format(experiment_name))
            print("result = {}".format(result))
            if not result:
                return
        if "experiment" in experiment_name:
            tk.messagebox.showwarning("Duplicate Experiment", "{} exists in db".format(experiment_name))
            return

        sql = "insert into table_experiment (name, date) values (%s, %s)"
        vals = (experiment_name, cur_date)
        self.experiment_id = self.msi.write_sql_values_get_row(sql, vals)
        self.load_experiment_list()
        self.experiment_combo.clear()
        self.experiment_combo.set_text(experiment_name)

    def load_experiment_callback(self, event = None):
        print("load_experiment_callback")
        s = self.experiment_combo.tk_combo.get()
        self.experiment_combo.clear()
        self.experiment_combo.set_text(s)
        results = self.msi.read_data("select id from table_experiment where name = %s", (s,))
        if len(results) > 0:
            self.experiment_id = results[0]['id']
            self.experiment_field.set_text(" experiment {}: {}".format(self.experiment_id, s))
        print("experiment_callback: experiment_id = {}".format(self.experiment_id))
        if self.experiment_id != -1:
            sql = "select MAX(run_id) as max from table_run where experiment_id = %s"
            vals = (self.experiment_id,)
            results = self.msi.read_data(sql, vals)
            print(results)
            self.generator_frame.set_prompt("No prompt available in database")
            self.runs_field.set_text('0')
            self.count_parsed(self.experiment_id)
            if results[0]['max'] != None:
                run_id = results[0]['max']
                #self.runs_field.set_text(run_id)
                sql = "select * from run_params_view where run_id = %s and experiment_id = %s"
                vals = (run_id, self.experiment_id)
                results = self.msi.read_data(sql, vals)
                d = results[0]

                ggs = GPT3GeneratorSettings()
                ggs.from_dict(d)
                self.generator_frame.set_params(ggs)

                ges = GPT3EmbeddingSettings()
                ges.from_dict(d)
                self.embedding_frame.set_params(ges)


    def update_experiment_callback(self):
        print("update_experiment_callback()")
        if self.experiment_id == -1:
            result = tk.messagebox.showwarning("Warning!", "Please create or select a database first")
            return
        params = self.get_current_params()
        # update the table_embedding_params for this experiment/runs
        sql = "select distinct emb_id from index_view where experiment_id = %s"
        vals = (self.experiment_id,)
        results = self.msi.read_data(sql, vals)
        d:Dict
        for d in results:
            embed_id = d['emb_id']
            sql = "update table_embedding_params set model = %s, PCA_dim = %s, EPS = %s, min_samples = %s, perplexity = %s where id = %s"
            vals = (params['embedding_model'], params['PCA_dimensions'], params['EPS'], params['min_samples'], params['perplexity'], embed_id)
            self.msi.write_sql_values_get_row(sql, vals)

        # update table_parsed_text with the reduced/mapped data
        et: EmbeddedText
        for et in self.mr.embedding_list:
            reduced_s = np.array(et.reduced).dumps()
            sql = "update table_parsed_text set mapped = %s, cluster_id = %s, cluster_name = %s where id = %s"
            vals = (reduced_s, int(et.cluster_id), et.cluster_name, int(et.row_id))
            self.msi.write_sql_values_get_row(sql, vals)

        self.count_parsed(self.experiment_id)


    def save_text_list_callback(self):
        print("NarrativeExplorer2.save_text_list_callback()")

        if self.experiment_id == -1:
            result = tk.messagebox.showwarning("Warning!", "Please create or select a database first")
            return
        gf = self.generator_frame
        ef = self.embedding_frame
        print("\tSaving {} lines".format(len(gf.parsed_full_text_list)))
        if len(gf.parsed_full_text_list) > 0:
            # create the run
            run_id = 1
            sql = "select MAX(run_id) as max from table_run where experiment_id = %s"
            vals = (self.experiment_id,)
            results = self.msi.read_data(sql, vals)
            print(results)
            if results[0]['max'] != None:
                run_id = results[0]['max'] + 1
            # get the language model params entry
            gs = gf.get_settings()
            sql = "insert into table_generate_params (tokens, temp, presence_penalty, frequency_penalty, model, automated_runs) values (%s, %s, %s, %s, %s, %s)"
            vals = (gs.tokens, gs.temperature, gs.presence_penalty, gs.frequency_penalty, gs.model, gs.auto_runs)
            lang_param_id = self.msi.write_sql_values_get_row(sql, vals)

            # get the embedding model entry
            es = ef.get_settings()
            sql = "insert into table_embedding_params (model, PCA_dim, EPS, min_samples, perplexity) values (%s, %s, %s, %s, %s)"
            vals = (es.model, es.pca_dim, es.eps, es.min_samples, es.perplexity)
            embed_param_id = self.msi.write_sql_values_get_row(sql, vals)

            sql = "insert into table_run (experiment_id, run_id, prompt, response, generator_params, embedding_params) values (%s, %s, %s, %s, %s, %s)"
            vals = (self.experiment_id, run_id, gf.saved_prompt_text,
                    gf.saved_response_text, lang_param_id, embed_param_id)
            run_index = self.msi.write_sql_values_get_row(sql, vals)

            # store the text
            s:str
            for s in gf.parsed_full_text_list:
                sql = "insert into table_parsed_text (run_index, parsed_text) values (%s, %s)"
                vals = (run_index, s)
                self.msi.write_sql_values_get_row(sql, vals)
        else:
            result = tk.messagebox.showwarning("Warning!", "There are no parsed lines")
            return

        #reset the list
        gf.parsed_full_text_list = []

    def automate_callback(self):
        print("automate_callback():")
        if self.experiment_id == -1:
            result = tk.messagebox.showwarning("Warning!", "Please create or select a database first")
            return

        gf = self.generator_frame
        num_runs = gf.auto_field.get_as_int()
        parsed_lines = 0
        for i in range(num_runs):
            prompt = gf.prompt_text_field.get_text()
            print("{}: prompting: ...{}".format(i, prompt[-80:]))
            gf.new_prompt_callback()
            response = gf.response_text_field.get_text()
            print("\tgetting response: {}...".format(response[:80]))
            print("\tparsing response")
            gf.parse_response_callback()
            parsed_lines += len(gf.parsed_full_text_list)
            print("\tstoring data")
            self.save_text_list_callback()
            print("\tresetting")
            gf.parsed_full_text_list = []
            gf.response_text_field.clear()
            self.runs_field.set_text(str(i+1))
            self.parsed_field.set_text(str(parsed_lines))
        print("done")
        tk.messagebox.showinfo("Automate", "Finished processing {} rows".format(parsed_lines))

    def get_oai_embeddings_callback(self):
        print("get_oai_embeddings_callback")
        if self.experiment_id == -1:
            tk.messagebox.showwarning("Warning!", "Please create or select a database first")
            return
        # get all the embeddings for text that we don't have yet
        sql = "select experiment_id, id, parsed_text, embedding_model from parsed_view where experiment_id = %s and embedding IS NULL"
        vals = (self.experiment_id,)
        results = self.msi.read_data(sql, vals)
        d:Dict
        # create a list of text
        s_list = []
        for d in results:
            s_list.append(d['parsed_text'])

        # send that list to get embeddings
        engine = results[0]['embedding_model']
        d_list = self.oai.get_embedding_list(s_list, engine)

        # store embeddings
        for i in range(len(results)):
            rd = results[i]
            id = rd['id']
            d = d_list[i]
            embedding = d['embedding']
            text = d['text']
            embd_s = np.array(embedding)
            sql = "update table_parsed_text set embedding = %s where id = %s"
            vals = (embd_s.dumps(), id)
            self.msi.write_sql_values_get_row(sql, vals)

            print("[{}]: {} [{}]".format(id, text, embd_s))

        tk.messagebox.showinfo("get_oai_embeddings_callback", "Finished embedding {} rows".format(len(results)))


    def get_db_embeddings_callback(self):
        print("get_db_embeddings_callback")
        if self.experiment_id == -1:
            message.showwarning("DB Error", "get_db_embeddings_callback(): Please set database")
            return

        print("Loading from DB")
        print("\tClearing ManifoldReduction")
        self.mr.clear()
        sql = "select * from parsed_view where experiment_id = %s"
        vals = (self.experiment_id,)
        results = self.msi.read_data(sql, vals)
        d:Dict
        et:EmbeddedText
        for d in results:
            embed_s = d['embedding']
            id = d['id']
            et = self.mr.load_row(id, embed_s, None, None)
            et.text = self.safe_dict_read(d, 'parsed_text', 'unset')
            mapped = self.safe_dict_read(d, 'mapped', None)
            cluster_id = self.safe_dict_read(d, 'cluster_id', None)
            cluster_name = self.safe_dict_read(d, 'cluster_name', "clstr_{}".format(cluster_id))
            run_id = self.safe_dict_read(d, 'run_id', None)
            et.set_optional(mapped, cluster_id, cluster_name, run_id)
            print(et.to_string())
        self.mr.calc_clusters()
        message.showinfo("get_db_embeddings_callback", "Loaded ({:,}) embeddings".format(len(results)))

    def get_current_params(self) -> Dict:
        generator_settings = self.generator_frame.get_settings()
        embed_settings = self.embedding_frame.get_settings()
        d = {
            "probe_str": generator_settings.prompt,
            "name": self.experiment_field.get_text(),
            "automated_runs": generator_settings.auto_runs,
            "generate_model": generator_settings.model,
            "tokens": generator_settings.tokens,
            "temp": generator_settings.temperature,
            "presence_penalty": generator_settings.presence_penalty,
            "frequency_penalty": generator_settings.frequency_penalty,
            "embedding_model": embed_settings.model,
            "PCA_dimensions": embed_settings.pca_dim,
            "EPS": embed_settings.eps,
            "min_samples": embed_settings.min_samples,
            "perplexity": embed_settings.perplexity
        }
        return d

    def load_params_callback(self):
        defaults = self.get_current_params()

        param_dict = self.load_json(defaults)
        # print(param_dict)
        gs = GPT3GeneratorSettings()
        gs.from_dict(param_dict)
        self.generator_frame.set_params(gs)

        gs = GPT3EmbeddingSettings(defaults)
        gs.from_dict(param_dict)
        self.embedding_frame.set_params(gs)

        self.experiment_field.clear()

        if 'name' in param_dict:
            self.experiment_field.set_text(param_dict['name'])

    def save_params_callback(self):
        params = self.get_current_params()
        self.save_experiment_json(params)

    def export_network_callback(self):
        print("NarrativeExplorer2.export_network_callback()")
        if self.experiment_id == -1:
            message.showwarning("DB Error", "get_db_embeddings_callback(): Please set database")
            return
        user = os.getlogin()
        ng = NetworkxGraphing(name=self.experiment_field.get_text(), creator=user)

        sql = "select id, experiment_id, run_id from table_run where experiment_id = %s"
        vals = (self.experiment_id,)

        runs_response = self.msi.read_data(sql, vals)
        runs_d:Dict
        parsed_d1:Dict
        parsed_d2:Dict
        weight_dict = {}
        for runs_d in runs_response:
            run_index = int(runs_d['id'])
            sql = "select id, cluster_name, parsed_text from table_parsed_text where run_index = %s order by id"
            vals = (run_index,)
            parsed_response = self.msi.read_data(sql, vals)
            for i in range(len(parsed_response) - 1):
                parsed_d1 = parsed_response[i]
                parsed_d2 = parsed_response[i+1]
                cl = str(parsed_d1['cluster_name']).split("\n")
                cl = [x.strip() for x in cl]
                source = "-".join(cl).strip()
                cl = str(parsed_d2['cluster_name']).split("\n")
                cl = [x.strip() for x in cl]
                target = "-".join(cl).strip()
                if source != target:
                    if source in weight_dict:
                        weight_dict[source] += 1
                    else:
                        weight_dict[source] = 1
                    if target in weight_dict:
                        weight_dict[target] += 1
                    else:
                        weight_dict[target] = 1

        for key, val in weight_dict.items():
            print("{} = {}".format(key, val))

        for runs_d in runs_response:
            run_index = int(runs_d['id'])
            sql = "select id, cluster_name, parsed_text from table_parsed_text where run_index = %s order by id"
            vals = (run_index,)
            parsed_response = self.msi.read_data(sql, vals)
            for i in range(len(parsed_response) - 1):
                parsed_d1 = parsed_response[i]
                parsed_d2 = parsed_response[i+1]
                cl = str(parsed_d1['cluster_name']).split("\n")
                cl = [x.strip() for x in cl]
                source = "-".join(cl).strip()
                cl = str(parsed_d2['cluster_name']).split("\n")
                cl = [x.strip() for x in cl]
                target = "-".join(cl).strip()
                if source != target:
                    # print("[{}] source: {}, target: {}".format(runs_d['run_id'], source, target))
                    ng.add_weighted_nodes(source, weight_dict[source], target, weight_dict[target])
        #ng.draw(self.experiment_field.get_text(), draw_legend=False, do_show=True, scalar=2.0)
        result = filedialog.asksaveasfile(filetypes=(("gml files", "*.gml"),("All Files", "*.*")), title="Save/Update network", initialfile = "{}.gml".format(self.experiment_field.get_text()))
        if result:
            filename = result.name
            ng.to_gml(filename, graph_creator=user, node_attributes=['weight'])
            self.dp.dprint("Saved network to {}".format(filename))




def main():
    app = NarrativeExplorer2()
    app.mainloop()

if __name__ == "__main__":
    main()