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

import re
import getpass
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
from keyword_explorer.tkUtils.DateEntryField import DateEntryField
from keyword_explorer.tkUtils.ListField import ListField
from keyword_explorer.tkUtils.TextField import TextField
from keyword_explorer.tkUtils.DataField import DataField
from keyword_explorer.tkUtils.TopicComboExt import TopicComboExt

from keyword_explorer.OpenAI.OpenAIComms import OpenAIComms
from keyword_explorer.utils.MySqlInterface import MySqlInterface
from keyword_explorer.utils.ManifoldReduction import ManifoldReduction, EmbeddedText, ClusterInfo
from keyword_explorer.tkUtils.LabeledParam import LabeledParam

from typing import List, Dict

class NarrativeExplorer(AppBase):
    oai: OpenAIComms
    msi: MySqlInterface
    mr: ManifoldReduction
    embed_model_combo: TopicComboExt
    generate_model_combo: TopicComboExt
    tokens_param: LabeledParam
    temp_param: LabeledParam
    presence_param: LabeledParam
    frequency_param: LabeledParam
    experiment_combo: TopicComboExt
    new_experiment_button:Buttons
    pca_dim_param: LabeledParam
    eps_param: LabeledParam
    min_samples_param: LabeledParam
    perplexity_param: LabeledParam
    prompt_text_field:TextField
    response_text_field:TextField
    embed_state_text_field:TextField
    regex_field:DataField
    auto_field:DataField
    runs_field:DataField
    parsed_field:DataField
    embedded_field:DataField
    reduced_field:DataField
    saved_prompt_text:str
    saved_response_text:str
    experiment_id:int
    run_id:int
    parsed_full_text_list:List

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("NarrativeExplorer")
        self.text_width = 60
        self.label_width = 15

        dt = datetime.now()
        experiment_str = "{}_{}_{}".format(self.app_name, getpass.getuser(), dt.strftime("%H:%M:%S"))
        self.experiment_field.set_text(experiment_str)
        self.load_experiment_list()
        # self.test_data_callback()

    def setup_app(self):
        self.app_name = "NarrativeExplorer"
        self.app_version = "3.15.2023"
        self.geom = (840, 670)
        self.oai = OpenAIComms()
        self.msi = MySqlInterface(user_name="root", db_name="narrative_maps")
        self.mr = ManifoldReduction()

        if not self.oai.key_exists():
            message.showwarning("Key Error", "Could not find Environment key 'OPENAI_KEY'")

        self.saved_prompt_text = "unset"
        self.saved_response_text = "unset"
        self.experiment_id = -1
        self.run_id = -1
        self.parsed_full_text_list = []


    def build_app_view(self, row: int, text_width: int, label_width: int) -> int:
        print("build_app_view")
        lf = tk.LabelFrame(self, text="GPT")
        lf.grid(row=row, column=0, columnspan = 2, sticky="nsew", padx=5, pady=2)
        self.build_gpt(lf, text_width, label_width)

        lf = tk.LabelFrame(self, text="Params")
        lf.grid(row=row, column=2, sticky="nsew", padx=5, pady=2)
        self.build_params(lf, int(text_width/3), int(label_width/2))

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
        menu_file.add_command(label='Test data', command=self.test_data_callback)
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
        buttons = Buttons(lf, row, "Experiments")
        b = buttons.add_button("Create", self.create_experiment_callback)
        ToolTip(b, "Create a new, named experiment")
        b = buttons.add_button("Load", self.load_experiment_callback)
        ToolTip(b, "Load an existing experiment")
        b = buttons.add_button("Update", self.update_experiment_callback)
        ToolTip(b, "Update an existing experiment")
        row = buttons.get_next_row()

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

    def build_generator_tab(self, tab: ttk.Frame, text_width:int, label_width:int):
        engine_list = self.oai.list_models(exclude_list = ["embed", "similarity", "code", "edit", "search", "audio", "instruct", "2020", "if", "insert", "whisper"])
        engine_list = sorted(engine_list)
        row = 0
        self.generate_model_combo = TopicComboExt(tab, row, "Model:", self.dp, entry_width=25, combo_width=25)
        self.generate_model_combo.set_combo_list(engine_list)
        self.generate_model_combo.set_text(engine_list[0])
        self.generate_model_combo.tk_combo.current(0)
        ToolTip(self.generate_model_combo.tk_combo, "The GPT-3 model used to generate text")

        row = self.generate_model_combo.get_next_row()
        row = self.build_generate_params(tab, row)

        self.prompt_text_field = TextField(tab, row, "Prompt:", text_width, height=6, label_width=label_width)
        self.prompt_text_field.set_text("Once upon a time there was")
        ToolTip(self.prompt_text_field.tk_text, "The prompt that the GPT will use to generate text from")
        row = self.prompt_text_field.get_next_row()

        self.response_text_field = TextField(tab, row, 'Response:', text_width, height=11, label_width=label_width)
        ToolTip(self.response_text_field.tk_text, "The response from the GPT will be displayed here")
        row = self.response_text_field.get_next_row()

        self.regex_field = DataField(tab, row, 'Parse regex:', text_width, label_width=label_width)
        self.regex_field.set_text(r"\n|[\.!?] |([\.!?]\")")
        ToolTip(self.regex_field.tk_entry, "The regex used to parse the GPT response. Editable")
        row = self.regex_field.get_next_row()

        self.auto_field = DataField(tab, row, 'Run count:', text_width, label_width=label_width)
        self.auto_field.set_text("10")
        ToolTip(self.auto_field.tk_entry, "The number of times the prompt will be run by 'Automate'")
        row = self.auto_field.get_next_row()

        buttons = Buttons(tab, row, "Actions")
        b = buttons.add_button("Generate", self.new_prompt_callback)
        ToolTip(b, "Sends the prompt to the GPT")
        b = buttons.add_button("Add", self.extend_prompt_callback)
        ToolTip(b, "Adds the response to the prompt")
        b = buttons.add_button("Parse", self.parse_response_callback)
        ToolTip(b, "Parses the response into a list for embeddings")
        b = buttons.add_button("Save", self.save_text_list_callback)
        ToolTip(b, "Manually saves the result to the database")
        b = buttons.add_button("Automate", self.automate_callback)
        ToolTip(b, "Automatically runs probes, parses, and stores the results\n the number of times in the 'Run Count' field")

    def build_embed_tab(self, tab: ttk.Frame, text_width:int, label_width:int):
        engine_list = self.oai.list_models(keep_list = ["embedding"])
        row = 0
        self.embed_model_combo = TopicComboExt(tab, row, "Engine:", self.dp, entry_width=25, combo_width=25)
        self.embed_model_combo.set_combo_list(engine_list)
        self.embed_model_combo.set_text(engine_list[0])
        self.embed_model_combo.tk_combo.current(0)
        row = self.embed_model_combo.get_next_row()
        row = self.build_embed_params(tab, row)
        self.embed_state_text_field = TextField(tab, row, "Embed state:", text_width, height=10, label_width=label_width)
        ToolTip(self.embed_state_text_field.tk_text, "Embedding progess")
        row = self.embed_state_text_field.get_next_row()
        buttons = Buttons(tab, row, "Commands", label_width=10)
        b = buttons.add_button("GPT embed", self.get_oai_embeddings_callback, -1)
        ToolTip(b, "Get source embeddings from the GPT")
        b = buttons.add_button("Retreive", self.get_db_embeddings_callback, -1)
        ToolTip(b, "Get the high-dimensional embeddings from the DB")
        b = buttons.add_button("Reduce", self.reduce_dimensions_callback, -1)
        ToolTip(b, "Reduce to 2 dimensions with PCS and TSNE")
        b = buttons.add_button("Cluster", self.cluster_callback, -1)
        ToolTip(b, "Compute clusters on reduced data")
        b = buttons.add_button("Plot", self.plot_callback, -1)
        ToolTip(b, "Plot the clustered points using PyPlot")
        b = buttons.add_button("Topics", self.topic_callback, -1)
        ToolTip(b, "Use GPT to guess at topic names for clusters")
        row = buttons.get_next_row()

    def build_generate_params(self, parent:tk.Frame, row:int) -> int:
        f = tk.Frame(parent)
        f.grid(row=row, column=0, columnspan=2, sticky="nsew", padx=1, pady=1)

        self.tokens_param = LabeledParam(f, 0, "Tokens:")
        self.tokens_param.set_text('256')
        ToolTip(self.tokens_param.tk_entry, "The number of tokens the model will generate")

        self.temp_param = LabeledParam(f, 2, "Temp:")
        self.temp_param.set_text('0.7')
        ToolTip(self.temp_param.tk_entry, "The randomness of the response (0.0 - 1.0)")

        self.presence_param = LabeledParam(f, 4, "Presence penalty:")
        self.presence_param.set_text('0.3')
        ToolTip(self.presence_param.tk_entry, "Increases liklihood of new topics")


        self.frequency_param = LabeledParam(f, 6, "Frequency penalty:")
        self.frequency_param.set_text('0.3')
        ToolTip(self.frequency_param.tk_entry, "Supresses repeating text")
        return row + 1


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

    def get_gpt3_response(self, prompt:str) -> str:
        """
        Method that takes a prompt and gets the response back via the OpenAI API
        :param prompt: The prompt to be sent to the GPT-3
        :return: The GPT-3 Response
        """
        if len(prompt) < 3:
            self.dp.dprint("get_gpt3_response() Error. Prompt too short: '{}'".format(prompt))
            return ""

        # print(prompt)
        self.oai.max_tokens = self.tokens_param.get_as_int()
        self.oai.temperature = self.temp_param.get_as_float()
        self.oai.frequency_penalty = self.frequency_param.get_as_float()
        self.oai.presence_penalty = self.presence_param.get_as_float()
        self.oai.engine = self.generate_model_combo.get_text()

        results = self.oai.get_prompt_result(prompt, False)
        self.dp.dprint("\n------------\ntokens = {}, engine = {}\nprompt = {}".format(self.oai.max_tokens, self.oai.engine, prompt))
        self.log_action("gpt_prompt", {"tokens":self.oai.max_tokens, "engine":self.oai.engine, "prompt":prompt})

        # clean up before returning
        s = results[0].strip()
        self.dp.dprint("gpt_response: {}".format(s))
        self.log_action("gpt_response", {"gpt_text":s})
        return s

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

    def new_prompt_callback(self):
        print("GPT3GeneratorFrame.new_prompts_callback()")
        split_regex = re.compile(r"[\n]+")
        prompt = self.prompt_text_field.get_text()
        print("\tSending prompt ...{}".format(prompt[-80:]))
        response = self.get_gpt3_response(prompt)
        print("\tGot response {}...".format(response[:80]))
        l = split_regex.split(response)
        response = "\n".join(l)
        self.response_text_field.set_text(response)

    def extend_prompt_callback(self):
        prompt = "{} {}".format(self.prompt_text_field.get_text(), self.response_text_field.get_text())
        self.prompt_text_field.set_text(prompt)
        self.response_text_field.clear()

    def get_list(self, to_parse:str, regex_str:str = ",") -> List:
        rlist = re.split(regex_str, to_parse)
        to_return = []
        for t in rlist:
            if t != None:
                to_return.append(t.strip())
        to_return = [x for x in to_return if x] # filter out the blanks
        return to_return

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
            self.prompt_text_field.set_text("No prompt available in database")
            self.runs_field.set_text('0')
            self.count_parsed(self.experiment_id)
            if results[0]['max'] != None:
                run_id = results[0]['max']
                #self.runs_field.set_text(run_id)
                sql = "select * from run_params_view where run_id = %s and experiment_id = %s"
                vals = (run_id, self.experiment_id)
                results = self.msi.read_data(sql, vals)
                d = results[0]
                # safe_dict_read(self, d:Dict, key:str, default:Any) -> Any:
                self.prompt_text_field.set_text(self.safe_dict_read(d, 'prompt', self.prompt_text_field.get_text()))
                self.generate_model_combo.clear()
                self.generate_model_combo.set_text(self.safe_dict_read(d, 'generate_model', self.generate_model_combo.get_text()))
                self.tokens_param.set_text(self.safe_dict_read(d, 'tokens', self.tokens_param.get_text()))
                self.presence_param.set_text(self.safe_dict_read(d, 'presence_penalty', self.presence_param.get_text()))
                self.frequency_param.set_text(self.safe_dict_read(d, 'frequency_penalty', self.frequency_param.get_text()))
                self.embed_model_combo.clear()
                self.embed_model_combo.set_text(self.safe_dict_read(d, 'embedding_model', self.embed_model_combo.get_text()))
                self.pca_dim_param.set_text(self.safe_dict_read(d, 'PCA_dim', self.pca_dim_param.get_text()))
                self.eps_param.set_text(self.safe_dict_read(d, 'EPS', self.eps_param.get_text()))
                self.min_samples_param.set_text(self.safe_dict_read(d, 'min_samples', self.min_samples_param.get_text()))
                self.perplexity_param.set_text(self.safe_dict_read(d, 'perplexity', self.perplexity_param.get_text()))

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
        et:EmbeddedText
        for et in self.mr.embedding_list:
            reduced_s = np.array(et.reduced).dumps()
            sql = "update table_parsed_text set mapped = %s, cluster_id = %s where id = %s"
            vals = (reduced_s, int(et.cluster_id), int(et.row_id))

            self.msi.write_sql_values_get_row(sql, vals)

        self.count_parsed(self.experiment_id)


    def parse_response_callback(self):
        # get the regex
        split_regex = self.regex_field.get_text()

        # get the prompt and respnse text blocks
        self.saved_prompt_text = self.prompt_text_field.get_text()
        self.saved_response_text = self.response_text_field.get_text()
        full_text = self.saved_prompt_text + " " + self.saved_response_text

        # build the list of parsed text
        self.parsed_full_text_list = self.get_list(full_text, split_regex)
        # print(response_list)

        if len(self.parsed_full_text_list) > 1:
            count = 0
            for r in self.parsed_full_text_list:
                if len(r) > 1:
                    self.dp.dprint("line {}: {}".format(count, r))
                    count += 1
        else:
            message.showwarning("Parse Error",
                                "Could not parse [{}]".format(self.response_text_field.get_text()))

    def save_text_list_callback(self):
        print("save_text_list_callback")

        if self.experiment_id == -1:
            result = tk.messagebox.showwarning("Warning!", "Please create or select a database first")
            return

        if len(self.parsed_full_text_list) > 0:
            # create the run
            run_id = 1
            sql = "select MAX(run_id) as max from table_run where experiment_id = %s"
            vals = (self.experiment_id,)
            results = self.msi.read_data(sql, vals)
            print(results)
            if results[0]['max'] != None:
                run_id = results[0]['max'] + 1
            # get the language model params entry
            sql = "insert into table_generate_params (tokens, presence_penalty, frequency_penalty, model) values (%s, %s, %s, %s)"
            vals = (self.tokens_param.get_as_int(), self.presence_param.get_as_float(),
                    self.frequency_param.get_as_float(), self.generate_model_combo.get_text())
            lang_param_id = self.msi.write_sql_values_get_row(sql, vals)

            # get the embedding model entry
            sql = "insert into table_embedding_params (model, PCA_dim, EPS, min_samples, perplexity) values (%s, %s, %s, %s, %s)"
            vals = (self.embed_model_combo.get_text(), self.pca_dim_param.get_as_int(), self.eps_param.get_as_float(),
                    self.min_samples_param.get_as_int(), self.perplexity_param.get_as_int())
            embed_param_id = self.msi.write_sql_values_get_row(sql, vals)

            sql = "insert into table_run (experiment_id, run_id, prompt, response, generator_params, embedding_params) values (%s, %s, %s, %s, %s, %s)"
            vals = (self.experiment_id, run_id, self.saved_prompt_text,
                    self.saved_response_text, lang_param_id, embed_param_id)
            self.msi.write_sql_values_get_row(sql, vals)

            # store the text
            s:str
            for s in self.parsed_full_text_list:
                sql = "insert into table_parsed_text (run_index, parsed_text) values (%s, %s)"
                vals = (run_id, s)
                self.msi.write_sql_values_get_row(sql, vals)

        #reset the list
        self.parsed_full_text_list = []

    def automate_callback(self):
        print("automate_callback():")
        num_runs = self.auto_field.get_as_int()
        for i in range(num_runs):
            prompt = self.prompt_text_field.get_text()
            print("{}: prompting: {}".format(i, prompt))
            self.new_prompt_callback()
            response = self.response_text_field.get_text()
            print("\tgetting response: {}".format(response))
            print("\tparsing response")
            self.parse_response_callback()
            print("\tstoring data")
            self.save_text_list_callback()
            print("\tresetting")
            self.parsed_full_text_list = []
            self.response_text_field.clear()
        print("done")

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
            s_list.append(['parsed_text'])

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
            self.embed_state_text_field.insert_text("[{}] {}\n".format(id, text))


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
            et.set_optional(mapped, cluster_id, cluster_name)
            self.embed_state_text_field.insert_text("[{}] {}\n".format(id, et.text))
            print(et.to_string())
        self.mr.calc_clusters()


    def reduce_dimensions_callback(self):
        pca_dim = self.pca_dim_param.get_as_int()
        perplexity = self.perplexity_param.get_as_int()
        self.dp.dprint("Reducing: PCA dim = {}  perplexity = {}".format(pca_dim, perplexity))
        self.mr.calc_embeding(perplexity=perplexity, pca_components=pca_dim)
        self.reduced_field.set_text(len(self.mr.embedding_list))
        print("\tFinished dimension reduction")
        message.showinfo("reduce_dimensions_callback", "Reduced to {} dimensions".format(pca_dim))

    def cluster_callback(self):
        print("Clustering")
        eps = self.eps_param.get_as_float()
        min_samples = self.min_samples_param.get_as_int()
        self.mr.dbscan(eps=eps, min_samples=min_samples)
        self.mr.calc_clusters()
        self.clusters_field.set_text(str(len(self.mr.embedding_list)))
        self.dp.dprint("Finished clustering")

    def topic_callback(self):
        ci:ClusterInfo
        et:EmbeddedText
        split_regex = re.compile("\d+\)")

        for ci in self.mr.cluster_list:
            text_list = []
            for et in ci.member_list:
                text_list.append(et.text)
            prompt = "Extract keywords from this text:\n\n{}\n\nTop three keywords\n1)".format(" ".join(text_list))
            # print("\nCluster ID {} query text:\n{}".format(ci.id, prompt))
            result = self.oai.get_prompt_result_params(prompt, temperature=0.5, max_tokens=60, top_p=1.0, frequency_penalty=0.8, presence_penalty=0)
            l = split_regex.split(result)
            response = "".join(l)
            ci.label = "[{}] {}".format(ci.id, response)
            print("Cluster {}:\n{}".format(ci.id, response))
        self.dp.dprint("topic_callback complete")

    def plot_callback(self):
        print("Plotting")
        title = self.experiment_field.get_text()
        perplexity = self.perplexity_param.get_as_int()
        eps = self.eps_param.get_as_int()
        min_samples = self.min_samples_param.get_as_int()
        pca_dim = self.pca_dim_param.get_as_int()
        self.mr.plot("{}\ndim: {}, eps: {}, min_sample: {}, perplex = {}".format(
            title, pca_dim, eps, min_samples, perplexity))
        plt.show()

    def get_current_params(self) -> Dict:
        d = {
            "probe_str": self.prompt_text_field.get_text(),
            "name": self.experiment_field.get_text(),
            "automated_runs": self.auto_field.get_as_int(),
            "generate_model": self.generate_model_combo.get_text(),
            "tokens": self.tokens_param.get_as_int(),
            "temp": self.temp_param.get_as_float(),
            "presence_penalty": self.presence_param.get_as_float(),
            "frequency_penalty": self.frequency_param.get_as_float(),
            "embedding_model": self.embed_model_combo.get_text(),
            "PCA_dimensions": self.pca_dim_param.get_as_int(),
            "EPS": self.eps_param.get_as_float(),
            "min_samples": self.min_samples_param.get_as_int(),
            "perplexity": self.perplexity_param.get_as_int()
        }
        return d

    def load_params_callback(self):
        defaults = self.get_current_params()

        param_dict = self.load_json(defaults)
        # print(param_dict)

        self.prompt_text_field.clear()
        self.experiment_field.clear()
        self.auto_field.clear()
        self.tokens_param.clear()
        self.generate_model_combo.clear()
        self.temp_param.clear()
        self.presence_param.clear()
        self.frequency_param.clear()
        self.embed_model_combo.clear()
        self.pca_dim_param.clear()
        self.eps_param.clear()
        self.min_samples_param.clear()
        self.perplexity_param.clear()

        self.prompt_text_field.set_text(param_dict['probe_str'])
        self.experiment_field.set_text(param_dict['name'])
        self.auto_field.set_text(param_dict['automated_runs'])
        self.tokens_param.set_text(param_dict['tokens'])
        self.generate_model_combo.set_text(param_dict['generate_model'])
        self.temp_param.set_text(param_dict['temp'])
        self.presence_param.set_text(param_dict['presence_penalty'])
        self.frequency_param.set_text(param_dict['frequency_penalty'])
        self.embed_model_combo.set_text(param_dict['embedding_model'])
        self.pca_dim_param.set_text(param_dict['PCA_dimensions'])
        self.eps_param.set_text(param_dict['EPS'])
        self.min_samples_param.set_text(param_dict['min_samples'])
        self.perplexity_param.set_text(param_dict['perplexity'])

    def save_params_callback(self):
        params = self.get_current_params()
        self.save_experiment_json(params)

    # make this a "restore" button?
    def test_data_callback(self):
        prompt_text = '''Once upon a time there was a man who had been a soldier, and who had fought in the wars. After some years he became tired of fighting, and he stopped his soldiering and went away to live by himself in the mountains. He built a hut for himself, and there he lived for many years. At last one day there was a knocking at his door. He opened it and found no one there.

The next day, and the next, and the next after that there was a knocking at his door, but when he opened it no one was ever there.

At last he got so cross that he could not keep away from home any more than usual. When he opened the door and found no one there, he was so angry that he threw a great stone after whoever it was that knocked.

Presently a voice called out to him and said: “I am coming back soon again; you must be careful not to throw stones at me then”; but the voice did not say who it was that spoke.

The second time the man’s heart failed him as soon as he opened his door; but when he heard the voice saying: “Be careful not to throw stones this time,” he felt quite sure that'''
        response_text = '''it was the same voice. Then he knew that it was his Guardian Spirit that spoke to him.

The third time the man was not afraid, but as soon as he opened the door and saw no one, he threw stones at it.

Then a great storm arose and the thunder rolled among the mountains, and the lightning flashed in his eyes and blinded him, and all about him there were voices shouting: “It is your Guardian Spirit that you have killed!”

And when he could see again, he looked up and saw that the hut had disappeared and that in its place stood a dark pine-tree. He ran to look for his hut, but it was nowhere to be found; he looked up and down the valley, but there was no sign of it anywhere. He called out loudly for his hut to come back,—but it never came back again. The hut had become a big pine-tree, and even the Guardian Spirit could not make it come back again.'''
        self.prompt_text_field.set_text(prompt_text)
        self.response_text_field.set_text(response_text)


def main():
    app = NarrativeExplorer()
    app.mainloop()

if __name__ == "__main__":
    main()